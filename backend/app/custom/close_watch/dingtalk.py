"""Durable tail-alert delivery; HTTP runs only in the dedicated sender thread."""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from app import secrets_store
from app.market_time import CN_TZ, cn_now
from app.services import dingtalk_adapter, preferences
from app.services.fs_utils import atomic_write_text

logger = logging.getLogger(__name__)
TERMINAL = {'sent', 'failed', 'expired', 'cancelled', 'unknown'}


class CloseWatchDingTalk:
    def __init__(self, data_dir, context, *, sender=None):
        self.path = Path(data_dir) / 'user_data/extensions/close_watch/dingtalk.json'
        self.context = context
        self.sender = sender or dingtalk_adapter.send_text_result
        self.lock = threading.RLock()
        self.send_lock = threading.Lock()
        self.recovered = False

    def _load(self):
        if not self.path.exists():
            return {'version': 1, 'enabled': False, 'since_ms': None, 'records': [], 'seen': {}}
        state = json.loads(self.path.read_text(encoding='utf8'))
        if (not isinstance(state, dict) or state.get('version') != 1
                or type(state.get('enabled')) is not bool
                or not isinstance(state.get('records'), list)
                or not isinstance(state.get('seen'), dict)
                or (state.get('since_ms') is not None and type(state['since_ms']) is not int)):
            raise ValueError('尾盘钉钉发送状态不可用，请检查扩展数据')
        for record in state['records']:
            if (not isinstance(record, dict) or not {'id', 'event', 'status', 'queued_at'} <= record.keys()
                    or record['status'] not in TERMINAL | {'queued', 'sending'}
                    or not isinstance(record['event'], dict)):
                raise ValueError('尾盘钉钉发送记录不可用，请检查扩展数据')
        return state

    def _save(self, state):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Never evict pending work; completed records are for the UI, seen IDs dedupe separately.
        keep = {r['id'] for r in [r for r in state['records'] if r['status'] in TERMINAL][-200:]}
        state['records'] = [r for r in state['records'] if r['status'] not in TERMINAL or r['id'] in keep]
        atomic_write_text(self.path, json.dumps(state, ensure_ascii=False, allow_nan=False, indent=2))

    def status(self):
        configured = dingtalk_adapter.is_valid_url(preferences.get_dingtalk_webhook_url())
        try:
            with self.lock:
                state = self._load()
        except (ValueError, OSError):
            return {'enabled': False, 'configured': configured, 'status': 'blocked',
                    'message': '钉钉发送状态不可用，请检查扩展数据', 'pending': 0, 'records': []}
        records = [{k: v for k, v in r.items() if k != 'event'} | {
            'symbol': r['event'].get('symbol', ''), 'phase': r['event'].get('phase'),
            'type': r['event'].get('type'),
        } for r in reversed(state['records'])]
        pending = sum(r['status'] in {'queued', 'sending'} for r in records)
        message = '钉钉推送未启用'
        if state['enabled']:
            message = '复用已配置机器人，等待新尾盘信号' if configured else '钉钉机器人尚未配置'
        return {'enabled': state['enabled'], 'configured': configured, 'status': 'ready',
                'message': message, 'pending': pending, 'records': records}

    def configure(self, enabled, *, now=None):
        if type(enabled) is not bool:
            raise ValueError('钉钉启用状态必须为布尔值')
        if enabled and not dingtalk_adapter.is_valid_url(preferences.get_dingtalk_webhook_url()):
            raise ValueError('请先在设置中配置钉钉机器人')
        now = now or cn_now()
        with self.lock:
            state = self._load()
            if enabled and not state['enabled']:
                state['since_ms'] = int(now.timestamp()*1000)
            if not enabled:
                for record in state['records']:
                    if record['status'] == 'queued':
                        record.update(status='cancelled', detail='钉钉推送已暂停')
            state['enabled'] = enabled
            self._save(state)
        return self.status()

    def enqueue(self, events, *, now=None):
        now = now or cn_now()
        with self.lock:
            state = self._load()
            if not state['enabled']:
                return
            for event in events:
                if (event['ts'] < (state['since_ms'] or 0) or event['id'] in state['seen']
                        or event['day'] != str(now.date())):
                    continue
                state['seen'][event['id']] = event['day']
                state['records'].append({'id': event['id'], 'event': dict(event),
                    'queued_at': int(now.timestamp()*1000), 'status': 'queued', 'detail': '等待发送'})
            # Retain IDs across restart without retaining an unlimited event history.
            state['seen'] = {k: v for k, v in state['seen'].items() if v >= str(now.date())}
            self._save(state)

    def enqueue_test(self, *, now=None):
        now = now or cn_now()
        context = self.context()
        with self.lock:
            state = self._load()
            if not state['enabled']:
                raise ValueError('请先启用尾盘钉钉推送')
            if any(r['event'].get('phase') == 'test' and
                   int(now.timestamp()*1000)-r['queued_at'] < 60000 for r in state['records']):
                raise ValueError('一分钟内只发送一条连接测试，请查看发送记录')
            event = {'id': 'test:'+uuid.uuid4().hex, 'ts': int(now.timestamp()*1000),
                'day': str(now.date()), 'phase': 'test', 'type': 'connection_test',
                'symbol': '', 'message': '这是钉钉接入测试，不是买卖信号。尾盘检查开始时间：'
                + str(context.get('start_time') or '未绑定'),
                'strategy_id': context.get('strategy_id'), 'strategy_revision': context.get('revision')}
            self.enqueue([event], now=now)

    @staticmethod
    def _message(event):
        phase = event['phase']
        if phase == 'test':
            return 'TickFlow 股票尾盘监控连接测试', event['message']
        title = '尾盘买入预警' if event['type'] == 'buy_signal' else '尾盘卖出预警'
        if phase == 'confirmed':
            title = '尾盘预警未获确认' if event['type'] == 'pool_exit' else '尾盘信号收盘确认'
        stamp = datetime.fromtimestamp(event['ts']/1000, CN_TZ).strftime('%Y-%m-%d %H:%M:%S')
        price = event.get('price')
        body = f"{event.get('symbol', '')} {event.get('name', '')}\n{event['message']}\n信号时间：{stamp}（北京时间）"
        if price is not None:
            body += f'\n信号参考价：{float(price):.2f}'
        body += ('\n收盘后复核消息，当日已收盘，不是当日可执行的买入指令。'
                 if phase == 'confirmed' else '\n尾盘预警可能变化，请留意后续确认或取消。')
        return 'TickFlow 股票'+title, body

    def tick(self, *, now=None):
        if not self.send_lock.acquire(blocking=False):
            return
        try:
            self._tick(now or cn_now())
        finally:
            self.send_lock.release()

    def _tick(self, now):
        # Read monitor state before taking the delivery lock; config uses the opposite service.
        context = self.context()
        with self.lock:
            state = self._load()
            if not self.recovered:
                for row in state['records']:
                    if row['status'] == 'sending':
                        row.update(status='unknown', detail='上次发送被中断，结果未知；未自动补发以免重复')
                self.recovered = True
                self._save(state)
            if not state['enabled']:
                return
            record = next((r for r in state['records'] if r['status'] == 'queued'), None)
            if record is None:
                return
            event = record['event']
            age = now.timestamp()-event['ts']/1000
            expired = event['day'] != str(now.date()) or age > (120 if event['phase'] == 'preview' else 600)
            expired |= event['phase'] == 'preview' and now.hour >= 15
            if expired:
                record.update(status='expired', detail='信号已过发送时限，未补发过时提醒')
                self._save(state)
                return
            if event['phase'] != 'test' and (not context.get('enabled') or
                    event.get('strategy_id') != context.get('strategy_id') or
                    event.get('strategy_revision') != context.get('revision')):
                record.update(status='cancelled', detail='监控暂停或策略绑定已变化，未发送旧信号')
                self._save(state)
                return
            record.update(status='sending', attempted_at=int(now.timestamp()*1000), detail='正在发送')
            self._save(state)
        started = time.perf_counter()
        try:
            title, body = self._message(event)
            result = self.sender(preferences.get_dingtalk_webhook_url(), title, body,
                                 secrets_store.get_dingtalk_webhook_secret(), max_attempts=1)
            ok, detail = result.ok, result.detail
        except Exception:  # noqa: BLE001 -- channel isolation; do not log credential-bearing errors
            ok, detail = False, '钉钉发送失败，请检查网络及机器人配置'
            logger.warning('尾盘钉钉发送失败，请查看渠道状态')
        with self.lock:
            state = self._load()
            saved = next(r for r in state['records'] if r['id'] == record['id'])
            elapsed = round((time.perf_counter()-started)*1000)
            saved.update(status='sent' if ok else 'failed', detail='钉钉已接收' if ok else detail,
                         elapsed_ms=elapsed, completed_at=int(now.timestamp()*1000)+elapsed)
            self._save(state)

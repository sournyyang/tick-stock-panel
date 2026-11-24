"""Reuse native strategy matrices, with explicit original-signal binding and phases."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
from dataclasses import replace
from datetime import datetime, time
from pathlib import Path

import numpy as np

from app.backtest.matrix import MatrixPipelineConfig, MatrixStrategyPipeline
from app.market_time import CN_TZ, cn_now
from app.services import alert_store
from app.services.fs_utils import atomic_write_text
from app.services.monitor_watchlist import MonitorWatchlistSync
from app.strategy import config as strategy_config
from app.strategy.engine import StrategyDataContext
from app.strategy.scoring import effective_scoring, effective_scoring_directions

logger = logging.getLogger(__name__)
LABELS = {'signal_close_below_ma3': '收盘低于3日均线',
          'signal_triple_volume_breakout': '三倍量整理反弹买入'}


def watch_phase(now: datetime, start: str) -> str | None:
    now = now.astimezone(CN_TZ)
    if now.weekday() >= 5 or now.time() < time.fromisoformat(start):
        return None
    return 'preview' if now.time() < time(15) else 'confirmed'


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def _price(value):
    return float(value) if value is not None and math.isfinite(value) and value > 0 else None


class CloseWatchService:
    def __init__(self, data_dir, engine, screener, quotes, notifications=None, monitor_rules=None):
        self.data_dir = Path(data_dir)
        self.path = self.data_dir / 'user_data/extensions/close_watch/state.json'
        self.engine, self.screener, self.quotes = engine, screener, quotes
        self.lock = threading.RLock()
        self.eval_lock = threading.Lock()
        self.stopping = threading.Event()
        self.notifications = notifications
        self.monitor_rules = monitor_rules or (lambda: {})
        self.watchlist_sync = MonitorWatchlistSync(self._active_watchlist_rules,
            lambda: self.quotes.notify_watchlist_updated() if self.quotes else None)

    def _load(self):
        if not self.path.exists():
            return {'version':1, 'generation':0, 'enabled':False, 'binding':None,
                    'events':[], 'seen':{}, 'outbox':[], 'tracks':[], 'seeded':False,
                    'candidates':[], 'watch_status':[], 'status':'paused',
                    'message':'尚未启用尾盘监控', 'phase':None}
        state = json.loads(self.path.read_text(encoding='utf8'))
        if (not isinstance(state, dict) or state.get('version') != 1
                or type(state.get('enabled')) is not bool
                or type(state.get('generation')) is not int
                or type(state.get('seeded')) is not bool
                or not isinstance(state.get('seen'), dict)
                or any(not isinstance(state.get(k), list)
                       for k in ['events','outbox','tracks','candidates','watch_status'])):
            raise ValueError('尾盘监控状态版本或内容不可用，请检查扩展数据')
        binding = state.get('binding')
        if binding is not None and (not isinstance(binding, dict) or not
                {'strategy_id','revision','start_time','max_hold_days','name'} <= binding.keys()):
            raise ValueError('尾盘监控策略绑定内容不可用，请检查扩展数据')
        if state['enabled'] and binding is None:
            raise ValueError('尾盘监控缺少策略绑定，请重新配置')
        if any(not isinstance(t, dict) or not {'id','symbol','signal_date'} <= t.keys()
               for t in state['tracks']):
            raise ValueError('自动信号观察内容不可用，请检查扩展数据')
        return state

    def _save(self, state):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.path, json.dumps(state, ensure_ascii=False, allow_nan=False, indent=2))

    def get(self):
        with self.lock:
            result = {k:v for k,v in self._load().items() if k not in {'seen','outbox'}}
        if self.notifications:
            result['dingtalk'] = self.notifications.status()
        rules = self._active_watchlist_rules()
        warnings = [self.watchlist_sync.warning_for(rid) for rid in rules]
        result['watchlist'] = {'enabled': bool(rules),
            'message': '已确认买点自动加入策略分组，结束观察后保留自选' if rules else
                       '自动加入自选未启用，请在同策略普通监控规则中开启',
            'warning': next((w for w in warnings if w), None)}
        return result

    def _active_watchlist_rules(self):
        with self.lock:
            state = self._load()
        if not state['enabled'] or not state['binding'] or self.stopping.is_set():
            return {}
        sid = state['binding']['strategy_id']
        return {rid: rule for rid, rule in self.monitor_rules().items()
                if rule.get('type') == 'strategy' and rule.get('enabled', True)
                and rule.get('auto_watchlist') and rule.get('asset_type', 'stock') == 'stock'
                and rule.get('strategy_id') == sid}

    def _sync_watchlist(self):
        # The durable observations recover missed additions even before today's tail window.
        # Queue outside the monitor lock; the shared worker rechecks enabled rules before IO.
        rules = self._active_watchlist_rules()
        with self.lock:
            state = self._load()
        if not rules or not state['enabled'] or not state['binding']:
            return
        self.watchlist_sync.submit([{'rule': rule,
            'rules_version': (state['generation'], state['binding']['revision']),
            'strategy_name': state['binding']['name'],
            'symbols': frozenset(t['symbol'] for t in state['tracks'])} for rule in rules.values()])

    def notification_context(self):
        with self.lock:
            state = self._load()
            return {'enabled': state['enabled'], **(state['binding'] or {})}

    def configure_dingtalk(self, enabled):
        self.notifications.configure(enabled)
        return self.get()

    def test_dingtalk(self):
        self.notifications.enqueue_test()
        return self.get()

    def _definition(self, sid):
        strategy = self.engine.get(sid)
        if (strategy.execution_backend != 'matrix_native'
                or 'stock' not in strategy.meta.get('asset_types', ['stock'])
                or '1d' not in strategy.meta.get('timeframes', ['1d'])):
            raise ValueError('尾盘监控当前支持股票日线矩阵策略')
        requirements = strategy.meta.get('backtest_requirements') or {}
        if any(requirements.get(k, strategy.meta.get('default_'+k, 'open_t+1')) != 'open_t+1'
               for k in ['entry_fill','exit_fill']):
            raise ValueError('该尾盘模块仅支持买卖次日开盘口径的日线策略')
        overrides = strategy_config.load_override(self.data_dir, sid)
        if overrides.get('overlay_filter'):
            raise ValueError('当前尾盘信号观察暂不支持额外叠加条件，请使用原始策略定义')
        risk = ['stop_loss','take_profit','trailing_stop','trailing_take_profit_activate',
                'trailing_take_profit_drawdown']
        if any(overrides.get(k, getattr(strategy, k, None)) is not None for k in risk):
            raise ValueError('该策略含额外风控，尾盘模块当前仅覆盖策略卖点、绑定防守线及到期提醒')
        if not strategy.file_path:
            raise ValueError('策略缺少可校验的源文件')
        params = self.engine.resolve_params(strategy, overrides=overrides)
        limit = getattr(strategy.matrix_strategy, 'entry_candidate_limit', None)
        if not callable(limit) or limit(params) != 1:
            raise ValueError('当前自动尾盘模块支持策略自身日选1只的矩阵策略')
        basic = {**strategy.basic_filter, **(overrides.get('basic_filter') or {})}
        cfg = MatrixPipelineConfig(basic_filter=basic,
            scoring=effective_scoring(strategy.meta.get('scoring'), overrides),
            scoring_directions=effective_scoring_directions(overrides),
            order_by=strategy.meta.get('order_by'), descending=bool(strategy.meta.get('descending', True)))
        revision = _digest({'code':hashlib.sha256(strategy.file_path.read_bytes()).hexdigest(),
                            'meta':strategy.meta, 'overrides':overrides})
        hold = overrides.get('max_hold_days', strategy.max_hold_days)
        return strategy, params, overrides, cfg, revision, hold

    def configure(self, strategy_id, enabled, start_time):
        with self.lock:
            if type(enabled) is not bool:
                raise ValueError('启用状态必须为布尔值')
            try:
                start = time.fromisoformat(start_time)
            except (TypeError, ValueError) as exc:
                raise ValueError('尾盘开始时间格式应为HH:MM') from exc
            if not time(14,30) <= start < time(15) or len(start_time) != 5:
                raise ValueError('尾盘开始时间须在14:30至14:59之间')
            strategy, _, _, _, revision, hold = self._definition(strategy_id)
            state = self._load()
            old = state.get('binding') or {}
            if old.get('revision') != revision:
                state.update(tracks=[], seeded=False, candidates=[], watch_status=[])
            state.update(enabled=enabled, status='waiting' if enabled else 'paused',
                message='等待交易日尾盘时段' if enabled else '尾盘监控已暂停', phase=None,
                binding={'strategy_id':strategy_id, 'name':strategy.meta.get('name', strategy_id),
                    'revision':revision, 'start_time':start_time, 'max_hold_days':hold,
                    'max_positions':strategy.meta.get('backtest_defaults', {}).get('max_positions', 2)})
            state['generation'] += 1
            self._save(state)
            return self.get()

    def _market(self, state, frame, day, params, overrides):
        sid = state['binding']['strategy_id']
        key = 'close-watch:' + sid
        from app.services.daily_health import revision

        generation = revision(self.data_dir, 'stock')
        context = StrategyDataContext('stock','1d',day,current=frame,cache_key=key,data_generation=generation)
        kwargs = {'params_map':{sid:params}, 'overrides_map':{sid:overrides}}
        try:
            return self.engine.prepare_realtime_matrix(context, [sid], **kwargs)
        except ValueError as exc:
            if 'history' not in str(exc):
                raise
        context = self.screener.build_strategy_context(self.engine, day, [sid],
            current=frame, cache_key=key, **kwargs)
        context = replace(context, data_generation=generation)
        return self.engine.prepare_realtime_matrix(context, [sid], **kwargs)

    def _wait(self, state, message, status='waiting', phase=None):
        if any(state.get(k) != v for k,v in {'message':message,'status':status,'phase':phase}.items()):
            state.update(message=message,status=status,phase=phase)

    def tick(self, now=None):
        now = (now or cn_now()).astimezone(CN_TZ)
        if not self.eval_lock.acquire(blocking=False):
            return
        try:
            with self.lock:
                state = self._load()
            if not state['enabled'] or self.stopping.is_set():
                return
            before = json.dumps(state, sort_keys=True)
            generation = state['generation']
            phase = watch_phase(now, state['binding']['start_time'])
            if phase is None:
                self._wait(state, '等待交易日尾盘时段')
            elif not (phase == 'confirmed' and state.get('phase') == 'confirmed'
                      and state.get('as_of') == str(now.date()) and state.get('status') == 'running'):
                try:
                    self._evaluate(state, now, phase)
                except (ValueError, KeyError, TypeError, OSError) as exc:
                    message = str(exc) if isinstance(exc, ValueError) else '数据或存储暂不可用，等待下轮检查'
                    self._wait(state, message, 'blocked', phase)
            # Heavy strategy/history work runs outside the state lock. An API
            # edit made during it wins; never overwrite a newer observation/config.
            with self.lock:
                if self._load()['generation'] != generation or self.stopping.is_set():
                    return
                if json.dumps(state, sort_keys=True) != before:
                    self._save(state)
            self._sync_watchlist()
            if phase is not None and state['status'] != 'blocked':
                self._drain(now)
        finally:
            self.eval_lock.release()

    @staticmethod
    def _track(symbol, signal_date, name):
        return {'id':_digest([symbol,signal_date])[:16], 'symbol':symbol,
                'signal_date':signal_date, 'name':name or symbol}

    @staticmethod
    def _reasons(track, market, signals, t, dates, assets, hold):
        symbol = track['symbol']
        a = assets.get(symbol)
        result = {'reasons':[], 'floor_price':None, 'signal_age':None, 'message':'等待有效行情'}
        if a is None:
            return result
        close = _price(market.close[t,a])
        raw = _price(market.field('raw_close')[t,a])
        if close is None or raw is None:
            return result
        result.update(price=raw,message='未触发卖点')
        signal_date = track['signal_date']
        if signal_date in dates:
            source = dates.index(signal_date)
            age = t-source
            result['signal_age'] = age
            # Show the signal's line immediately, without a same-day round trip.
            if age <= 0:
                if signals.entry_close_floor is not None:
                    floor = _price(signals.entry_close_floor[source,a])
                    if floor is not None:
                        result['floor_price'] = floor*raw/close
                return result
            if signals.exit[t,a]:
                result['reasons'].append('收盘低于3日均线' if 'signal_close_below_ma3' in signals.exit_signal_ids else '策略卖出条件')
            if signals.entry_close_floor is not None and signals.entry[source,a]:
                floor = _price(signals.entry_close_floor[source,a])
                if floor is not None:
                    result['floor_price'] = floor*raw/close
                    if close < floor:
                        result['reasons'].append('低于原买点绑定防守线')
                else:
                    result['message'] = '原买点防守线暂不可计算'
            else:
                result['message'] = '原买点历史待核对，未猜测防守线'
            if hold is not None and age >= hold:
                result['reasons'].append(f'买点后已过{hold}个市场交易日，达到观察期限')
        else:
            result['message'] = '原买点超出可用历史窗口，防守线待核对'
            if hold is not None and signal_date < dates[0] and t+1 >= hold:
                result['reasons'].append(f'买点后至少已过{hold}个市场交易日，达到观察期限')
        return result

    def _seed_tracks(self, market, signals, t, dates, assets, hold):
        # Replay indicators only, with no prices/fills/cash/portfolio simulation.
        # Past exits end past observations; no historical notification is sent.
        active = {}
        for k in range(t):
            ended = set()
            for symbol, track in list(active.items()):
                if self._reasons(track,market,signals,k,dates,assets,hold)['reasons']:
                    active.pop(symbol)
                    ended.add(symbol)
            for a in np.flatnonzero(signals.entry[k]):
                symbol = market.symbols[int(a)]
                if symbol not in active and symbol not in ended:
                    active[symbol] = self._track(symbol,dates[k],market.names[int(a)])
        return list(active.values())

    def _evaluate(self, state, now, phase):
        strategy, params, overrides, cfg, revision, hold = self._definition(state['binding']['strategy_id'])
        if revision != state['binding']['revision']:
            raise ValueError('策略代码或参数已变化，尾盘监控暂停评估，请重新核对绑定')
        if self.quotes is None:
            raise ValueError('行情服务未就绪')
        status = self.quotes.status()
        if status.get('paused') or status.get('enabled') is False:
            self._wait(state,'行情服务已暂停，等待恢复',phase=phase)
            return
        if phase == 'confirmed' and (not status.get('final_sync_done') or status.get('final_sync_failed')):
            self._wait(state,'等待当日收盘行情确认',phase=phase)
            return
        age = status.get('quote_age_ms')
        if status.get('mode') != 'full_market' or age is None or age < 0 or (phase == 'preview' and age > 180000):
            self._wait(state,'全市场实时行情未就绪或已过期',phase=phase)
            return
        frame, day = self.quotes.get_enriched_today()
        if frame.is_empty() or day != now.date() or 'quote_ts' not in frame.columns:
            self._wait(state,'缺少当日行情或行情时间戳，等待有效数据',phase=phase)
            return
        if frame['date'].min() != day or frame['date'].max() != day:
            raise ValueError('行情日期混杂，尾盘监控等待一致的当日快照')
        market = self._market(state,frame,day,params,overrides)
        if market is None:
            raise ValueError('策略历史矩阵尚未就绪')
        signals = MatrixStrategyPipeline().run(strategy.matrix_strategy,market,params,cfg)
        if self._definition(state['binding']['strategy_id'])[4] != revision:
            raise ValueError('策略在计算期间发生变化，尾盘结果已丢弃，请重新核对绑定')
        dates = [d[:10] for d in market.timestamp_labels]
        if str(day) not in dates:
            raise ValueError('策略矩阵不含当日数据')
        t = dates.index(str(day))
        rows = {str(r['symbol']):r for r in frame.iter_rows(named=True)}
        assets = {s:a for a,s in enumerate(market.symbols)}
        boundary = datetime.combine(day,time(15),tzinfo=CN_TZ).timestamp()*1000
        minimum = now.timestamp()*1000-180000 if phase=='preview' else boundary-5000

        def fresh(symbol):
            row = rows.get(symbol,{})
            ts = row.get('quote_ts')
            return (isinstance(ts,(int,float)) and math.isfinite(ts) and minimum <= ts <= now.timestamp()*1000+5000
                    and _price(row.get('raw_close')) is not None)

        if not any(fresh(symbol) for symbol in rows):
            self._wait(state,'行情尚未到达本轮尾盘或收盘确认边界',phase=phase)
            return
        if not state['seeded']:
            state['tracks'] = self._seed_tracks(market,signals,t,dates,assets,hold)
            state['seeded'] = True
            state['seeded_on'] = str(day)
        events, candidates, checked, remaining = [], [], [], []
        exiting = set()
        for track in state['tracks']:
            symbol = track['symbol']
            item = {**track,'price':None,'floor_price':None,'signal_age':None,
                    'reasons':[],'message':'该标的行情缺失或过期，未判断卖点','ended':False}
            checked.append(item)
            if fresh(symbol):
                item.update(self._reasons(track,market,signals,t,dates,assets,hold))
                if item['reasons']:
                    exiting.add(symbol)
                    events.append(self._event(state,now,phase,symbol,track['name'],'sell_signal',
                        track['id']+':exit','；'.join(item['reasons']),rows[symbol]['raw_close']))
                    item['ended'] = phase=='confirmed'
            if not item['ended']:
                remaining.append(track)
        state['tracks'] = remaining
        active = {x['symbol'] for x in remaining}
        stale_candidates = set()
        for a in np.flatnonzero(signals.entry[t]):
            symbol = market.symbols[int(a)]
            if not fresh(symbol):
                stale_candidates.add(symbol)
                continue
            note = '原观察出现卖点，卖点优先' if symbol in exiting else '已在信号观察中，不重置原防守线' if symbol in active else ''
            row = {'symbol':symbol,'name':market.names[int(a)] or symbol,'price':rows[symbol]['raw_close'],
                   'signal_date':str(day),'score':float(signals.score[t,a]),'note':note}
            candidates.append(row)
            if symbol in exiting:
                continue
            events.append(self._event(state,now,phase,symbol,row['name'],'buy_signal','entry',
                '当日买入候选'+('；'+note if note else ''),row['price']))
            if phase=='confirmed' and symbol not in active:
                track = self._track(symbol,str(day),row['name'])
                state['tracks'].append(track)
                active.add(symbol)
                checked.append({**track,**self._reasons(track,market,signals,t,dates,assets,hold),
                    'message':'新收盘买点，开始自动观察；不代表已买入','ended':False})
        if phase=='confirmed':
            final_symbols = {x['symbol'] for x in candidates}
            for previous in state['events']:
                if (previous['day']!=str(day) or previous['phase']!='preview'
                        or previous.get('strategy_revision')!=revision or not fresh(previous['symbol'])):
                    continue
                symbol = previous['symbol']
                if previous['type']=='buy_signal' and symbol not in final_symbols:
                    events.append(self._event(state,now,phase,symbol,previous['name'],'pool_exit',
                        'cancelled_buy','尾盘买入预警未获收盘确认',rows[symbol]['raw_close']))
                if previous['type']=='sell_signal' and any(x['symbol']==symbol and not x['reasons'] for x in checked):
                    events.append(self._event(state,now,phase,symbol,previous['name'],'pool_exit',
                        'cancelled_sell','尾盘卖出预警未获收盘确认',rows[symbol]['raw_close']))
        state.update(candidates=candidates,watch_status=checked,as_of=str(day),phase=phase,
            checked_at=now.isoformat(),status='running',
            message='当天尾盘信号已检查；预警仍待收盘确认' if phase=='preview' else '当天收盘信号已确认；不代表实际成交')
        missing = len(stale_candidates | {x['symbol'] for x in checked if x.get('price') is None})
        if missing:
            state.update(status='waiting',message=state['message']+f'；{missing}个观察标的行情缺失，相关条件未确认')
        state['coverage'] = {'observed_symbols':len(rows),'reported_symbols':status.get('symbol_count')}
        state['seen'] = {k:v for k,v in state['seen'].items() if v==str(day)}
        for event in events:
            if event['id'] in state['seen']:
                continue
            state['seen'][event['id']] = str(day)
            state['outbox'].append(event)
            state['events'].insert(0,event)
        state['events'] = state['events'][:200]

    def _event(self, state, now, phase, symbol, name, event_type, reason, text, price):
        tag = '尾盘预警' if phase=='preview' else '收盘确认'
        execution = '，当天信号提醒；不代表已买入或已卖出'
        if event_type=='pool_exit':
            execution = '，不作为已确认卖出条件' if '卖出' in text else '，不作为已确认买点'
        return {'id':_digest([str(now.date()),phase,event_type,symbol,reason,state['binding']['revision']]),
            'ts':int(now.timestamp()*1000),'day':str(now.date()),'phase':phase,
            'source':'strategy','rule_id':'close_watch','strategy_id':state['binding']['strategy_id'],
            'strategy_revision':state['binding']['revision'],
            'type':event_type,'symbol':symbol,'name':name,'price':float(price),'change_pct':0.,
            'severity':'warn' if event_type=='sell_signal' else 'info','signals':[text],
            'message':f'策略「{state["binding"]["name"]}」{tag}：{name} {text}{execution}'}

    def _drain(self, now):
        with self.lock:
            state = self._load()
            generation = state['generation']
            pending = [x for x in state['outbox'] if x['day']==str(now.date())
                       and x.get('strategy_revision')==state['binding']['revision']]
        if not pending:
            return
        # Persisted outbox + stored IDs recover the append/checkpoint crash window.
        existing = {x.get('id') for x in alert_store.list_recent(self.data_dir, days=30, source='strategy')}
        with self.lock:
            state = self._load()
            if state['generation'] != generation or not state['enabled'] or self.stopping.is_set():
                return
            new = [x for x in pending if x['id'] not in existing]
            if new:
                alert_store.append_many(self.data_dir,new)
                if self.quotes:
                    self.quotes.push_alerts(new)
            if self.notifications:
                # Queue ALL pending IDs: recover a crash after local append but before enqueue.
                self.notifications.enqueue(pending, now=now)
            state['outbox'] = []
            self._save(state)

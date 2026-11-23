from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from app.custom.close_watch.dingtalk import CloseWatchDingTalk
from app.market_time import CN_TZ
from app.services.dingtalk_adapter import SendResult

NOW = datetime(2026, 11, 23, 14, 40, tzinfo=CN_TZ)


@pytest.fixture
def delivery(tmp_path, monkeypatch):
    monkeypatch.setattr('app.services.preferences.get_dingtalk_webhook_url',
                        lambda: 'https://oapi.dingtalk.com/robot/send?access_token=test-token')
    monkeypatch.setattr('app.secrets_store.get_dingtalk_webhook_secret', lambda: 'SEC-test-secret')
    context = {'enabled': True, 'strategy_id': 'demo', 'revision': 'rev'}
    sent = []

    def send(*args, **kwargs):
        sent.append((args, kwargs))
        return SendResult(True, 'accepted')

    service = CloseWatchDingTalk(tmp_path, lambda: dict(context), sender=send)
    service.configure(True, now=NOW)
    return SimpleNamespace(service=service, root=tmp_path, context=context, sent=sent, sender=send)


def event(kind='buy_signal', phase='preview', suffix='', now=NOW):
    return {'id': kind+phase+suffix, 'ts': int(now.timestamp()*1000), 'day': str(now.date()),
            'type': kind, 'phase': phase, 'symbol': '300885.SZ', 'name': '示例股票',
            'price': 15.23, 'message': '测试事件，不代表实际成交',
            'strategy_id': 'demo', 'strategy_revision': 'rev'}


@pytest.mark.parametrize('kind,phase', [
    ('buy_signal', 'preview'), ('sell_signal', 'preview'),
    ('buy_signal', 'confirmed'), ('sell_signal', 'confirmed'),
    ('pool_exit', 'confirmed'),
])
def test_each_phase_is_delivered_once_without_credentials_in_status(delivery, kind, phase):
    now = NOW if phase == 'preview' else NOW.replace(hour=15, minute=1)
    item = event(kind, phase, now=now)
    delivery.service.enqueue([item], now=now)
    delivery.service.enqueue([item], now=now)
    delivery.service.tick(now=now)
    delivery.service.tick(now=now)
    assert len(delivery.sent) == 1
    args, kwargs = delivery.sent[0]
    assert '300885.SZ' in args[2] and '15.23' in args[2]
    assert args[3] == 'SEC-test-secret' and kwargs['max_attempts'] == 1
    state = delivery.service.status()
    assert state['records'][0]['status'] == 'sent'
    assert 'test-token' not in str(state) and 'SEC-test-secret' not in str(state)
    assert 'test-token' not in delivery.service.path.read_text()
    if phase == 'confirmed':
        assert '收盘后复核' in args[2]


def test_enabling_does_not_replay_old_signals(delivery):
    delivery.service.enqueue([event(now=NOW-timedelta(seconds=1))], now=NOW)
    delivery.service.tick(now=NOW)
    assert delivery.sent == []


def test_restart_deduplicates_successful_delivery(delivery):
    item = event()
    delivery.service.enqueue([item], now=NOW)
    delivery.service.tick(now=NOW)
    restarted = CloseWatchDingTalk(delivery.root, lambda: dict(delivery.context), sender=delivery.sender)
    restarted.enqueue([item], now=NOW)
    restarted.tick(now=NOW)
    assert len(delivery.sent) == 1


def test_sending_interrupted_by_process_restart_is_not_blindly_repeated(delivery):
    delivery.service.enqueue([event()], now=NOW)
    state = delivery.service._load()
    state['records'][0]['status'] = 'sending'
    delivery.service._save(state)
    restarted = CloseWatchDingTalk(delivery.root, lambda: dict(delivery.context), sender=delivery.sender)
    restarted.tick(now=NOW)
    assert delivery.sent == []
    assert restarted.status()['records'][0]['status'] == 'unknown'


def test_stale_preview_and_wrong_strategy_are_not_sent(delivery):
    delivery.service.enqueue([event()], now=NOW)
    delivery.service.tick(now=NOW.replace(hour=15, minute=0))
    assert delivery.sent == []
    assert delivery.service.status()['records'][0]['status'] == 'expired'
    delivery.service.enqueue([event(suffix='new')], now=NOW)
    delivery.context['revision'] = 'changed'
    delivery.service.tick(now=NOW)
    assert delivery.sent == []
    assert delivery.service.status()['records'][0]['status'] == 'cancelled'


def test_failure_is_visible_and_does_not_duplicate_or_expose_exception(delivery):
    def broken(*args, **kwargs):
        raise RuntimeError('secret-token in URL')

    delivery.service.sender = broken
    delivery.service.enqueue([event()], now=NOW)
    delivery.service.tick(now=NOW)
    state = delivery.service.status()
    assert state['records'][0]['status'] == 'failed'
    assert '失败' in state['records'][0]['detail']
    assert 'secret-token' not in str(state)
    delivery.service.sender = delivery.sender
    delivery.service.tick(now=NOW)
    assert delivery.sent == []


def test_disable_cancels_queue_and_test_messages_are_explicit_and_rate_limited(delivery):
    delivery.service.enqueue([event()], now=NOW)
    delivery.service.configure(False, now=NOW)
    delivery.service.tick(now=NOW)
    assert delivery.sent == []
    delivery.service.configure(True, now=NOW)
    delivery.service.enqueue_test(now=NOW)
    with pytest.raises(ValueError, match='一分钟'):
        delivery.service.enqueue_test(now=NOW)
    delivery.service.tick(now=NOW)
    assert len(delivery.sent) == 1
    assert '测试' in delivery.sent[0][0][1]
    assert '不是买卖信号' in delivery.sent[0][0][2]


def test_no_webhook_refuses_enable_and_bad_state_fails_closed(delivery, monkeypatch):
    monkeypatch.setattr('app.services.preferences.get_dingtalk_webhook_url', lambda: '')
    with pytest.raises(ValueError, match='配置'):
        delivery.service.configure(True, now=NOW)
    delivery.service.path.write_text('{"version":99}')
    assert delivery.service.status()['status'] == 'blocked'
    with pytest.raises(ValueError):
        delivery.service.configure(False, now=NOW)


def test_slow_sender_does_not_hold_state_or_enqueue_lock(delivery):
    import threading

    entered, release, configured = threading.Event(), threading.Event(), threading.Event()

    def slow(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return SendResult(True, 'accepted')

    delivery.service.sender = slow
    delivery.service.enqueue([event()], now=NOW)
    thread = threading.Thread(target=lambda: delivery.service.tick(now=NOW))
    thread.start()
    try:
        assert entered.wait(1)
        assert delivery.service.status()['records'][0]['status'] == 'sending'
        delivery.service.enqueue([event(suffix='next')], now=NOW)
        other = threading.Thread(target=lambda: (delivery.service.configure(False, now=NOW), configured.set()))
        other.start()
        assert configured.wait(1), 'configuration must not wait for network'
        other.join(1)
    finally:
        release.set()
        thread.join(3)
    statuses = {r['status'] for r in delivery.service.status()['records']}
    assert statuses == {'sent', 'cancelled'}

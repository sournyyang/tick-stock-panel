from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace

import polars as pl
import pytest

from app.backtest.matrix import build_market_data_matrix
from app.custom.close_watch.service import CloseWatchService, watch_phase
from app.market_time import CN_TZ
from app.services import alert_store
from app.strategy.engine import StrategyEngine

DAY = date(2026, 11, 20)
NOW = datetime(2026, 11, 20, 14, 51, tzinfo=CN_TZ)

@pytest.mark.parametrize('now,expected', [
    (NOW.replace(hour=14,minute=49),None),(NOW,'preview'),
    (NOW.replace(hour=15,minute=0),'confirmed'),(NOW.replace(day=21),None),
    (NOW.astimezone(UTC),'preview'),
])
def test_phase_uses_beijing_trading_session(now,expected):
    assert watch_phase(now,'14:50')==expected

@pytest.fixture
def env(tmp_path):
    folder = tmp_path / 'strategies' / 'custom'
    folder.mkdir(parents=True)
    path = folder / 'demo.py'
    path.write_text('''import numpy as np
from app.backtest.matrix import make_signal_matrix
META = {'id':'demo','name':'测试策略','asset_types':['stock'],'timeframes':['1d'],'params':[],'scoring':{},'order_by':'score','descending':True,'backtest_defaults':{'max_positions':2}}
BASIC_FILTER = {'enabled':False}
EXECUTION_BACKEND = 'matrix_native'
ENTRY_SIGNALS = ['signal_triple_volume_breakout']
EXIT_SIGNALS = ['signal_close_below_ma3']
MAX_HOLD_DAYS = 8
class Native:
 def required_fields(self): return frozenset({'open','high','low','close','volume','raw_close'})
 def required_warmup_bars(self,p): return 20
 def entry_candidate_limit(self,p): return 1
 def compute_signals(self,m,p):
  entry=(m.close>=m.open*1.025).astype(np.uint8)
  return make_signal_matrix(m.shape,entry=entry,exit=(m.close<10).astype(np.uint8),score=m.volume,
    entry_signal_code=np.where(entry,0,-1).astype(np.int16),entry_signal_ids=tuple(ENTRY_SIGNALS),
    exit_signal_code=np.where(m.close<10,0,-1).astype(np.int16),exit_signal_ids=tuple(EXIT_SIGNALS),
    entry_close_floor=(m.close*.97).astype(np.float32))
MATRIX_STRATEGY=Native()
''')
    engine = StrategyEngine([folder])
    days = [DAY - timedelta(days=i) for i in range(34, -1, -1) if (DAY-timedelta(days=i)).weekday()<5]
    frame = pl.DataFrame([{'date':d,'symbol':s,'open':10.,'close':10.4,'high':10.8,'low':9.8,
        'volume':v,'raw_close':10.4,'quote_ts':int(NOW.timestamp()*1000)}
        for d in days for s,v in [('000001.SZ',100.),('600000.SH',200.)]])
    market = build_market_data_matrix(frame, field_columns={'raw_close'})
    current = frame.filter(pl.col('date')==DAY)
    quotes = SimpleNamespace(
        status=lambda:{'mode':'full_market','quote_age_ms':0,'last_fetch_ms':NOW.timestamp()*1000,
            'final_sync_done':False,'final_sync_failed':None,'symbol_count':2},
        get_enriched_today=lambda:(current, DAY), pushed=[], watchlist_notices=[],
    )
    quotes.push_alerts=lambda events:quotes.pushed.extend(events)
    quotes.notify_watchlist_updated=lambda:quotes.watchlist_notices.append(True)
    repo = SimpleNamespace(get_instruments_asset=lambda asset:pl.DataFrame({'symbol':['000001.SZ','600000.SH']}))
    screener = SimpleNamespace(repo=repo)
    service = CloseWatchService(tmp_path, engine, screener, quotes)
    service._market=lambda *args,**kwargs:market
    service.configure('demo', True, '14:50')
    return SimpleNamespace(root=tmp_path,service=service,engine=engine,quotes=quotes,
        market=market,current=current,history=frame,days=days,path=path)


def tracked(env, index=-5, symbol='600000.SH'):
    state=env.service._load()
    state.update(seeded=True,tracks=[env.service._track(symbol,env.days[index].isoformat(),symbol)])
    env.service._save(state)
    return state['tracks'][0]


def changed_market(env,value):
    close=env.market.close.copy()
    close[-1,1]=value
    raw=env.market.field('raw_close').copy()
    raw[-1,1]=value
    return replace(env.market,close=close,fields={**env.market.fields,'raw_close':raw},_valid_bars=None)


def confirmed(env,price=10.4):
    later=NOW.replace(hour=15,minute=1)
    env.quotes.status=lambda:{'mode':'full_market','quote_age_ms':0,'final_sync_done':True,'symbol_count':2}
    env.quotes.get_enriched_today=lambda:(env.current.with_columns(
        pl.lit(int(later.timestamp()*1000)).alias('quote_ts'),
        pl.when(pl.col('symbol')=='600000.SH').then(price).otherwise(pl.col('raw_close')).alias('raw_close')),DAY)
    return later


def test_preview_today_then_confirmation_and_restart_dedupe(env):
    env.service.tick(NOW)
    env.service.tick(NOW+timedelta(seconds=20))
    state=env.service.get()
    assert len(state['events'])==1 and state['events'][0]['type']=='buy_signal'
    assert state['events'][0]['phase']=='preview'
    assert '当天信号' in state['events'][0]['message']
    assert '等待下个交易日' not in state['events'][0]['message']
    assert all(t['signal_date']<str(DAY) for t in state['tracks'])
    later=confirmed(env)
    env.service.tick(later)
    assert len(env.service.get()['events'])==2
    restarted=CloseWatchService(env.root,env.engine,env.service.screener,env.quotes)
    restarted._market=env.service._market
    restarted.tick(later)
    assert len(restarted.get()['events'])==2
    assert not list(env.root.rglob('orders*'))


def test_new_confirmed_buy_starts_observation_without_assuming_a_fill(env):
    close=env.market.close.copy()
    close[:-1]=10.
    raw=env.market.field('raw_close').copy()
    raw[:-1]=10.
    market=replace(env.market,close=close,fields={**env.market.fields,'raw_close':raw},_valid_bars=None)
    env.service._market=lambda *a,**k:market
    env.service.tick(NOW)
    assert env.service.get()['tracks']==[]
    env.service.tick(confirmed(env))
    state=env.service.get()
    assert len(state['tracks'])==1
    assert state['tracks'][0]['signal_date']==str(DAY)
    assert not any('entry_date' in t or 'qty' in t for t in state['tracks'])
    assert state['watch_status'][0]['signal_age']==0
    assert abs(state['watch_status'][0]['floor_price']-10.4*.97)<1e-5
    assert not any(e['type']=='sell_signal' for e in state['events'])


def test_later_buy_signal_never_resets_original_observation_floor(env):
    track=tracked(env)
    env.service.tick(NOW)
    assert env.service.get()['tracks'][0]['signal_date']==track['signal_date']
    market=changed_market(env,9.9)
    env.service._market=lambda *a,**k:market
    env.quotes.get_enriched_today=lambda:(env.current.with_columns(
        pl.when(pl.col('symbol')=='600000.SH').then(9.9).otherwise(pl.col('raw_close')).alias('raw_close')),DAY)
    env.service.tick(NOW)
    row=env.service.get()['watch_status'][0]
    assert any('防守线' in r for r in row['reasons'])
    assert abs(row['floor_price']-10.4*.97)<1e-5
    market=replace(market,close=market.close*.5,open=market.open*.5,high=market.high*.5,low=market.low*.5,_valid_bars=None)
    env.service._market=lambda *a,**k:market
    env.service.tick(NOW)
    assert abs(env.service.get()['watch_status'][0]['floor_price']-row['floor_price'])<1e-5


def test_only_bought_signal_observations_emit_sell_and_expiry(env):
    tracked(env,index=-9)
    env.service.tick(NOW)
    sells=[e for e in env.service.get()['events'] if e['type']=='sell_signal']
    assert len(sells)==1 and sells[0]['symbol']=='600000.SH'
    assert '买点后已过8个市场交易日' in sells[0]['message']
    assert env.service.get()['watch_status'][0]['signal_age']==8


def test_sell_preview_can_clear_before_close_without_ending_observation(env):
    tracked(env)
    env.service._market=lambda *a,**k:changed_market(env,9.9)
    env.quotes.get_enriched_today=lambda:(env.current.with_columns(
        pl.when(pl.col('symbol')=='600000.SH').then(9.9).otherwise(pl.col('raw_close')).alias('raw_close')),DAY)
    env.service.tick(NOW)
    assert env.service.get()['tracks']
    env.service._market=lambda *a,**k:env.market
    env.service.tick(confirmed(env))
    state=env.service.get()
    assert state['tracks']
    assert any(e['type']=='pool_exit' and '卖出预警未获收盘确认' in e['message'] for e in state['events'])


def test_confirmed_sell_ends_observation_not_actual_position(env):
    tracked(env)
    env.service._market=lambda *a,**k:changed_market(env,9.9)
    env.service.tick(confirmed(env,9.9))
    state=env.service.get()
    assert not any(t['symbol']=='600000.SH' for t in state['tracks'])
    assert state['watch_status'][0]['ended'] is True
    assert len([e for e in state['events'] if e['type']=='sell_signal'])==1
    assert not list(env.root.rglob('orders*'))


def test_stale_top_candidate_does_not_promote_lower_rank(env):
    env.quotes.get_enriched_today=lambda:(env.current.with_columns(
        pl.when(pl.col('symbol')=='600000.SH').then(0).otherwise(pl.col('quote_ts')).alias('quote_ts')),DAY)
    env.service.tick(NOW)
    state=env.service.get()
    assert state['candidates']==[] and state['events']==[]
    assert '未确认' in state['message']


def test_missing_stale_and_unconfirmed_data_never_emit(env):
    env.quotes.status=lambda:{'mode':'full_market','quote_age_ms':999999,'final_sync_done':False}
    env.service.tick(NOW)
    assert env.service.get()['events']==[]
    env.quotes.status=lambda:{'mode':'full_market','quote_age_ms':0,'final_sync_done':False}
    env.service.tick(NOW.replace(hour=15,minute=1))
    assert env.service.get()['events']==[]
    assert '确认' in env.service.get()['message']


def test_strategy_change_and_bad_state_fail_closed(env):
    env.path.write_text(env.path.read_text()+'\n# changed\n')
    env.service.tick(NOW)
    assert env.service.get()['status']=='blocked' and not env.service.get()['events']
    env.service.path.write_text('{"version":99}')
    with pytest.raises(ValueError,match='版本'):
        env.service.get()


def test_edit_during_slow_evaluation_wins(env):
    def market(*a,**k):
        env.service.configure('demo',False,'14:50')
        return env.market
    env.service._market=market
    env.service.tick(NOW)
    assert env.service.get()['enabled'] is False
    assert not env.service.get()['events']


def test_append_checkpoint_crash_recovers_without_duplicate(env,monkeypatch):
    append=alert_store.append_many
    def crash(root,events):
        append(root,events)
        raise OSError('simulated crash after append')
    monkeypatch.setattr(alert_store,'append_many',crash)
    with pytest.raises(OSError):
        env.service.tick(NOW)
    monkeypatch.setattr(alert_store,'append_many',append)
    restarted=CloseWatchService(env.root,env.engine,env.service.screener,env.quotes)
    restarted._market=env.service._market
    restarted.tick(NOW)
    assert len(alert_store.list_recent(env.root,days=30,source='strategy'))==1
    assert len(restarted.get()['events'])==1


def test_realtime_history_reused_until_repair_generation(env,monkeypatch):
    from app.services import daily_health
    from app.strategy.engine import StrategyDataContext
    builds=[]
    def context(engine,day,ids,**kwargs):
        builds.append(day)
        return StrategyDataContext('stock','1d',day,current=kwargs['current'],history=env.history,cache_key=kwargs['cache_key'])
    screener=SimpleNamespace(repo=env.service.screener.repo,build_strategy_context=context)
    service=CloseWatchService(env.root,env.engine,screener,env.quotes)
    monkeypatch.setattr(daily_health,'revision',lambda *a:'first')
    service.tick(NOW)
    service.tick(NOW+timedelta(seconds=20))
    assert builds==[DAY]
    assert env.engine.realtime_matrix_stats('close-watch:demo')['build_count']==1
    monkeypatch.setattr(daily_health,'revision',lambda *a:'repaired')
    service.tick(NOW+timedelta(seconds=40))
    assert builds==[DAY,DAY]


def test_router_only_controls_signal_monitor_not_positions(env, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.custom.close_watch.routes import build_router
    from app.services.dingtalk_adapter import SendResult
    monkeypatch.setattr('app.services.preferences.get_dingtalk_webhook_url', lambda: 'https://oapi.dingtalk.com/robot/send?access_token=fake')
    monkeypatch.setattr('app.services.dingtalk_adapter.send_text_result', lambda *a, **kw: SendResult(True, 'accepted'))
    app=FastAPI()
    app.state.repo=SimpleNamespace(store=SimpleNamespace(data_dir=env.root))
    app.state.strategy_engine=env.engine
    app.state.monitor_engine=SimpleNamespace(rules={'auto': {'id': 'auto', 'type': 'strategy',
        'strategy_id': 'demo', 'enabled': True, 'auto_watchlist': True}})
    app.state.quote_service=env.quotes
    app.include_router(build_router())
    with TestClient(app) as client:
        root='/api/custom/close-watch'
        assert client.get(root+'/state').status_code==200
        result=client.put(root+'/config',json={'strategy_id':'demo','enabled':False,'start_time':'14:50'})
        assert result.status_code==200
        assert client.get(root+'/state').json()['watchlist']['enabled'] is False
        assert client.put(root+'/config',json={'strategy_id':'demo','enabled':True,'start_time':'09:30'}).status_code==422
        assert client.post(root+'/positions',json={}).status_code==404
        assert client.put(root+'/dingtalk', json={'enabled': 'true'}).status_code == 422
        assert client.put(root+'/dingtalk', json={'enabled': True, 'url': 'bad'}).status_code == 422
        enabled = client.put(root+'/dingtalk', json={'enabled': True})
        assert enabled.status_code == 200 and enabled.json()['dingtalk']['enabled']
        tested = client.post(root+'/dingtalk/test')
        assert tested.status_code == 200 and tested.json()['dingtalk']['records'][0]['phase'] == 'test'
        assert client.post(root+'/dingtalk/test').status_code == 409



def test_verified_close_is_kept_without_recomputing_all_night(env):
    env.service.tick(NOW)
    later = confirmed(env)
    env.service.tick(later)
    before = env.service.get()
    def unexpected(*args, **kwargs):
        raise AssertionError('already confirmed close must not recompute')
    env.service._evaluate = unexpected
    env.service.tick(later.replace(hour=22))
    assert env.service.get() == before


def test_notification_queue_recovers_local_append_checkpoint_without_duplicate(env, monkeypatch):
    from app.custom.close_watch.dingtalk import CloseWatchDingTalk
    from app.services.dingtalk_adapter import SendResult
    monkeypatch.setattr('app.services.preferences.get_dingtalk_webhook_url', lambda: 'https://oapi.dingtalk.com/robot/send?access_token=fake')
    sent = []
    channel = CloseWatchDingTalk(env.root, env.service.notification_context,
        sender=lambda *a, **kw: (sent.append(a), SendResult(True, 'accepted'))[1])
    env.service.notifications = channel
    channel.configure(True, now=NOW)
    enqueue = channel.enqueue
    monkeypatch.setattr(channel, 'enqueue', lambda *a, **kw: (_ for _ in ()).throw(OSError('checkpoint unavailable')))
    with pytest.raises(OSError):
        env.service.tick(NOW)
    first = alert_store.list_recent(env.root, source='strategy')
    assert len(first) == 1
    assert env.service._load()['outbox']
    monkeypatch.setattr(channel, 'enqueue', enqueue)
    env.service._drain(NOW)
    channel.tick(now=NOW)
    env.service._drain(NOW)
    assert len(alert_store.list_recent(env.root, source='strategy')) == 1
    assert len(sent) == 1
    assert not env.service._load()['outbox']
    assert env.service.get()['dingtalk']['records'][0]['status'] == 'sent'


def test_confirmed_tail_buy_joins_persistent_strategy_group(env, monkeypatch):
    from app.config import settings
    from app.services import watchlist
    monkeypatch.setattr(settings, 'data_dir', env.root)
    rule = {'id': 'auto', 'type': 'strategy', 'strategy_id': 'demo',
            'enabled': True, 'auto_watchlist': True, 'asset_type': 'stock'}
    env.service.monitor_rules = lambda: {'auto': rule}
    env.service.tick(confirmed(env))
    sync = getattr(env.service, 'watchlist_sync', None)
    if sync and sync._thread:
        sync._thread.join(timeout=3)
    rows = watchlist.list_symbols()
    assert [r['symbol'] for r in rows] == ['600000.SH']
    assert watchlist.list_groups()[0]['name'] == '测试策略'
    assert rows[0]['group_ids'] == [watchlist.list_groups()[0]['id']]


def _wire_tail_watchlist(env, monkeypatch, **changes):
    from app.config import settings
    monkeypatch.setattr(settings, 'data_dir', env.root)
    rule = {'id': 'auto', 'type': 'strategy', 'strategy_id': 'demo',
            'enabled': True, 'auto_watchlist': True, 'asset_type': 'stock', **changes}
    env.service.monitor_rules = lambda: {'auto': rule}
    return rule


def _wait_tail_watchlist(service):
    if service.watchlist_sync._thread:
        service.watchlist_sync._thread.join(timeout=3)
        assert not service.watchlist_sync._thread.is_alive()


@pytest.mark.parametrize('change', [{'auto_watchlist': False}, {'enabled': False},
    {'strategy_id': 'other'}, {'asset_type': 'etf'}, {'type': 'signal'}])
def test_tail_watchlist_honors_existing_matching_rule_switch(env, monkeypatch, change):
    from app.services import watchlist
    _wire_tail_watchlist(env, monkeypatch, **change)
    env.service.tick(confirmed(env))
    _wait_tail_watchlist(env.service)
    assert watchlist.list_symbols() == []
    assert not env.service.get()['watchlist']['enabled']


def test_tail_watchlist_does_not_add_preview_or_cancelled_buy(env, monkeypatch):
    from app.services import watchlist
    _wire_tail_watchlist(env, monkeypatch)
    state = env.service._load()
    state['seeded'] = True
    env.service._save(state)
    env.service.tick(NOW)
    _wait_tail_watchlist(env.service)
    assert env.service.get()['candidates'][0]['symbol'] == '600000.SH'
    assert watchlist.list_symbols() == []
    env.service._market = lambda *a: changed_market(env, 9.8)
    env.service.tick(confirmed(env, price=9.8))
    _wait_tail_watchlist(env.service)
    assert '600000.SH' not in {r['symbol'] for r in watchlist.list_symbols()}


def test_tail_watchlist_restart_recovers_active_buys_before_window_without_alert_replay(env, monkeypatch):
    from app.services import watchlist
    rule = _wire_tail_watchlist(env, monkeypatch)
    tracked(env)
    restarted = CloseWatchService(env.root, env.engine, env.service.screener, env.quotes,
                                  monitor_rules=lambda: {'auto': rule})
    restarted.tick(NOW.replace(hour=10))
    _wait_tail_watchlist(restarted)
    assert [r['symbol'] for r in watchlist.list_symbols()] == ['600000.SH']
    assert env.quotes.watchlist_notices == [True]
    assert env.quotes.pushed == [] and restarted.get()['events'] == []
    before = watchlist.list_symbols()
    restarted.tick(NOW.replace(hour=10, minute=52))
    _wait_tail_watchlist(restarted)
    assert watchlist.list_symbols() == before
    assert env.quotes.watchlist_notices == [True]


def test_tail_sell_and_paused_rule_retain_existing_favorite_and_metadata(env, monkeypatch):
    from app.services import watchlist
    rule = _wire_tail_watchlist(env, monkeypatch)
    group = watchlist.create_group('我的分组')[1]
    watchlist.add('600000.SH', '我的备注', group['id'])
    original = watchlist.list_symbols()[0]
    tracked(env)
    env.service.tick(NOW.replace(hour=10))
    _wait_tail_watchlist(env.service)
    saved = watchlist.list_symbols()[0]
    assert saved['note'] == original['note'] and saved['added_at'] == original['added_at']
    assert group['id'] in saved['group_ids'] and len(saved['group_ids']) == 2
    env.service._market = lambda *a: changed_market(env, 9.8)
    env.service.tick(confirmed(env, price=9.8))
    _wait_tail_watchlist(env.service)
    assert all(t['symbol'] != '600000.SH' for t in env.service.get()['tracks'])
    rule['enabled'] = False
    env.service.tick(NOW.replace(hour=15, minute=2))
    _wait_tail_watchlist(env.service)
    assert next(r for r in watchlist.list_symbols() if r['symbol'] == '600000.SH') == saved


def test_tail_watchlist_disk_failure_is_visible_retries_and_does_not_block_alerts(env, monkeypatch):
    from app.services import watchlist
    _wire_tail_watchlist(env, monkeypatch)
    original = watchlist.add_strategy_results
    monkeypatch.setattr(watchlist, 'add_strategy_results', lambda *a: (_ for _ in ()).throw(OSError('disk unavailable')))
    env.service.tick(confirmed(env))
    _wait_tail_watchlist(env.service)
    assert len(env.quotes.pushed) == 1
    assert env.service.get()['watchlist']['warning']
    monkeypatch.setattr(watchlist, 'add_strategy_results', original)
    env.service.tick(NOW.replace(hour=15, minute=2))
    _wait_tail_watchlist(env.service)
    assert [r['symbol'] for r in watchlist.list_symbols()] == ['600000.SH']
    assert env.service.get()['watchlist']['warning'] is None
    assert len(env.quotes.pushed) == 1


def test_tail_buy_has_canonical_name_and_signal_bound_defense(env):
    state = env.service._load()
    state.update(seeded=True, tracks=[])
    env.service._save(state)
    env.service.screener.repo.get_name_map = lambda: {'600000.SH': '示例名称'}
    env.service.tick(NOW)
    state = env.service.get()
    candidate = state['candidates'][0]
    assert candidate['name'] == '示例名称'
    assert abs(candidate['floor_price']-10.4*.97) < 1e-5
    alert = env.quotes.pushed[0]
    assert alert['name'] == '示例名称'
    assert '防守参考价10.09' in alert['message']
    assert alert['floor_signal_date'] == str(DAY)



def test_display_names_recover_old_events_without_replay_or_state_rewrite(env):
    env.service.tick(confirmed(env))
    before = env.service.path.read_text()
    sent = len(env.quotes.pushed)
    env.service.screener.repo.get_name_map = lambda: {'600000.SH': '示例名称'}
    state = env.service.get()
    for key in ['tracks', 'candidates', 'watch_status', 'events']:
        assert all(row['name'] == '示例名称' for row in state[key] if row['symbol'] == '600000.SH')
    assert '示例名称' in state['events'][0]['message']
    assert env.service.path.read_text() == before and len(env.quotes.pushed) == sent


def test_defense_is_unadjusted_for_display_and_existing_observation_keeps_its_signal_date(env):
    track = tracked(env)
    factor = 2
    market = replace(env.market, fields={**env.market.fields, 'raw_close': env.market.field('raw_close')*factor})
    env.service._market = lambda *a: market
    env.quotes.get_enriched_today = lambda: (env.current.with_columns((pl.col('raw_close')*factor).alias('raw_close')), DAY)
    env.service.tick(NOW)
    candidate = env.service.get()['candidates'][0]
    assert abs(candidate['floor_price']-10.4*.97*factor) < 1e-5
    assert candidate['floor_signal_date'] == track['signal_date']
    assert env.service.get()['tracks'][0]['signal_date'] == track['signal_date']


def test_floor_above_quote_is_explicit_in_buy_message_not_silently_capped(env, monkeypatch):
    from app.custom.close_watch.dingtalk import CloseWatchDingTalk
    native = env.engine.get('demo').matrix_strategy
    compute = native.compute_signals

    def above(market, params):
        signals = compute(market, params)
        floors = (market.close*1.2).astype('float32')
        floors.flags.writeable = False
        return replace(signals, entry_close_floor=floors)

    monkeypatch.setattr(native, 'compute_signals', above)
    # Empty observations keep this a new buy, rather than an existing-bound-line sell.
    state = env.service._load()
    state.update(seeded=True, tracks=[])
    env.service._save(state)
    env.service.tick(NOW)
    alert = env.quotes.pushed[0]
    assert alert['floor_price'] > alert['price']
    assert '参考价已低于防守线' in alert['message']
    assert '非保证成交价' in CloseWatchDingTalk._message(alert)[1]


def test_missing_names_do_not_stop_signal_calculation(env):
    def unavailable():
        raise OSError('dimension unavailable')

    env.service.screener.repo.get_name_map = unavailable
    env.service.tick(NOW)
    assert env.service.get()['status'] == 'running'
    assert env.quotes.pushed[0]['type'] == 'buy_signal'

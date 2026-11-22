from __future__ import annotations

import threading
from types import SimpleNamespace

import polars as pl
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import monitor_rules as rules_api
from app.config import settings
from app.market_time import cn_today
from app.services import watchlist
from app.services.monitor_watchlist import MonitorWatchlistSync
from app.services.quote_service import QuoteService
from app.strategy import monitor_rules
from app.strategy.engine import StrategyResult
from app.strategy.monitor import MonitorRuleEngine


@pytest.fixture(autouse=True)
def isolated_watchlist(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "data_dir", tmp_path)


def _rule(**changes):
    return monitor_rules.normalize({
        "id": "demo_rule", "name": "策略监控", "type": "strategy",
        "scope": "all", "strategy_id": "demo", "auto_watchlist": True,
        "notify_events": ["sell_signal"], **changes,
    })


def _candidate(rule, *symbols):
    return {"rule": rule, "rules_version": 1, "strategy_name": "示例策略", "symbols": frozenset(symbols)}


def _drain(sync):
    sync._thread.join(timeout=3)
    assert not sync._thread.is_alive()


def test_strategy_addition_preserves_metadata_order_and_other_groups():
    group = watchlist.create_group("手动分组")[1]
    watchlist.add("A", "保留备注", group["id"])
    watchlist.add("B", "另一条备注")
    original = watchlist.list_symbols()
    assert watchlist.add_strategy_results("demo", "示例策略", ["A", "C", "C"])
    auto_group = next(g for g in watchlist.list_groups() if g["id"] != group["id"])
    assert auto_group["name"] == "示例策略"
    rows = watchlist.list_symbols()
    assert [r["symbol"] for r in rows] == ["C", "B", "A"]
    a = next(r for r in rows if r["symbol"] == "A")
    assert a == {**original[1], "group_ids": [group["id"], auto_group["id"]]}
    rev = watchlist.revision()
    assert not watchlist.add_strategy_results("demo", "示例策略", ["A", "C"])
    assert watchlist.revision() == rev
    assert watchlist.list_symbols() == rows
    assert not watchlist.add_strategy_results("empty", "空池", [])
    assert len(watchlist.list_groups()) == 2


def test_strategy_identity_survives_restart_rename_and_same_named_strategies():
    long_name = "很长的策略名称" * 8
    watchlist.add_strategy_results("demo", long_name, ["A"])
    watchlist.add_strategy_results("other", long_name, ["A"])
    groups = watchlist.list_groups()
    assert len({g["name"] for g in groups}) == 2
    assert all(len(g["name"]) <= 24 for g in groups)
    watchlist.rename_group(groups[0]["id"], "我改的组名")
    watchlist.add_strategy_results("demo", "策略新名称", ["B"])
    assert len(watchlist.list_groups()) == 2
    assert watchlist.list_groups()[0]["name"] == "我改的组名"
    assert len(watchlist.list_symbols()[1]["group_ids"]) == 2


def test_engine_collects_initial_scored_pool_without_notifications_or_batch_loss():
    engine = MonitorRuleEngine()
    symbols = [f"stock{i}" for i in range(8)]
    strategy = SimpleNamespace(meta={"name": "示例策略"}, execution_backend="polars_expr", filter_history_fn=None)
    result = StrategyResult(
        strategy_id="demo", as_of=None, total=8,
        rows=[{"symbol": s, "score": 80 if i else 50} for i, s in enumerate(symbols)],
        scores={},
    )
    engine.set_strategy_engine(SimpleNamespace(get=lambda sid: strategy, run=lambda *a, **kw: result))
    rule = _rule(score_min=70)
    engine.set_rules([rule])
    quotes = pl.DataFrame({"symbol": symbols, "close": [10.0] * 8})
    assert engine.evaluate(quotes) == []
    candidates = engine.consume_watchlist_candidates()
    assert candidates == [_candidate(rule, *symbols[1:])]
    assert engine.consume_watchlist_candidates() == []
    engine.set_rules([{**rule, "auto_watchlist": False}])
    engine.evaluate(quotes)
    assert engine.consume_watchlist_candidates() == []
    engine.set_rules([{**rule, "asset_type": "etf"}])
    engine.evaluate(quotes, asset_type="etf")
    assert engine.consume_watchlist_candidates() == []


def test_background_sync_persists_initial_pool_and_never_removes_entries():
    rule = _rule()
    rules = {rule["id"]: rule}
    service = QuoteService()
    subscriber = service.subscribe()
    sync = MonitorWatchlistSync(lambda: rules, service.notify_watchlist_updated)
    sync.submit([_candidate(rule, "A", "B")])
    _drain(sync)
    assert subscriber.pop()["watchlist_updated"] is True
    sync.submit([_candidate(rule, "B")])  # A 移出策略池
    _drain(sync)
    rules.clear()  # 删除监控规则
    sync.submit([_candidate(rule, "C")])
    _drain(sync)
    assert {r["symbol"] for r in watchlist.list_symbols()} == {"A", "B"}
    assert subscriber.pop()["watchlist_updated"] is False
    # 新同步器模拟进程重启, 不重复分组或改写条目的时间/顺序。
    rules[rule["id"]] = rule
    rows = watchlist.list_symbols()
    restarted = MonitorWatchlistSync(lambda: rules, service.notify_watchlist_updated)
    restarted.submit([_candidate(rule, "B")])
    _drain(restarted)
    assert watchlist.list_symbols() == rows
    assert len(watchlist.list_groups()) == 1


def test_sync_failure_retries_transient_members_and_coalesces_pending_work(monkeypatch):
    rule = _rule()
    rules = {rule["id"]: rule}
    notices = []
    sync = MonitorWatchlistSync(lambda: rules, lambda: notices.append(True))
    original = watchlist.add_strategy_results
    monkeypatch.setattr(watchlist, "add_strategy_results", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    sync.submit([_candidate(rule, "A")])
    _drain(sync)
    assert "失败" in sync.warning_for(rule["id"])
    assert watchlist.list_symbols() == []
    entered, release = threading.Event(), threading.Event()

    def delayed(*args):
        entered.set()
        assert release.wait(3)
        return original(*args)

    monkeypatch.setattr(watchlist, "add_strategy_results", delayed)
    sync.submit([_candidate(rule, "B")])
    assert entered.wait(3)
    sync.submit([_candidate(rule, "C"), _candidate(rule, "D")])
    assert len(sync._pending) <= 1
    release.set()
    _drain(sync)
    assert {r["symbol"] for r in watchlist.list_symbols()} == {"A", "B", "C", "D"}
    assert sync.warning_for(rule["id"]) is None
    assert notices


def test_rule_model_roundtrip_defaults_and_nonstock_validation(tmp_path):
    legacy = _rule()
    legacy.pop("auto_watchlist")
    model = rules_api.RuleModel(**legacy)
    assert model.auto_watchlist is False
    monitor_rules.save_one(tmp_path, {**model.model_dump(), "auto_watchlist": True})
    assert monitor_rules.load_one(tmp_path, legacy["id"])["auto_watchlist"] is True
    with pytest.raises(ValueError, match="股票策略"):
        monitor_rules.validate(_rule(asset_type="etf"))


def test_quote_service_submits_pool_even_when_there_are_no_alerts(monkeypatch, tmp_path):
    rule = _rule()
    candidate = _candidate(rule, "A")
    sync = SimpleNamespace(submit=lambda batch: captured.extend(batch))
    captured = []
    engine = SimpleNamespace(
        rule_count=1, rules={rule["id"]: rule}, set_name_map=lambda names: None,
        has_rule_type=lambda kind: False, has_asset_rules=lambda asset: False,
        evaluate=lambda *args, **kw: [], consume_strategy_result_updates=lambda: False,
        consume_watchlist_candidates=lambda: [candidate],
    )
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path))
    service = QuoteService()
    service._monitor_watchlist_sync = sync
    service.set_app_state(SimpleNamespace(repo=repo, monitor_engine=engine))
    service.get_enriched_today = lambda: (pl.DataFrame({"symbol": ["A"]}), cn_today())
    monkeypatch.setattr(service, "_is_continuous_trading", lambda: True)
    service._evaluate_monitors(pl.DataFrame(), None)
    assert captured == [candidate]


def test_group_creation_then_entry_write_failure_recovers_without_duplicates(monkeypatch):
    with monkeypatch.context() as patcher:
        patcher.setattr(watchlist, "_write_entries", lambda df: (_ for _ in ()).throw(PermissionError("read only")))
        with pytest.raises(PermissionError):
            watchlist.add_strategy_results("demo", "示例策略", ["A"])
    assert watchlist.list_symbols() == []
    assert len(watchlist.list_groups()) == 1
    watchlist.add_strategy_results("demo", "示例策略", ["A"])
    assert len(watchlist.list_groups()) == 1
    assert watchlist.list_symbols()[0]["group_ids"] == [watchlist.list_groups()[0]["id"]]


def test_reenabling_rule_backfills_again_after_manual_watchlist_removal():
    rule = _rule()
    sync = MonitorWatchlistSync(lambda: {rule["id"]: rule}, lambda: None)
    sync.submit([_candidate(rule, "A")])
    _drain(sync)
    watchlist.remove("A")
    sync.submit([{**_candidate(rule, "A"), "rules_version": 3}])
    _drain(sync)
    assert [r["symbol"] for r in watchlist.list_symbols()] == ["A"]


def test_new_pool_entry_can_readd_a_manually_removed_stock():
    rule = _rule()
    sync = MonitorWatchlistSync(lambda: {rule["id"]: rule}, lambda: None)
    sync.submit([_candidate(rule, "A")])
    _drain(sync)
    watchlist.remove("A")
    sync.submit([_candidate(rule)])
    sync.submit([_candidate(rule, "A")])
    _drain(sync)
    assert [r["symbol"] for r in watchlist.list_symbols()] == ["A"]


def test_api_saves_flag_lists_it_and_leaves_watchlist_after_delete(tmp_path):
    app = FastAPI()
    app.include_router(rules_api.router)
    engine = MonitorRuleEngine()
    app.state.repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path))
    app.state.monitor_engine = engine
    app.state.strategy_engine = SimpleNamespace(get=lambda sid: object(), validate_context=lambda *args: None)
    app.state.quote_service = QuoteService()
    with TestClient(app) as client:
        assert client.post('/api/monitor-rules', json=_rule()).status_code == 200
        assert client.get('/api/monitor-rules').json()['rules'][0]['auto_watchlist'] is True
        assert engine.rules['demo_rule']['auto_watchlist'] is True
        watchlist.add_strategy_results("demo", "示例策略", ["A"])
        invalid = client.post('/api/monitor-rules', json=_rule(asset_type='etf'))
        assert invalid.status_code == 400
        assert "股票策略" in invalid.json()['detail']
        assert client.delete('/api/monitor-rules/demo_rule').status_code == 200
        assert [r['symbol'] for r in watchlist.list_symbols()] == ['A']
        assert len(watchlist.list_groups()) == 1


def test_full_monitor_to_persisted_watchlist_flow_without_alerts(monkeypatch, tmp_path):
    rule = _rule(score_min=70)
    engine = MonitorRuleEngine()
    strategy = SimpleNamespace(meta={"name": "示例策略"}, execution_backend="polars_expr", filter_history_fn=None)
    results = iter([
        StrategyResult(strategy_id="demo", as_of=cn_today(), total=2, scores={}, rows=[
            {"symbol": "A", "score": 80}, {"symbol": "B", "score": 60},
        ]),
        StrategyResult(strategy_id="demo", as_of=cn_today(), total=0, scores={}, rows=[]),
    ])
    engine.set_strategy_engine(SimpleNamespace(get=lambda sid: strategy, run=lambda *args, **kw: next(results)))
    engine.set_rules([rule])
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path), get_name_map=lambda: {})
    service = QuoteService()
    service.set_app_state(SimpleNamespace(repo=repo, monitor_engine=engine))
    service.get_enriched_today = lambda: (pl.DataFrame({"symbol": ["A", "B"]}), cn_today())
    monkeypatch.setattr(service, "_is_continuous_trading", lambda: True)
    subscriber = service.subscribe()
    service._evaluate_monitors(pl.DataFrame(), None)
    _drain(service._monitor_watchlist_sync)
    assert [r["symbol"] for r in watchlist.list_symbols()] == ["A"]
    assert watchlist.list_groups()[0]["name"] == "示例策略"
    notice = subscriber.pop()
    assert notice["watchlist_updated"] is True
    assert notice["alerts"] == []
    service._evaluate_monitors(pl.DataFrame(), None)
    engine.set_rules([])
    assert [r["symbol"] for r in watchlist.list_symbols()] == ["A"]

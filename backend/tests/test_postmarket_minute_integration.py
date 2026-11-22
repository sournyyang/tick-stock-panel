from datetime import date, datetime, timedelta
from types import SimpleNamespace

import polars as pl
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import pipeline
from app.jobs import daily_pipeline
from app.market_time import CN_TZ
from app.plugins.stocksdk import provider as sp
from app.services import postmarket_minute as pm


def test_postmarket_provider_has_bounded_bridge_and_real_amount(monkeypatch):
    calls = []

    def fake(job, timeout):
        calls.append((job, timeout))
        return {"rows": {"600000.SH": [{"timestamp": 1791874800000, "open": 10.0,
                                       "high": 10.0, "low": 10.0, "close": 10.0,
                                       "volume": 3, "amount": 3000}]}}

    monkeypatch.setattr(sp.bridge, "run_job", fake)
    df = sp.StockSDKProvider().get_minute_postmarket(["600000.SH"], date(2026, 10, 13))
    assert calls[0] == ({"op": "minute_postmarket", "symbols": ["600000.SH"], "date": "2026-10-13"}, 25)
    assert df["datetime"][0] == datetime(2026, 10, 13, 15)
    assert df.schema["datetime"] == pl.Datetime("us")
    assert df["amount"][0] == 3000
    assert df["volume"][0] == 3


def test_source_failure_does_not_fallback_to_other_provider(monkeypatch, tmp_path):
    monkeypatch.setattr(pm.preferences, "get_minute_data_provider", lambda: "custom")
    monkeypatch.setattr(pm.kline_sync, "_resolve_minute_provider", lambda name: (None, True, "unavailable"))
    monkeypatch.setattr(pm.kline_sync, "sync_minute_batch", lambda *a, **k: pytest.fail("no silent fallback"))
    with pytest.raises(RuntimeError, match="unavailable"):
        pm._fetch(["600000.SH"], SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path)), None, date(2026, 10, 13))


def test_closed_gate_does_not_resolve_universe(monkeypatch):
    monkeypatch.setattr(pm, "gate_reason", lambda now=None: "non_trading_day")
    monkeypatch.setattr(daily_pipeline, "_resolve_minute_symbols", lambda *a: pytest.fail("no holiday network"))
    assert daily_pipeline.run_postmarket_minute(None, None) == {"status": "non_trading_day"}


def test_ready_minute_work_continues_without_waiting_for_scheduler(tmp_path, monkeypatch):
    now = datetime(2026, 10, 13, 16, tzinfo=CN_TZ)
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path))
    symbols = ["600000.SH", "000001.SZ"]
    report = {"date": str(now.date()), "symbols": {
        symbol: {"status": "pending", "attempts": 0} for symbol in symbols
    }}
    calls = []
    progress = object()
    monkeypatch.setattr(pm, "cn_now", lambda: now)
    monkeypatch.setattr(pm, "gate_reason", lambda now=None: None)
    monkeypatch.setattr(daily_pipeline, "_resolve_minute_symbols", lambda *a: symbols)

    def round_run(repo, capset, universe, *, on_progress):
        assert universe == symbols
        assert on_progress is progress
        symbol = symbols[len(calls)]
        calls.append(symbol)
        report["symbols"][symbol].update(status="complete", attempts=1)
        return pm._save(tmp_path, report, now)

    monkeypatch.setattr(pm, "run_session", round_run)
    result = daily_pipeline.run_postmarket_minute(repo, None, on_progress=progress)
    assert calls == symbols
    assert result["status"] == "complete"
    assert result["complete_symbols"] == 2


@pytest.mark.parametrize("waiting", ["retry", "cooldown", "midnight", "disabled"])
def test_continuation_yields_when_work_must_wait(tmp_path, monkeypatch, waiting):
    now = datetime(2026, 10, 13, 23, 59, tzinfo=CN_TZ)
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path))
    symbol = "600000.SH"
    record = {"status": "failed", "attempts": 1}
    report = {"date": str(now.date()), "symbols": {symbol: record}}
    if waiting == "retry":
        record["next_retry_at"] = (now + timedelta(minutes=15)).isoformat()
    if waiting == "cooldown":
        report["cooldown_until"] = (now + timedelta(minutes=15)).isoformat()
    calls = []
    monkeypatch.setattr(pm, "cn_now", lambda: now)
    monkeypatch.setattr(pm, "gate_reason", lambda now=None: None)
    monkeypatch.setattr(daily_pipeline, "_resolve_minute_symbols", lambda *a: [symbol])

    def round_run(*args, **kwargs):
        calls.append(1)
        assert len(calls) == 1, "must release job instead of spinning or bypassing delay"
        result = pm._save(tmp_path, report, now)
        if waiting == "midnight":
            monkeypatch.setattr(pm, "cn_now", lambda: now + timedelta(minutes=2))
        if waiting == "disabled":
            monkeypatch.setattr(pm, "gate_reason", lambda now=None: "disabled")
        return result

    monkeypatch.setattr(pm, "run_session", round_run)
    assert daily_pipeline.run_postmarket_minute(repo, None)["status"] == "pending"
    assert calls == [1]


def test_status_api_and_expanded_failures(tmp_path, monkeypatch):
    app = FastAPI()
    app.state.repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path))
    app.include_router(pipeline.router)
    seen = []

    def status(data_dir, day, *, include_symbols):
        seen.append((data_dir, day, include_symbols))
        return {"coverage_pct": 80.0, "failed_symbols": 1,
                **({"symbols": {"920001.BJ": {"reason": "empty"}}} if include_symbols else {})}

    monkeypatch.setattr(pm, "session_status", status)
    with TestClient(app) as client:
        response = client.get("/api/pipeline/minute/status?trade_date=2026-10-13&include_symbols=true")
        assert response.status_code == 200
        assert response.json()["symbols"]["920001.BJ"]["reason"] == "empty"
        assert client.get("/api/pipeline/minute/status?trade_date=invalid").status_code == 422
    assert seen == [(tmp_path, date(2026, 10, 13), True)]


def test_run_api_rejects_pre_close_without_creating_job(monkeypatch):
    app = FastAPI()
    app.include_router(pipeline.router)
    monkeypatch.setattr(pm, "gate_reason", lambda: "before_close")
    monkeypatch.setattr(pipeline, "_submit_job", lambda *a: pytest.fail("must not submit"))
    with TestClient(app) as client:
        response = client.post("/api/pipeline/minute/run")
        assert response.status_code == 400
        assert "15:45" in response.json()["detail"]


def test_scheduler_registers_independent_coalesced_minute_job(tmp_path, monkeypatch):
    jobs = {}

    class Scheduler:
        def __init__(self, **kwargs):
            assert kwargs["timezone"] == "Asia/Shanghai"

        def add_job(self, fn, **kwargs):
            jobs[kwargs["id"]] = (fn, kwargs)

        def start(self):
            pass

    monkeypatch.setattr(daily_pipeline, "AsyncIOScheduler", Scheduler)
    monkeypatch.setattr(daily_pipeline._prefs, "get_review_schedule", lambda: {"enabled": False})
    daily_pipeline.start_scheduler(SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path)), None)
    fn, kwargs = jobs["postmarket_minute"]
    assert kwargs["max_instances"] == 1
    assert kwargs["coalesce"] is True
    fire = kwargs["trigger"].get_next_fire_time(None, datetime(2026, 10, 13, 15, 44, tzinfo=CN_TZ))
    assert fire == datetime(2026, 10, 13, 15, 45, tzinfo=CN_TZ)
    monkeypatch.setattr(pm, "gate_reason", lambda now=None: "before_close")
    monkeypatch.setattr(daily_pipeline, "_run_tracked", lambda *a: pytest.fail("closed gate"))
    fn()

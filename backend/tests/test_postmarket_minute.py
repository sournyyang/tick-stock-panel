from datetime import datetime, timedelta
from threading import RLock
from types import SimpleNamespace

import polars as pl
import pytest

from app.market_time import CN_TZ
from app.services import postmarket_minute as pm

DAY = datetime(2026, 10, 13, 16, tzinfo=CN_TZ)
SYMBOLS = ["600000.SH", "000001.SZ", "920001.BJ"]


def bars(symbol, *, amount=2000.0, volume=2.0):
    return pl.DataFrame({
        "symbol": [symbol, symbol],
        "datetime": [DAY.replace(hour=9, minute=31, tzinfo=None),
                     DAY.replace(hour=15, minute=0, tzinfo=None)],
        "open": [10.0, 10.0], "high": [10.0, 10.0],
        "low": [10.0, 10.0], "close": [10.0, 10.0],
        "volume": [volume / 2, volume / 2], "amount": [amount / 2, amount / 2],
    })


@pytest.fixture
def repo(tmp_path, monkeypatch):
    path = tmp_path / "kline_daily" / "date=2026-10-13" / "part.parquet"
    path.parent.mkdir(parents=True)
    pl.DataFrame({"symbol": SYMBOLS, "volume": [2.0, 2.0, 0.0],
                  "amount": [2000.0, 2000.0, 0.0], "close": [10.0] * 3}).write_parquet(path)
    monkeypatch.setattr(pm.preferences, "get_minute_sync_enabled", lambda: True)
    monkeypatch.setattr(pm.preferences, "get_minute_data_provider", lambda: "test")
    monkeypatch.setattr(pm.trading_day, "is_trading_day", lambda now=None: True)
    monkeypatch.setattr(pm, "cn_now", lambda: DAY)
    monkeypatch.setattr(pm, "MIN_UNIVERSE_SYMBOLS", 1)
    monkeypatch.setattr(pm, "_publish", lambda repo: None)
    return SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path), _write_lock=RLock())


def test_collect_only_today_and_retry_only_failed_after_delay(repo, monkeypatch):
    calls = []

    def fetch(symbols, repo, capset, day):
        calls.append((symbols, day))
        # One symbol fails the initial scan; delayed retry succeeds.
        if len(calls) == 1:
            stale = bars(SYMBOLS[0]).with_columns(pl.col("datetime") - pl.duration(days=1))
            return pl.concat([bars(SYMBOLS[0]), stale])
        return bars(SYMBOLS[1])

    monkeypatch.setattr(pm, "_fetch", fetch)
    result = pm.run_session(repo, None, SYMBOLS, now=DAY)
    assert result["complete_symbols"] == 1
    assert result["failed_symbols"] == 1
    assert result["suspended_symbols"] == 1
    assert result["coverage_pct"] == 50.0
    assert calls == [(sorted(SYMBOLS[:2]), DAY.date())]
    out = pl.read_parquet(repo.store.data_dir / "kline_minute/date=2026-10-13/part.parquet")
    assert out["datetime"].dt.date().unique().to_list() == [DAY.date()]
    assert not (repo.store.data_dir / "kline_minute/date=2026-10-12").exists()
    pm.run_session(repo, None, SYMBOLS, now=DAY + timedelta(minutes=1))
    assert len(calls) == 1
    result = pm.run_session(repo, None, SYMBOLS, now=DAY + timedelta(minutes=16))
    assert calls[-1][0] == [SYMBOLS[1]]
    assert result["status"] == "complete"
    assert result["coverage_pct"] == 100.0
    pm.run_session(repo, None, SYMBOLS, now=DAY + timedelta(minutes=40))
    assert len(calls) == 2


@pytest.mark.parametrize("frame,reason", [
    (bars(SYMBOLS[0]).drop("amount"), "missing_columns"),
    (bars(SYMBOLS[0], amount=1000), "amount_mismatch"),
    (bars(SYMBOLS[0], volume=1), "volume_mismatch"),
    (pl.concat([bars(SYMBOLS[0]), bars(SYMBOLS[0])]), "duplicate_minutes"),
    (bars(SYMBOLS[0]).with_columns(pl.lit(float("nan")).alias("close")), "invalid_values"),
])
def test_incomplete_data_is_not_success(frame, reason):
    assert pm.validate_symbol(frame, {"volume": 2.0, "amount": 2000.0, "close": 10.0}) == reason


def test_failure_checkpoint_survives_restart_and_attempt_limit(repo, monkeypatch):
    monkeypatch.setattr(pm, "_fetch", lambda *a: pl.DataFrame())
    for n in range(3):
        result = pm.run_session(repo, None, SYMBOLS, now=DAY + timedelta(minutes=n * 20))
    assert result["status"] == "incomplete"
    assert result["failed_symbols"] == 2
    saved = pm.session_status(repo.store.data_dir, DAY.date(), include_symbols=True)
    assert all(saved["symbols"][s]["attempts"] == 3 for s in SYMBOLS[:2])
    monkeypatch.setattr(pm, "_fetch", lambda *a: pytest.fail("exhausted symbols must not retry"))
    pm.run_session(repo, None, SYMBOLS, now=DAY + timedelta(hours=2))


def test_holiday_before_close_and_unknown_calendar_do_not_fetch(repo, monkeypatch):
    monkeypatch.setattr(pm, "_fetch", lambda *a: pytest.fail("must not fetch"))
    assert pm.run_session(repo, None, SYMBOLS, now=DAY.replace(hour=14))["status"] == "before_close"
    for verdict, reason in [(False, "non_trading_day"), (None, "calendar_unknown")]:
        monkeypatch.setattr(pm.trading_day, "is_trading_day", lambda now=None, verdict=verdict: verdict)
        assert pm.run_session(repo, None, SYMBOLS, now=DAY)["status"] == reason


def test_no_daily_reference_does_not_claim_coverage(repo, monkeypatch):
    other = DAY + timedelta(days=1)
    monkeypatch.setattr(pm, "_fetch", lambda *a: pytest.fail("must await daily reference"))
    assert pm.run_session(repo, None, SYMBOLS, now=other)["status"] == "awaiting_daily"


def test_missing_daily_symbols_remain_in_full_market_denominator(repo, monkeypatch):
    extra = "300001.SZ"
    monkeypatch.setattr(pm, "_fetch", lambda symbols, *a: pl.concat([bars(s) for s in symbols]))
    result = pm.run_session(repo, None, [*SYMBOLS, extra], now=DAY)
    assert result["expected_symbols"] == 3
    assert result["unverified_symbols"] == 1
    assert result["coverage_pct"] == pytest.approx(200 / 3, abs=0.01)
    assert result["status"] != "complete"
    path = repo.store.data_dir / "kline_minute/date=2026-10-13/part.parquet"
    assert extra in pl.read_parquet(path)["symbol"].to_list()
    assert result["data_symbols"] == 3


def test_repeated_runs_do_not_overwrite_complete_symbols(repo, monkeypatch):
    monkeypatch.setattr(pm, "_fetch", lambda symbols, *a: pl.concat([bars(s) for s in symbols]))
    pm.run_session(repo, None, SYMBOLS, now=DAY)
    monkeypatch.setattr(pm, "_fetch", lambda *a: pytest.fail("complete checkpoint must skip"))
    assert pm.run_session(repo, None, SYMBOLS, now=DAY + timedelta(minutes=20))["status"] == "complete"


def test_narrow_universe_is_not_reported_as_full_market(repo, monkeypatch):
    monkeypatch.setattr(pm, "MIN_UNIVERSE_SYMBOLS", 4000)
    monkeypatch.setattr(pm, "_fetch", lambda *a: pytest.fail("narrow pool must not collect"))
    result = pm.run_session(repo, None, SYMBOLS, now=DAY)
    assert result["status"] == "universe_incomplete"
    assert "coverage_pct" not in result


def test_expired_retry_stops_at_beijing_midnight(repo, monkeypatch):
    monkeypatch.setattr(pm, "cn_now", lambda: DAY + timedelta(days=1))
    monkeypatch.setattr(pm, "_fetch", lambda *a: pytest.fail("no prior-day backfill"))
    assert pm.run_session(repo, None, SYMBOLS, now=DAY)["pending_symbols"] == 2


def test_partial_rows_never_replace_existing_valid_rows(repo, monkeypatch):
    part = repo.store.data_dir / "kline_minute/date=2026-10-13/part.parquet"
    part.parent.mkdir(parents=True)
    old = bars(SYMBOLS[0], amount=500)
    old.write_parquet(part)
    monkeypatch.setattr(pm, "_fetch", lambda *a: bars(SYMBOLS[0], amount=1000))
    result = pm.run_session(repo, None, SYMBOLS, now=DAY)
    assert result["complete_symbols"] == 0
    assert pl.read_parquet(part).equals(old)


def test_cancel_preserves_durable_batch_and_releases_session_lock(repo, monkeypatch):
    from app.services.pipeline_jobs import JobCancelledError

    symbols = [f"6000{i:02d}.SH" for i in range(10)]
    daily = repo.store.data_dir / "kline_daily/date=2026-10-13/part.parquet"
    pl.DataFrame({"symbol": symbols, "volume": [2.0] * 10,
                  "amount": [2000.0] * 10, "close": [10.0] * 10}).write_parquet(daily)
    monkeypatch.setattr(pm, "_fetch", lambda group, *a: pl.concat([bars(s) for s in group]))
    monkeypatch.setattr(pm.time, "sleep", lambda seconds: None)
    seen = []

    def progress(*args):
        seen.append(args)
        if len(seen) == 2:
            raise JobCancelledError("test")

    with pytest.raises(JobCancelledError):
        pm.run_session(repo, None, symbols, now=DAY, on_progress=progress)
    assert pm.session_status(repo.store.data_dir, DAY.date())["complete_symbols"] == 8
    assert pm._RUN_LOCK.acquire(blocking=False)
    pm._RUN_LOCK.release()


def test_old_unverified_checkpoint_without_rows_does_not_claim_data(repo, monkeypatch):
    report = {"date": str(DAY.date()), "symbols": {
        SYMBOLS[0]: {"status": "unverified", "attempts": 1,
                     "next_retry_at": (DAY + timedelta(minutes=15)).isoformat()},
    }}
    pm._save(repo.store.data_dir, report, DAY)
    monkeypatch.setattr(pm, "_fetch", lambda *a: pl.DataFrame())
    result = pm.run_session(repo, None, SYMBOLS, now=DAY)
    assert result["unverified_symbols"] == 0
    assert result["data_symbols"] == 0

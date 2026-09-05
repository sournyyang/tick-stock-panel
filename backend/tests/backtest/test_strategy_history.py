from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.api.backtest import strategy_history, strategy_history_result
from app.backtest.history import (
    get_strategy_backtest_history,
    list_strategy_backtest_history,
    persist_strategy_backtest_result,
)


def _result(run_id: str, *, total_return: float = 0.12) -> dict:
    return {
        "run_id": run_id,
        "config": {
            "strategy_id": "trial_line_breakout",
            "asset_type": "stock",
            "mode": "position",
            "start": "2025-09-04",
            "end": "2026-09-04",
            "params": {"max_trial_volume_multiple": 4.5},
        },
        "stats": {
            "total_return": total_return,
            "max_drawdown": -0.03,
            "n_trades": 88,
        },
        "strategy_info": {
            "id": "trial_line_breakout",
            "name": "试盘线战法",
            "version": "1.1.0",
            "code_hash": "abc123def456",
        },
        "trades": [{"symbol": "000001.SZ"}],
        "error": None,
    }


def _request(data_dir):
    store = SimpleNamespace(data_dir=data_dir)
    repo = SimpleNamespace(store=store)
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(repo=repo)))


def test_strategy_history_round_trip_and_api_listing(tmp_path):
    first = _result("abc123def0", total_return=0.12)
    second = _result("abc123def1", total_return=0.22)

    meta_first = persist_strategy_backtest_result(tmp_path, first)
    meta_second = persist_strategy_backtest_result(tmp_path, second)

    assert meta_first is not None
    assert meta_second is not None
    assert meta_second["strategy_version"] == "1.1.0"
    assert meta_second["code_hash"] == "abc123def456"
    assert meta_second["stats"]["total_return"] == 0.22
    assert meta_second["trade_count"] == 1

    items = list_strategy_backtest_history(
        tmp_path,
        strategy_id="trial_line_breakout",
    )
    assert {item["run_id"] for item in items} == {"abc123def0", "abc123def1"}

    wrapper = get_strategy_backtest_history(tmp_path, "abc123def1")
    assert wrapper is not None
    assert wrapper["result"] == second

    api_items = strategy_history(_request(tmp_path), "trial_line_breakout", 50)
    assert len(api_items["items"]) == 2
    api_result = strategy_history_result("abc123def1", _request(tmp_path))
    assert api_result["result"]["stats"]["total_return"] == 0.22


def test_strategy_history_ignores_failed_runs_and_rejects_bad_ids(tmp_path):
    failed = _result("abc123def0")
    failed["error"] = "cancelled"

    assert persist_strategy_backtest_result(tmp_path, failed) is None
    assert list_strategy_backtest_history(tmp_path) == []

    with pytest.raises(ValueError, match="invalid backtest run id"):
        get_strategy_backtest_history(tmp_path, "../../secret")
    with pytest.raises(HTTPException) as exc_info:
        strategy_history_result("../../secret", _request(tmp_path))
    assert exc_info.value.status_code == 400


def test_strategy_history_skips_corrupt_metadata_and_payload(tmp_path):
    root = tmp_path / "backtest_results" / "strategy"
    root.mkdir(parents=True)
    (root / "broken.meta.json").write_text("not-json", encoding="utf-8")
    (root / "broken.json.gz").write_bytes(b"not-gzip")

    assert list_strategy_backtest_history(tmp_path) == []
    assert get_strategy_backtest_history(tmp_path, "broken") is None

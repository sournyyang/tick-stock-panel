from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from app.backtest.engine import BacktestEngine, MatcherConfig, SimulationOptions
from app.backtest.matrix import build_market_data_matrix, build_market_matrix_from_signals
from app.backtest.strategy import StrategyDependencyResolver
from app.strategy.engine import StrategyDataContext, StrategyEngine

BUILTIN = Path(__file__).resolve().parents[2] / "app/strategy/seeds"
STRATEGY_ID = "limit_up_pullback"


def _strategy():
    return StrategyEngine._load_file(BUILTIN / f"{STRATEGY_ID}.py")


def _rows(stage="涨停当日", *, factor=0.5):
    rows = []
    day = date(2025, 1, 2)
    for i in range(73):
        while day.weekday() >= 5:
            day += timedelta(days=1)
        raw_open, raw_high, raw_low, raw_close, volume = 10.0, 10.2, 9.9, 10.0, 100.0
        offset = int(stage == "涨停次日")
        if i == 60:
            raw_open, raw_high, raw_low, raw_close, volume = 10.1, 11.0, 10.0, 11.0, 300.0
        elif offset and i == 61:
            raw_open, raw_high, raw_low, raw_close, volume = 11.1, 11.6, 11.0, 11.5, 900.0
        elif i == 61 + offset:
            raw_open, raw_high, raw_low, raw_close, volume = 10.9, 11.0, 10.4, 10.6, 200.0
        elif i == 62 + offset:
            raw_open, raw_high, raw_low, raw_close, volume = 10.6, 10.8, 10.2, 10.3, 100.0
        elif i == 63 + offset:
            raw_open, raw_high, raw_low, raw_close, volume = 10.3, 10.8, 10.25, 10.7, 120.0
        elif i > 63 + offset:
            raw_open, raw_high, raw_low, raw_close, volume = 10.8, 11.0, 10.6, 10.8, 100.0
        rows.append({
            "symbol": "000001.SZ", "name": "测试股票", "date": day,
            "open": raw_open * factor, "high": raw_high * factor,
            "low": raw_low * factor, "close": raw_close * factor,
            "raw_high": raw_high, "raw_low": raw_low, "raw_close": raw_close,
            "volume": volume, "signal_limit_up": False, "signal_limit_down": False,
        })
        day += timedelta(days=1)
    return rows, 63 + int(stage == "涨停次日")


def _market(rows):
    strategy = _strategy()
    return build_market_data_matrix(pl.DataFrame(rows), field_columns=strategy.matrix_strategy.required_fields())


def _signals(rows, params=None):
    return _strategy().matrix_strategy.compute_signals(_market(rows), params or {})


@pytest.mark.parametrize("stage", ["涨停当日", "涨停次日"])
def test_both_patterns_use_raw_limit_price_and_only_emit_on_confirmation(stage):
    rows, signal = _rows(stage)
    output = _signals(rows, {"volume_stage": stage})
    assert np.flatnonzero(output.entry[:, 0]).tolist() == [signal]
    assert not output.exit.any()
    assert output.entry_signal_ids == ("signal_limit_up_pullback",)


@pytest.mark.parametrize("stage", ["涨停当日", "涨停次日"])
def test_confirmation_variants_distinguish_intraday_new_low_volume_and_breakout(stage):
    rows, signal = _rows(stage)
    rows[signal]["low"] = 5.0
    rows[signal]["raw_low"] = 10.0
    assert not _signals(rows, {"volume_stage": stage}).entry[signal, 0]
    for mode in ("收复昨收", "放量收复昨收"):
        assert _signals(rows, {"volume_stage": stage, "confirmation": mode}).entry[signal, 0]
    assert not _signals(rows, {"volume_stage": stage, "confirmation": "收盘突破昨高"}).entry[signal, 0]
    rows[signal]["volume"] = rows[signal-1]["volume"]
    assert not _signals(rows, {"volume_stage": stage, "confirmation": "放量收复昨收"}).entry[signal, 0]
    rows[signal]["close"], rows[signal]["raw_close"] = 5.425, 10.85
    rows[signal]["high"], rows[signal]["raw_high"] = 5.45, 10.9
    assert _signals(rows, {"volume_stage": stage, "confirmation": "收盘突破昨高"}).entry[signal, 0]


def test_invalid_confirmation_mode_is_rejected():
    rows, _ = _rows()
    with pytest.raises(ValueError, match="阳线确认"):
        _signals(rows, {"confirmation": "unknown"})


@pytest.mark.parametrize("stage", ["涨停当日", "涨停次日"])
def test_five_day_volume_reference_excludes_current_observation_and_preserves_default(stage):
    rows, signal = _rows(stage)
    if stage == "涨停当日":
        rows[59]["volume"] = 200.0
        rows[60]["volume"] = 400.0
    else:
        rows[61]["volume"] = 450.0
    assert not _signals(rows, {"volume_stage": stage}).entry[signal, 0]
    assert _signals(rows, {"volume_stage": stage, "volume_reference": "前5日均量"}).entry[signal, 0]
    future = [dict(r) for r in rows]
    for row in future[signal+1:]:
        row["volume"] *= 10000
    a = _signals(rows, {"volume_stage": stage, "volume_reference": "前5日均量"})
    b = _signals(future, {"volume_stage": stage, "volume_reference": "前5日均量"})
    np.testing.assert_equal(a.entry[:signal+1], b.entry[:signal+1])


def test_unknown_volume_reference_is_rejected():
    rows, _ = _rows()
    with pytest.raises(ValueError, match="倍量比较"):
        _signals(rows, {"volume_reference": "unknown"})


@pytest.mark.parametrize("stage", ["涨停当日", "涨停次日"])
def test_three_times_threshold_and_control_switch(stage):
    rows, signal = _rows(stage)
    rows[60 if stage == "涨停当日" else 61]["volume"] -= 0.1
    assert not _signals(rows, {"volume_stage": stage}).entry.any()
    assert _signals(rows, {"volume_stage": stage, "use_volume_filter": False}).entry[signal, 0]
    assert _signals(rows, {"volume_stage": stage, "volume_multiple": 2.9}).entry[signal, 0]


@pytest.mark.parametrize("kind", ["volume", "bearish", "declining_close", "bullish", "low", "missing", "one_price", "fake_limit"])
def test_rejects_invalid_pattern(kind):
    rows, _ = _rows()
    if kind == "volume":
        rows[62]["volume"] = 201.0
    elif kind == "bearish":
        rows[62]["open"] = 5.0
    elif kind == "declining_close":
        rows[61]["close"] = 5.55
        rows[61]["open"] = 5.6
    elif kind == "bullish":
        rows[63]["open"] = 5.4
    elif kind == "low":
        rows[63]["low"] = 5.09
    elif kind == "missing":
        rows[62]["close"] = None
        rows[62]["volume"] = 0.0
    elif kind == "one_price":
        rows[60]["raw_low"] = rows[60]["raw_high"]
    elif kind == "fake_limit":
        rows[60]["signal_limit_up"] = True
        rows[60]["raw_close"] = 10.2
        rows[60]["close"] = 5.1
    assert not _signals(rows).entry.any()


def test_ex_right_anchor_is_excluded_even_if_raw_price_looks_locked():
    rows, _ = _rows()
    rows[60]["close"] = 5.6
    assert not _signals(rows).entry.any()


def test_missing_previous_volume_is_not_three_times_zero():
    rows, _ = _rows()
    rows[59]["volume"] = 0.0
    assert not _signals(rows).entry.any()


def test_pattern_does_not_jump_over_a_missing_market_session():
    rows, _ = _rows()
    missing_day = rows[62]["date"]
    rows = [row for row in rows if row["date"] != missing_day]
    rows.append({**rows[0], "symbol": "600000.SH", "date": missing_day})
    assert not _signals(rows).entry.any()


def test_insufficient_history_and_missing_raw_fields_fail_closed():
    rows, _ = _rows()
    assert not _signals(rows[10:]).entry.any()
    m = _market(rows)
    with pytest.raises(ValueError, match="raw_high"):
        _strategy().matrix_strategy.compute_signals(replace(m, fields={}), {})


def test_future_candles_cannot_change_past_signal():
    rows, signal = _rows()
    prefix = _signals(rows[:signal+1])
    full = _signals(rows)
    np.testing.assert_array_equal(prefix.entry, full.entry[:signal+1])


@pytest.mark.parametrize("params", [
    {"volume_stage": "unknown"}, {"volume_multiple": 0},
    {"volume_multiple": float("nan")}, {"volume_multiple": float("inf")},
])
def test_invalid_params_raise_clear_errors(params):
    rows, _ = _rows()
    with pytest.raises(ValueError):
        _signals(rows, params)


def test_registry_defaults_and_dependency_plan():
    engine = StrategyEngine(strategy_dirs=[BUILTIN])
    assert engine.load_errors() == []
    strategy = engine.get(STRATEGY_ID)
    assert strategy.execution_backend == "matrix_native"
    assert strategy.max_hold_days == 5
    assert strategy.stop_loss is None and strategy.take_profit is None
    assert strategy.meta["default_entry_fill"] == "open_t+1"
    assert strategy.meta["default_exit_fill"] == "close_t"
    assert strategy.basic_filter["boards"] == ["沪主板", "深主板"]
    assert strategy.basic_filter["amount_min"] is None
    assert strategy.basic_filter["market_cap_min"] is None
    params = engine.resolve_params(strategy)
    assert params == {"volume_stage": "涨停当日", "confirmation": "低点不下移",
                      "volume_reference": "前一日", "use_volume_filter": True, "volume_multiple": 3.0}
    plan = StrategyDependencyResolver().resolve(strategy, params=params,
        basic_filter=strategy.basic_filter, entry_signals=strategy.entry_signals, exit_signals=[])
    assert {"raw_close", "raw_high", "raw_low", "price_limit_pct"} <= plan.matrix_columns
    assert not plan.full_feature_fallback
    assert engine.required_history_bars([STRATEGY_ID]) >= 65


def test_saved_params_switch_screening_pattern_without_stale_instance(tmp_path):
    from app.strategy.config import load_override, save_override

    rows, signal = _rows("涨停次日")
    history = pl.DataFrame(rows)
    target = rows[signal]["date"]
    context = StrategyDataContext(asset_type="stock", timeframe="1d", as_of=target,
        current=history.filter(pl.col("date") == target), history=history)
    engine = StrategyEngine(strategy_dirs=[BUILTIN], override_loader=lambda sid: load_override(tmp_path, sid))
    save_override(tmp_path, STRATEGY_ID, {"params": {"volume_stage": "涨停次日"}})
    assert engine.run(STRATEGY_ID, context, overrides=load_override(tmp_path, STRATEGY_ID)).total == 1
    save_override(tmp_path, STRATEGY_ID, {"params": {"volume_stage": "涨停当日"}})
    assert engine.run(STRATEGY_ID, context, overrides=load_override(tmp_path, STRATEGY_ID)).total == 0


@pytest.mark.parametrize("blocked_entry", [False, True])
def test_production_matcher_uses_next_open_costs_and_five_day_exit(blocked_entry):
    rows, signal = _rows()
    if blocked_entry:
        for column in ("open", "high", "low", "close"):
            rows[signal+1][column] = 11.77 * 0.5
        rows[signal+1]["signal_limit_up"] = True
    market = _market(rows)
    strategy = _strategy()
    signals = strategy.matrix_strategy.compute_signals(market, {})
    matrix = build_market_matrix_from_signals(market, signals, entry_delay_bars=1)
    config = MatcherConfig(entry_fill=strategy.meta["default_entry_fill"],
        exit_fill=strategy.meta["default_exit_fill"], max_hold_days=strategy.max_hold_days,
        commission_pct=0.0002, stamp_tax_pct=0.001, slippage_bps=5)
    result = BacktestEngine(repo=None).simulate_independent_market_matrix(
        matrix, int(signals.entry.sum()), config,
        options=SimulationOptions(include_monte_carlo=False, include_curves=False))
    if blocked_entry:
        assert result.trades == []
        assert result.stats["execution"]["buy_limit_up"] == 1
    else:
        trade, = result.trades
        assert str(trade.entry_signal_date) == str(rows[signal]["date"])
        assert str(trade.entry_date) == str(rows[signal+1]["date"])
        assert str(trade.exit_date) == str(rows[signal+6]["date"])
        assert trade.duration == 5 and trade.exit_reason == "max_hold"
        assert trade.pnl_pct == pytest.approx((1-0.0017)/(1+0.0007)-1, abs=1e-6)

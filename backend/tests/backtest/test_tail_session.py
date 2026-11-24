from dataclasses import replace
from datetime import date, datetime, time, timedelta

import numpy as np
import polars as pl
import pytest

from app.backtest.engine import BacktestEngine, MatcherConfig, SameSessionFill
from app.backtest.matrix import (
    build_market_data_matrix,
    build_market_matrix_from_signals,
    make_signal_matrix,
)
from app.backtest.tail_session import _snapshot_row, _with_partial_bar

START = date(2026, 11, 5)


def minute_frame(day, *, future_price=11.0, fill_price=10.2):
    timestamps = [
        datetime.combine(day, time(9, 31)) + timedelta(minutes=i) for i in range(120)
    ] + [
        datetime.combine(day, time(13, 1)) + timedelta(minutes=i) for i in range(120)
    ]
    prices = [
        10.0 if t.time() <= time(14, 39) else fill_price if t.time() == time(14, 42) else future_price
        for t in timestamps
    ]
    return pl.DataFrame({
        "symbol": ["600000.SH"] * len(prices), "datetime": timestamps,
        "open": prices, "high": prices, "low": prices, "close": prices,
        "volume": [10.0] * len(prices),
    })


def market_for_minutes(final=11.0):
    return build_market_data_matrix(pl.DataFrame({
        "symbol": ["600000.SH"], "date": [START], "open": [10.0],
        "high": [max(final, 10.2)], "low": [min(final, 10.0)],
        "close": [final], "raw_close": [final], "volume": [2400.0],
    }), field_columns={"raw_close"})


def test_after_signal_prices_do_not_leak_into_daily_snapshot(tmp_path):
    witnessed = []
    for close in (11.0, 9.6):
        market = market_for_minutes(close)
        partial, raw, prices, _, count = _snapshot_row(
            market, 0, minute_frame(START, future_price=close), START, tmp_path,
        )
        reconstructed = _with_partial_bar(market, 0, partial, raw)
        witnessed.append(tuple(getattr(reconstructed, k).item() for k in ("open", "high", "low", "close", "volume")))
        assert prices.tolist() == [10.2]
        assert count == 1
        assert market.close.item() == pytest.approx(close)
    assert witnessed[0] == witnessed[1] == (10.0, 10.0, 10.0, 10.0, 2190.0)


def test_missing_presignal_minute_rejects_even_when_total_volume_matches(tmp_path):
    frame = minute_frame(START)
    frame = frame.filter(pl.col("datetime").dt.time() != time(14, 39)).with_columns(
        pl.when(pl.col("datetime").dt.time() == time(14, 38)).then(20.0).otherwise(pl.col("volume")).alias("volume")
    )
    with pytest.raises(ValueError, match="分钟K覆盖不足"):
        _snapshot_row(market_for_minutes(), 0, frame, START, tmp_path)


def test_missing_execution_minute_keeps_signal_but_has_no_fill_price(tmp_path):
    frame = minute_frame(START)
    frame = frame.filter(pl.col("datetime").dt.time() != time(14, 42)).with_columns(
        pl.when(pl.col("datetime").dt.time() == time(14, 43)).then(20.0).otherwise(pl.col("volume")).alias("volume")
    )
    partial, _, quote, _, _ = _snapshot_row(market_for_minutes(), 0, frame, START, tmp_path)
    assert partial["close"].tolist() == [10.0]
    assert np.isnan(quote).all()


def _fixture(*, block=False):
    dates = [date(2026, 11, day) for day in (5, 6, 9)]
    market = build_market_data_matrix(pl.DataFrame({
        "symbol": ["600000.SH"] * 3, "date": dates, "open": [11.0] * 3,
        "high": [11.5] * 3, "low": [8.0] * 3, "close": [11.0] * 3, "volume": [5000.0] * 3,
    }))
    # A same-session exit signal on the initial buy date cannot sell the new position.
    signals = make_signal_matrix(market.shape, entry=[[1], [0], [0]], exit=[[1], [0], [0]])
    execution = build_market_matrix_from_signals(market, signals)
    execution = replace(execution, entry_close_floor=np.array([[9.0], [np.nan], [np.nan]]),
                        limit_down_locked=np.array([[False], [block], [False]]))
    tape = SameSessionFill(np.array([[10.0], [8.6], [9.8]]), np.array([[9.8], [8.8], [10.0]]))
    return dates, execution, tape


@pytest.mark.parametrize("block", [False, True])
def test_tail_floor_exit_uses_its_own_signal_day_and_retry_never_uses_next_open(block):
    dates, execution, tape = _fixture(block=block)
    result = BacktestEngine(repo=None).simulate_market_matrix(
        execution, MatcherConfig(initial_capital=100000, max_exposure_pct=.6, max_positions=2,
                                 commission_pct=.0002, stamp_tax_pct=.001, slippage_bps=12),
        same_session_fill=tape,
    )
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert str(trade.entry_date) == str(trade.entry_signal_date) == str(dates[0])
    assert str(trade.exit_signal_date) == str(dates[1])
    assert str(trade.exit_date) == str(dates[2 if block else 1])
    assert trade.entry_price == 10.0
    assert trade.exit_price == (9.8 if block else 8.6)
    assert trade.exit_reason == "structure_break"
    assert trade.blocked_exit_days == int(block)
    assert result.equity_curve[0]["positions"] == 1
    assert trade.exit_value < trade.shares * trade.exit_price


def test_tail_buy_at_limit_is_not_filled_and_window_values_unclosed_position():
    _, execution, tape = _fixture()
    result = BacktestEngine(repo=None).simulate_market_matrix(
        replace(execution, limit_up_locked=np.ones(execution.shape, dtype=bool)), MatcherConfig(),
        same_session_fill=tape,
    )
    assert result.stats["execution"]["buy_limit_up"] == 1
    assert all(row["positions"] == 0 for row in result.equity_curve)

    result = BacktestEngine(repo=None).simulate_market_matrix(
        execution, MatcherConfig(), same_session_fill=replace(tape, signal_close=np.full(execution.shape, 10.0)),
    )
    assert result.trades == []
    assert result.equity_curve[-1]["positions"] == 1


def test_timing_evidence_distinguishes_same_day_fills_from_next_open_setting():
    from app.backtest.evidence import audit_trade_timing

    trade = dict(symbol="600000.SH", entry_date="2026-11-05", entry_signal_date="2026-11-05",
                 exit_date="2026-11-06", exit_signal_date="2026-11-06", exit_reason="signal")
    config = {"tail_fill": True, "entry_fill": "open_t+1", "exit_fill": "open_t+1"}
    assert audit_trade_timing([trade], config, "matrix_native")["status"] == "passed"
    delayed = {**trade, "exit_date": "2026-11-09"}
    assert audit_trade_timing([delayed], config, "matrix_native")["status"] == "failed"
    assert audit_trade_timing([{**delayed, "blocked_exit_days": 1}], config, "matrix_native")["status"] == "passed"
    same_as_entry = {**trade, "exit_date": "2026-11-05", "exit_signal_date": "2026-11-05"}
    assert audit_trade_timing([same_as_entry], config, "matrix_native")["status"] == "failed"


def test_closing_reference_prices_and_t_plus_one_hold_through_limit_down():
    dates, execution, _ = _fixture(block=True)
    closes = np.array([[10.2], [8.8], [9.7]])
    booked = replace(execution, close=closes)
    result = BacktestEngine(repo=None).simulate_market_matrix(
        booked, MatcherConfig(initial_capital=100000),
        same_session_fill=SameSessionFill(closes, closes),
    )
    trade, = result.trades
    assert str(trade.entry_date) == str(trade.entry_signal_date) == str(dates[0])
    assert str(trade.exit_signal_date) == str(dates[1])
    assert str(trade.exit_date) == str(dates[2])
    assert trade.entry_price == 10.2 and trade.exit_price == 9.7
    assert trade.exit_reason == "structure_break"
    assert trade.blocked_exit_days == 1

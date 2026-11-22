from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from app.backtest.engine import BacktestEngine, MatcherConfig
from app.backtest.matrix import (
    build_market_data_matrix,
    build_market_matrix_from_signals,
    make_signal_matrix,
)


@pytest.mark.parametrize("method", ["simulate_portfolio", "simulate_portfolio_legacy"])
@pytest.mark.parametrize("reason", ["signal", "max_hold", "end"])
@pytest.mark.parametrize("risk", ["take_profit", "stop_loss"])
def test_known_opening_exit_precedes_later_intraday_risk(method, reason, risk):
    count = 4 if reason == "end" else 6
    rows = []
    for i in range(count):
        opening, high, low = 10.0, 10.0, 10.0
        if i == 3:
            opening, high, low = 10.2, 13.0, 8.0
        rows.append(
            {
                "symbol": "A",
                "date": date(2024, 1, 2) + timedelta(days=i),
                "open": opening,
                "high": high,
                "low": low,
                "close": 10.0,
                "volume": 1000.0,
            }
        )
    cfg = MatcherConfig(
        entry_fill="open_t+1",
        exit_fill="open_t+1",
        fees_pct=0,
        slippage_bps=0,
        take_profit_pct=0.2 if risk == "take_profit" else None,
        stop_loss_pct=0.06 if risk == "stop_loss" else None,
        max_hold_days=2 if reason == "max_hold" else None,
        max_positions=1,
    )
    result = getattr(BacktestEngine(repo=None), method)(
        pl.DataFrame(rows),
        pl.Series([i == 0 for i in range(count)]),
        pl.Series([reason == "signal" and i == 2 for i in range(count)]),
        cfg,
    )
    trade = result.trades[0]
    assert str(trade.exit_date) == "2024-01-05"
    assert trade.exit_reason == reason
    assert trade.exit_price == 10.2


@pytest.mark.parametrize("method", ["simulate_portfolio", "simulate_portfolio_legacy"])
def test_close_schedule_keeps_intraday_risk_before_closing_exit(method):
    rows = [
        {
            "symbol": "A",
            "date": date(2024, 1, 2) + timedelta(days=i),
            "open": 10.0,
            "high": 13.0 if i == 2 else 10.0,
            "low": 10.0,
            "close": 10.0,
            "volume": 1000.0,
        }
        for i in range(4)
    ]
    result = getattr(BacktestEngine(repo=None), method)(
        pl.DataFrame(rows),
        pl.Series([True, False, False, False]),
        pl.Series([False, False, True, False]),
        MatcherConfig(
            entry_fill="open_t+1",
            exit_fill="close_t",
            take_profit_pct=0.2,
            fees_pct=0,
            slippage_bps=0,
        ),
    )
    assert result.trades[0].exit_reason == "take_profit"
    assert result.trades[0].exit_price == 12.0


def test_opening_limit_down_exit_is_deferred_even_when_the_bar_later_rebounds():
    rows = []
    for i in range(6):
        opening, high, low, closing = 10.0, 10.0, 10.0, 10.0
        if i == 3:
            opening, high, low, closing = 9.0, 9.4, 9.0, 9.2
        rows.append(
            {
                "symbol": "000001.SZ",
                "date": date(2024, 1, 2) + timedelta(days=i),
                "open": opening,
                "high": high,
                "low": low,
                "close": closing,
                "raw_close": closing,
                "raw_high": high,
                "raw_low": low,
                "volume": 1000.0,
            }
        )
    market = build_market_data_matrix(
        pl.DataFrame(rows), field_columns={"raw_close", "raw_high", "raw_low", "price_limit_pct"}
    )
    entry = np.zeros(market.shape, dtype=np.uint8)
    entry[0, 0] = 1
    exits = np.zeros(market.shape, dtype=np.uint8)
    exits[2, 0] = 1
    signals = make_signal_matrix(market.shape, entry=entry, exit=exits)
    matrix = build_market_matrix_from_signals(
        market, signals, entry_delay_bars=1, exit_delay_bars=1
    )
    assert matrix.opening_sell_blocked[3, 0]
    trade = (
        BacktestEngine(repo=None)
        .simulate_market_matrix(
            matrix,
            MatcherConfig(entry_fill="open_t+1", exit_fill="open_t+1", fees_pct=0, slippage_bps=0),
        )
        .trades[0]
    )
    assert str(trade.exit_date) == "2024-01-06"
    assert trade.exit_price == 10
    assert trade.blocked_exit_days == 1


def test_unknown_ex_rights_open_reference_defers_once_without_blocking_later_sessions():
    rows = []
    for i in range(6):
        raw = 10.0 if i < 3 else 9.5
        rows.append(
            {
                "symbol": "000001.SZ",
                "date": date(2024, 1, 2) + timedelta(days=i),
                "open": 5.0,
                "high": 5.1,
                "low": 4.9,
                "close": 5.0,
                "raw_close": raw,
                "raw_high": raw * 1.02,
                "raw_low": raw * 0.98,
                "volume": 1000.0,
            }
        )
    market = build_market_data_matrix(
        pl.DataFrame(rows), field_columns={"raw_close", "raw_high", "raw_low", "price_limit_pct"}
    )
    entry = np.zeros(market.shape, dtype=np.uint8)
    entry[0, 0] = 1
    exits = np.zeros(market.shape, dtype=np.uint8)
    exits[2, 0] = 1
    matrix = build_market_matrix_from_signals(
        market,
        make_signal_matrix(market.shape, entry=entry, exit=exits),
        entry_delay_bars=1,
        exit_delay_bars=1,
    )
    assert matrix.opening_sell_blocked[3, 0]
    assert not matrix.opening_sell_blocked[4:, 0].any()
    assert not matrix.opening_sell_blocked.flags.writeable
    result = BacktestEngine(repo=None).simulate_market_matrix(
        matrix,
        MatcherConfig(entry_fill="open_t+1", exit_fill="open_t+1", fees_pct=0, slippage_bps=0),
    )
    assert result.trades[0].exit_date == "2024-01-06"
    assert result.trades[0].blocked_exit_days == 1


def test_opening_guard_defaults_to_none_without_raw_price_context():
    market = build_market_data_matrix(
        pl.DataFrame(
            {
                "symbol": ["A", "A"],
                "date": [date(2024, 1, 2), date(2024, 1, 3)],
                "open": [10.0, 10.0],
                "high": [10.0, 10.0],
                "low": [10.0, 10.0],
                "close": [10.0, 10.0],
                "volume": [1000.0, 1000.0],
            }
        )
    )
    matrix = build_market_matrix_from_signals(market, make_signal_matrix(market.shape))
    assert matrix.opening_sell_blocked is None


@pytest.mark.parametrize("method", ["simulate_portfolio", "simulate_portfolio_legacy"])
@pytest.mark.parametrize("exit_fill", ["close_t", "open_t+1"])
def test_known_opening_take_profit_precedes_a_later_intraday_stop(method, exit_fill):
    rows = [
        {
            "symbol": "A",
            "date": date(2024, 1, 2) + timedelta(days=i),
            "open": 13.0 if i == 2 else 10.0,
            "high": 13.0 if i == 2 else 10.0,
            "low": 8.0 if i == 2 else 10.0,
            "close": 10.0,
            "volume": 1000.0,
        }
        for i in range(5)
    ]
    result = getattr(BacktestEngine(repo=None), method)(
        pl.DataFrame(rows),
        pl.Series([True, False, False, False, False]),
        None,
        MatcherConfig(
            entry_fill="open_t+1",
            exit_fill=exit_fill,
            take_profit_pct=0.2,
            stop_loss_pct=0.06,
            fees_pct=0,
            slippage_bps=0,
        ),
    )
    assert result.trades[0].exit_reason == "take_profit"
    assert result.trades[0].exit_price == 13.0

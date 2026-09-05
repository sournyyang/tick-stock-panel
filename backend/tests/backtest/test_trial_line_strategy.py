from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import polars as pl

from app.backtest.matrix import build_market_data_matrix
from app.strategy.engine import StrategyEngine

STRATEGY_PATH = (
    Path(__file__).resolve().parents[2]
    / "app"
    / "strategy"
    / "builtin"
    / "trial_line_breakout.py"
)


def _signals(rows: list[dict], params: dict | None = None):
    market = build_market_data_matrix(pl.DataFrame(rows))
    strategy = StrategyEngine._load_file(STRATEGY_PATH)
    return strategy.matrix_strategy.compute_signals(market, params or {})


def test_trial_line_entry_and_both_exit_rules():
    start = date(2026, 1, 1)
    values = [
        # 普通阳线, 给下一根试盘线提供前一日量。
        (10.0, 10.2, 9.9, 10.1, 100.0),
        # 三倍量、带明显上影的阳线: 试盘线。
        (10.0, 11.0, 9.9, 10.5, 300.0),
        # 未站上试盘线收盘价, 不买。
        (10.4, 10.5, 10.2, 10.3, 100.0),
        # 实体阳线收盘站上 10.5, 尾盘买入。
        (10.3, 10.7, 10.2, 10.6, 100.0),
        # 阴线但未达到 3 倍标杆量, 不卖。
        (10.7, 10.8, 10.4, 10.5, 899.0),
        # 阴线且成交量达到试盘量的 3 倍, 卖出。
        (10.6, 10.7, 10.3, 10.4, 900.0),
        # 第二个试盘线(前一日量 900 的 3 倍)。
        (10.5, 11.2, 10.4, 10.8, 2700.0),
        # 第二次买入价 11.0。
        (10.7, 11.1, 10.6, 11.0, 300.0),
        # 收盘达到买入价的 110%, 触发收益率卖点。
        (11.0, 12.2, 10.9, 12.1, 300.0),
    ]
    rows = [
        {
            "symbol": "000001.SZ",
            "date": start + timedelta(days=offset),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
        for offset, (open_, high, low, close, volume) in enumerate(values)
    ]

    signals = _signals(rows)

    assert signals.entry[:, 0].tolist() == [0, 0, 0, 1, 0, 0, 0, 1, 0]
    assert signals.exit[:, 0].tolist() == [0, 0, 0, 0, 0, 1, 0, 0, 1]
    assert signals.exit_signal_code[5, 0] == 0
    assert signals.exit_signal_code[8, 0] == 1


def test_trial_line_defaults_to_same_day_close_fills():
    strategy = StrategyEngine._load_file(STRATEGY_PATH)

    assert strategy.meta["default_entry_fill"] == "close_t"
    assert strategy.meta["default_exit_fill"] == "close_t"
    assert strategy.basic_filter["boards"] == ["沪主板", "深主板", "创业板"]
    params = {item["id"]: item["default"] for item in strategy.meta["params"]}
    assert params["trial_volume_multiple"] == 3.0
    assert params["max_trial_volume_multiple"] == 4.5
    assert params["min_trial_close_location_pct"] == 40.0
    assert params["min_pretrial_low_distance_pct"] == 3.0
    assert params["max_pretrial_20d_gain_pct"] == 7.0
    assert params["recent_limit_window_days"] == 10
    assert params["max_recent_limit_ups"] == 1
    assert params["recent_limit_down_window_days"] == 5
    assert params["max_recent_limit_downs"] == 0
    assert params["max_post_trial_volume_pct"] == 50.0
    assert params["min_breakout_margin_pct"] == 0.5
    assert params["take_profit_pct"] == 10.0
    assert params["close_stop_loss_pct"] == 6.0
    assert strategy.stop_loss is None
    assert strategy.max_hold_days == 5


def test_stop_loss_requires_close_confirmation():
    start = date(2026, 1, 1)
    values = [
        (10.0, 10.2, 9.9, 10.1, 100.0),
        (10.0, 11.0, 9.9, 10.5, 300.0),
        # 以 10.6 收盘买入, 6% 止损线为 9.964。
        (10.3, 10.7, 10.2, 10.6, 100.0),
        # 盘中跌破止损线但收盘收回, 不卖。
        (10.1, 10.2, 9.8, 10.0, 100.0),
        # 收盘跌破止损线, 才触发止损卖出。
        (10.0, 10.1, 9.8, 9.9, 100.0),
    ]
    rows = [
        {
            "symbol": "000001.SZ",
            "date": start + timedelta(days=offset),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
        for offset, (open_, high, low, close, volume) in enumerate(values)
    ]

    signals = _signals(rows)

    assert signals.exit[:, 0].tolist() == [0, 0, 0, 0, 1]
    assert signals.exit_signal_code[4, 0] == 2
    assert signals.exit_signal_ids[2] == "signal_close_stop_loss"


def test_trial_breakout_must_arrive_within_entry_window():
    start = date(2026, 1, 1)
    rows = []
    for offset in range(8):
        open_ = 10.0
        close = 10.1
        high = 10.2
        volume = 100.0
        if offset == 1:
            close, high, volume = 10.5, 11.0, 300.0
        if offset == 7:
            close, high = 10.6, 10.7
        rows.append({
            "symbol": "000001.SZ",
            "date": start + timedelta(days=offset),
            "open": open_,
            "high": high,
            "low": 9.9,
            "close": close,
            "volume": volume,
        })

    signals = _signals(rows, {"entry_window_days": 5})

    assert not signals.entry.any()


def test_internal_benchmark_state_resets_after_max_hold():
    start = date(2026, 1, 1)
    values = [
        (10.0, 10.2, 9.9, 10.1, 100.0),
        (10.0, 11.0, 9.9, 10.5, 300.0),
        (10.3, 10.7, 10.2, 10.6, 100.0),
        (10.5, 10.7, 10.3, 10.6, 100.0),
        (10.5, 10.7, 10.3, 10.6, 100.0),
        (10.5, 10.7, 10.3, 10.6, 100.0),
        (10.5, 10.7, 10.3, 10.6, 100.0),
        (10.3, 10.5, 10.2, 10.4, 100.0),
        (10.0, 11.0, 9.9, 10.5, 300.0),
        (10.3, 10.7, 10.2, 10.6, 100.0),
    ]
    rows = [
        {
            "symbol": "000001.SZ",
            "date": start + timedelta(days=offset),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
        for offset, (open_, high, low, close, volume) in enumerate(values)
    ]

    signals = _signals(rows)

    assert signals.entry[:, 0].tolist() == [0, 0, 1, 0, 0, 0, 0, 0, 0, 1]


def test_trial_line_is_rejected_after_two_limit_ups_in_five_days():
    start = date(2026, 1, 1)
    values = [
        (10.0, 10.2, 10.1, 100.0, False),
        (10.1, 11.1, 11.1, 100.0, True),
        (11.1, 12.2, 12.2, 100.0, True),
        (12.0, 13.2, 12.6, 300.0, False),
        (12.5, 12.9, 12.8, 100.0, False),
    ]
    rows = [
        {
            "symbol": "000001.SZ",
            "date": start + timedelta(days=offset),
            "open": open_,
            "high": high,
            "low": open_ - 0.1,
            "close": close,
            "volume": volume,
            "signal_limit_up": is_limit_up,
        }
        for offset, (open_, high, close, volume, is_limit_up) in enumerate(values)
    ]

    assert not _signals(rows).entry.any()


def test_trial_line_is_rejected_after_limit_down_in_previous_five_days():
    start = date(2026, 1, 1)
    values = [
        (10.0, 10.2, 10.1, 100.0, False),
        (10.1, 10.2, 10.0, 100.0, True),
        (10.0, 11.0, 10.5, 300.0, False),
        (10.3, 10.7, 10.6, 100.0, False),
    ]
    rows = [
        {
            "symbol": "000001.SZ",
            "date": start + timedelta(days=offset),
            "open": open_,
            "high": high,
            "low": open_ - 0.1,
            "close": close,
            "volume": volume,
            "signal_limit_down": is_limit_down,
        }
        for offset, (open_, high, close, volume, is_limit_down) in enumerate(values)
    ]

    assert not _signals(rows).entry.any()


def test_trial_line_rejects_volume_above_four_and_half_multiple():
    start = date(2026, 1, 1)
    values = [
        (10.0, 10.2, 10.1, 100.0),
        # 五倍量超过默认 4.5 倍上限, 不作为试盘线。
        (10.0, 11.0, 10.5, 500.0),
        (10.3, 10.7, 10.6, 100.0),
    ]
    rows = [
        {
            "symbol": "000001.SZ",
            "date": start + timedelta(days=offset),
            "open": open_,
            "high": high,
            "low": open_ - 0.1,
            "close": close,
            "volume": volume,
        }
        for offset, (open_, high, close, volume) in enumerate(values)
    ]

    assert not _signals(rows).entry.any()


def test_moderate_trial_volume_receives_higher_score():
    start = date(2026, 1, 1)
    rows = []
    for symbol, trial_volume in (("000001.SZ", 300.0), ("000002.SZ", 400.0)):
        values = [
            (10.0, 10.2, 10.1, 100.0),
            (10.0, 11.0, 10.5, trial_volume),
            (10.3, 10.7, 10.6, 100.0),
        ]
        rows.extend(
            {
                "symbol": symbol,
                "date": start + timedelta(days=offset),
                "open": open_,
                "high": high,
                "low": open_ - 0.1,
                "close": close,
                "volume": volume,
            }
            for offset, (open_, high, close, volume) in enumerate(values)
        )

    signals = _signals(rows)

    assert signals.entry[2].tolist() == [1, 1]
    assert signals.score[2, 0] > signals.score[2, 1]


def test_trial_line_rejects_weak_close_location():
    start = date(2026, 1, 1)
    values = [
        (10.0, 10.2, 9.9, 10.1, 100.0),
        # 虽是放量上影阳线, 但收盘只处于当日振幅约 25% 的低位。
        (10.0, 11.5, 9.9, 10.3, 300.0),
        (10.2, 10.5, 10.1, 10.4, 100.0),
    ]
    rows = [
        {
            "symbol": "000001.SZ",
            "date": start + timedelta(days=offset),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
        for offset, (open_, high, low, close, volume) in enumerate(values)
    ]

    assert not _signals(rows).entry.any()


def test_trial_line_rejects_setup_near_twenty_day_low():
    start = date(2026, 1, 1)
    rows = [
        {
            "symbol": "000001.SZ",
            "date": start + timedelta(days=offset),
            "open": 10.0,
            "high": 10.2,
            "low": 10.0,
            "close": 10.1,
            "volume": 100.0,
        }
        for offset in range(20)
    ]
    rows.extend([
        {
            "symbol": "000001.SZ", "date": start + timedelta(days=20),
            "open": 10.1, "high": 11.0, "low": 10.0, "close": 10.5, "volume": 300.0,
        },
        {
            "symbol": "000001.SZ", "date": start + timedelta(days=21),
            "open": 10.4, "high": 10.7, "low": 10.3, "close": 10.6, "volume": 100.0,
        },
    ])

    assert not _signals(rows).entry.any()


def test_trial_line_rejects_more_than_seven_percent_twenty_day_runup():
    start = date(2026, 1, 1)
    rows = []
    for offset in range(20):
        close = 9.0 + offset / 19
        rows.append({
            "symbol": "000001.SZ",
            "date": start + timedelta(days=offset),
            "open": close - 0.05,
            "high": close + 0.1,
            "low": close - 0.1,
            "close": close,
            "volume": 100.0,
        })
    rows.extend([
        {
            "symbol": "000001.SZ", "date": start + timedelta(days=20),
            "open": 10.0, "high": 10.9, "low": 9.9, "close": 10.4, "volume": 300.0,
        },
        {
            "symbol": "000001.SZ", "date": start + timedelta(days=21),
            "open": 10.4, "high": 10.7, "low": 10.3, "close": 10.5, "volume": 100.0,
        },
    ])

    assert not _signals(rows).entry.any()


def test_trial_line_rejects_weak_breakout_margin():
    start = date(2026, 1, 1)
    values = [
        (10.0, 10.2, 9.9, 10.1, 100.0),
        (10.0, 11.0, 9.9, 10.5, 300.0),
        # 只高出试盘收盘约 0.38%, 未达到默认 0.5%。
        (10.3, 10.6, 10.2, 10.54, 100.0),
    ]
    rows = [
        {
            "symbol": "000001.SZ",
            "date": start + timedelta(days=offset),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
        for offset, (open_, high, low, close, volume) in enumerate(values)
    ]

    assert not _signals(rows).entry.any()


def test_trial_line_uses_ten_days_for_recent_limit_ups():
    start = date(2026, 1, 1)
    rows = []
    for offset in range(12):
        row = {
            "symbol": "000001.SZ",
            "date": start + timedelta(days=offset),
            "open": 10.0,
            "high": 10.2,
            "low": 9.9,
            "close": 10.1,
            "volume": 100.0,
            "signal_limit_up": offset in {1, 5},
        }
        if offset == 10:
            row.update(open=10.0, high=11.0, close=10.5, volume=300.0)
        if offset == 11:
            row.update(open=10.3, high=10.7, close=10.6, volume=100.0)
        rows.append(row)

    assert not _signals(rows).entry.any()


def test_gap_down_small_bullish_candle_is_not_a_trial_line():
    start = date(2026, 1, 1)
    values = [
        (13.80, 14.00, 13.95, 100.0),
        # 收盘略高于开盘, 但相对前收盘大跌, 不能视为试盘阳线。
        (12.56, 13.11, 12.58, 300.0),
        (12.45, 12.79, 12.65, 100.0),
    ]
    rows = [
        {
            "symbol": "600449.SH",
            "date": start + timedelta(days=offset),
            "open": open_,
            "high": high,
            "low": open_ - 0.1,
            "close": close,
            "volume": volume,
        }
        for offset, (open_, high, close, volume) in enumerate(values)
    ]

    assert not _signals(rows).entry.any()


def test_direct_breakout_requires_half_volume_contraction():
    start = date(2026, 1, 1)
    values = [
        (10.0, 10.2, 10.1, 100.0),
        (10.0, 11.0, 10.5, 300.0),
        # 首日突破量超过试盘量的一半, 不买并作废本次试盘。
        (10.3, 10.7, 10.6, 151.0),
        (10.4, 10.8, 10.7, 100.0),
    ]
    rows = [
        {
            "symbol": "000001.SZ",
            "date": start + timedelta(days=offset),
            "open": open_,
            "high": high,
            "low": open_ - 0.1,
            "close": close,
            "volume": volume,
        }
        for offset, (open_, high, close, volume) in enumerate(values)
    ]

    assert not _signals(rows).entry.any()


def test_delayed_breakout_uses_average_pullback_volume():
    start = date(2026, 1, 1)
    values = [
        (10.0, 10.2, 10.1, 100.0),
        (10.0, 11.0, 10.5, 300.0),
        # 两个洗盘日均量正好是试盘量的 50%。
        (10.4, 10.5, 10.3, 160.0),
        (10.3, 10.5, 10.4, 140.0),
        # 突破日成交量不参与此前洗盘均量计算。
        (10.3, 10.7, 10.6, 200.0),
    ]
    rows = [
        {
            "symbol": "000001.SZ",
            "date": start + timedelta(days=offset),
            "open": open_,
            "high": high,
            "low": open_ - 0.1,
            "close": close,
            "volume": volume,
        }
        for offset, (open_, high, close, volume) in enumerate(values)
    ]

    signals = _signals(rows)

    assert signals.entry[:, 0].tolist() == [0, 0, 0, 0, 1]


def test_first_unbuyable_breakout_invalidates_trial_setup():
    start = date(2026, 1, 1)
    values = [
        (10.0, 10.2, 10.1, 100.0),
        (10.0, 11.0, 10.5, 300.0),
        # 首次站上试盘收盘价, 但开收相等, 不是实体阳线。
        (11.0, 11.0, 11.0, 100.0),
        (10.3, 10.5, 10.4, 100.0),
        # 后续再次突破不能重新触发买点。
        (10.4, 10.7, 10.6, 100.0),
    ]
    rows = [
        {
            "symbol": "000001.SZ",
            "date": start + timedelta(days=offset),
            "open": open_,
            "high": high,
            "low": open_ - 0.1,
            "close": close,
            "volume": volume,
        }
        for offset, (open_, high, close, volume) in enumerate(values)
    ]

    assert not _signals(rows).entry.any()


def test_setup_is_invalid_after_price_trades_ten_percent_above_trial_close():
    start = date(2026, 1, 1)
    values = [
        (10.0, 10.2, 10.1, 100.0),
        (10.0, 11.0, 10.5, 300.0),
        # 盘中超过试盘收盘价 10%, 即使没有收盘突破也作废。
        (10.4, 11.6, 10.4, 100.0),
        (10.4, 10.7, 10.6, 100.0),
    ]
    rows = [
        {
            "symbol": "000001.SZ",
            "date": start + timedelta(days=offset),
            "open": open_,
            "high": high,
            "low": open_ - 0.1,
            "close": close,
            "volume": volume,
        }
        for offset, (open_, high, close, volume) in enumerate(values)
    ]

    assert not _signals(rows).entry.any()

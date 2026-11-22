from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import polars as pl

from app.backtest.matrix import build_market_data_matrix
from app.strategy.engine import StrategyEngine

REPO_ROOT = Path(__file__).resolve().parents[3]
STRATEGY_DIR = REPO_ROOT / "backend" / "app" / "strategy" / "seeds"


def _load(name: str):
    strategy = StrategyEngine._load_file(STRATEGY_DIR / name)
    assert strategy.execution_backend == "matrix_native"
    assert strategy.matrix_strategy is not None
    return strategy


def test_sequoia_turtle_trade_matches_source_entry_formula():
    start = date(2026, 1, 1)
    rows = []
    for offset in range(23):
        close = 9.8
        open_ = 9.7
        high = 10.0
        amount = 100_100_000.0
        if offset == 20:
            close, open_, high = 10.5, 10.1, 10.8
        elif offset == 21:
            # Equal to the new 20-day high: strict breakout must fail.
            close, open_, high = 10.8, 10.6, 10.9
        elif offset == 22:
            # Price conditions pass but source's strict amount gate fails at equality.
            close, open_, high, amount = 11.0, 10.9, 11.1, 100_000_000.0
        rows.append({
            "symbol": "000001.SZ",
            "date": start + timedelta(days=offset),
            "open": open_,
            "high": high,
            "low": open_ - 0.1,
            "close": close,
            "volume": 1_000_000.0,
            "amount": amount,
        })

    market = build_market_data_matrix(pl.DataFrame(rows), field_columns={"amount"})
    strategy = _load("sequoia_turtle_trade.py")
    signals = strategy.matrix_strategy.compute_signals(market, {})

    assert signals.entry[:, 0].tolist() == [0] * 20 + [1, 0, 0]
    assert signals.score[20, 0] > 0
    assert strategy.max_hold_days == 5
    assert strategy.exit_signals == []


def test_sequoia_ma_volume_matches_source_entry_formula():
    start = date(2026, 1, 1)
    rows = []
    closes = [10.0] * 16 + [9.0] * 4 + [20.0, 20.0]
    volumes = [100.0] * 20 + [1_000.0, 100.0]
    for offset, (close, volume) in enumerate(zip(closes, volumes, strict=True)):
        rows.append({
            "symbol": "000001.SZ",
            "date": start + timedelta(days=offset),
            "open": close - 0.1,
            "high": close + 0.1,
            "low": close - 0.2,
            "close": close,
            "volume": volume,
            "amount": volume * close,
        })

    market = build_market_data_matrix(pl.DataFrame(rows))
    strategy = _load("sequoia_ma_volume.py")
    signals = strategy.matrix_strategy.compute_signals(market, {})

    assert signals.entry[:, 0].tolist() == [0] * 20 + [1, 0]
    assert signals.score[20, 0] > 1.5
    assert strategy.max_hold_days == 5
    assert strategy.exit_signals == []


def test_sequoia_high_tight_flag_matches_source_entry_formula():
    start = date(2026, 1, 1)
    rows = []
    for symbol, last_volume in (("000001.SZ", 59.0), ("000002.SZ", 60.0)):
        for offset in range(40):
            if offset < 30:
                low = 10.0 + offset * 0.2
                high = low + 0.5
            else:
                low = 16.0 + (offset % 2) * 0.1
                high = 17.0 + (offset % 3) * 0.1
            volume = last_volume if offset == 39 else 100.0
            rows.append({
                "symbol": symbol,
                "date": start + timedelta(days=offset),
                "open": low + 0.2,
                "high": high,
                "low": low,
                "close": low + 0.3,
                "volume": volume,
                "amount": volume * (low + 0.3),
            })

    market = build_market_data_matrix(pl.DataFrame(rows))
    strategy = _load("sequoia_high_tight_flag.py")
    signals = strategy.matrix_strategy.compute_signals(market, {})
    first = market.symbols.index("000001.SZ")
    second = market.symbols.index("000002.SZ")

    assert signals.entry[-1, first] == 1
    assert signals.entry[-1, second] == 0
    assert signals.score[-1, first] > 1.6
    assert strategy.max_hold_days == 5
    assert strategy.exit_signals == []


def test_sequoia_limit_up_shakeout_matches_source_entry_formula():
    start = date(2026, 1, 1)
    rows = []
    for symbol, last_volume in (("000001.SZ", 201.0), ("000002.SZ", 200.0)):
        values = [
            (10.0, 10.1, 9.9, 10.0, 100.0, False),
            (10.5, 11.0, 10.5, 11.0, 100.0, True),
            (11.5, 11.6, 11.0, 11.2, last_volume, False),
        ]
        for offset, (open_, high, low, close, volume, limit_up) in enumerate(values):
            rows.append({
                "symbol": symbol,
                "date": start + timedelta(days=offset),
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "amount": volume * close,
                "signal_limit_up": limit_up,
            })

    market = build_market_data_matrix(pl.DataFrame(rows))
    strategy = _load("sequoia_limit_up_shakeout.py")
    signals = strategy.matrix_strategy.compute_signals(market, {})
    first = market.symbols.index("000001.SZ")
    second = market.symbols.index("000002.SZ")

    assert signals.entry[-1, first] == 1
    assert signals.entry[-1, second] == 0
    assert signals.score[-1, first] > 2.0
    assert strategy.max_hold_days == 5
    assert strategy.exit_signals == []


def test_sequoia_uptrend_limit_down_matches_source_entry_formula():
    start = date(2026, 1, 1)
    rows = []
    for symbol, last_volume in (("000001.SZ", 1_000.0), ("000002.SZ", 200.0)):
        for offset in range(61):
            close = 10.0 + offset * 0.1
            volume = 100.0
            limit_down = False
            if offset == 60:
                close = 14.3
                volume = last_volume
                limit_down = True
            rows.append({
                "symbol": symbol,
                "date": start + timedelta(days=offset),
                "open": close + (0.2 if limit_down else -0.1),
                "high": close + 0.2,
                "low": close - 0.2,
                "close": close,
                "volume": volume,
                "amount": volume * close,
                "signal_limit_down": limit_down,
            })

    market = build_market_data_matrix(pl.DataFrame(rows))
    strategy = _load("sequoia_uptrend_limit_down.py")
    signals = strategy.matrix_strategy.compute_signals(market, {})
    first = market.symbols.index("000001.SZ")
    second = market.symbols.index("000002.SZ")

    assert signals.entry[-1, first] == 1
    assert signals.entry[-1, second] == 0
    assert signals.score[-1, first] > 2.0
    assert strategy.max_hold_days == 5
    assert strategy.exit_signals == []


def test_sequoia_rps_breakout_matches_source_entry_formula():
    start = date(2025, 1, 1)
    rows = []
    symbols = [f"0000{asset:02d}.SZ" for asset in range(10)]
    for asset, symbol in enumerate(symbols):
        for offset in range(121):
            close = 10.0 + offset * (asset + 1) * 0.01
            rows.append({
                "symbol": symbol,
                "date": start + timedelta(days=offset),
                "open": close - 0.1,
                "high": close + 0.1,
                "low": close - 0.2,
                "close": close,
                "volume": 100.0,
                "amount": 1_000.0,
            })

    market = build_market_data_matrix(pl.DataFrame(rows))
    strategy = _load("sequoia_rps_breakout.py")
    signals = strategy.matrix_strategy.compute_signals(market, {})
    selected = [
        market.symbols[asset_id]
        for asset_id in range(len(market.symbols))
        if signals.entry[-1, asset_id]
    ]

    assert selected == symbols[-2:]
    assert signals.score[-1, market.symbols.index(symbols[-2])] == 90.0
    assert signals.score[-1, market.symbols.index(symbols[-1])] == 100.0
    assert strategy.max_hold_days == 5
    assert strategy.exit_signals == []

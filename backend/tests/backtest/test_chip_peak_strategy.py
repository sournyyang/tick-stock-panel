from __future__ import annotations

from pathlib import Path

import numpy as np

from app.backtest.chip_peak import compute_chip_peak_entries, compute_market_breadth_mask
from app.strategy.engine import StrategyEngine


def _run(
    close: list[float],
    *,
    turnover: float = 20.0,
    volume_ratio: float = 2.0,
    confirmation_bars: int = 1,
    max_breakout_extension_pct: float = 1_000.0,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(close, dtype=np.float32)[:, None]
    return compute_chip_peak_entries(
        values * 1.01,
        values * 0.99,
        values,
        np.full_like(values, turnover),
        np.full_like(values, volume_ratio),
        min_history_bars=20,
        position_window=20,
        volume_ratio_min=1.5,
        concentration_max_pct=25.0,
        peak_share_min_pct=5.0,
        peak_position_max_pct=60.0,
        max_initial_residual_pct=10.0,
        confirmation_bars=confirmation_bars,
        max_breakout_extension_pct=max_breakout_extension_pct,
    )


def test_chip_peak_breakout_emits_only_after_causal_initialization():
    closes = [10.0] * 20 + [10.1, 10.2, 10.3, 10.8]
    entry, score = _run(closes)

    assert not entry[:20].any()
    assert entry[-1, 0] == 1
    assert score[-1, 0] > 0


def test_chip_peak_breakout_fails_closed_without_turnover_history():
    closes = [10.0] * 20 + [10.8]
    values = np.asarray(closes, dtype=np.float32)[:, None]
    entry, _ = compute_chip_peak_entries(
        values * 1.01,
        values * 0.99,
        values,
        np.full_like(values, np.nan),
        np.full_like(values, 2.0),
        min_history_bars=20,
        position_window=20,
        volume_ratio_min=1.5,
        concentration_max_pct=25.0,
        peak_share_min_pct=5.0,
        peak_position_max_pct=60.0,
        max_initial_residual_pct=10.0,
    )

    assert not entry.any()


def test_chip_peak_history_is_unchanged_by_appended_future_prices():
    history = [10.0] * 20 + [10.8, 10.6, 10.9]
    original, _ = _run(history)
    extended, _ = _run([*history, 300.0, 2.0, 250.0])

    np.testing.assert_array_equal(original, extended[: len(history)])


def test_confirmed_breakout_waits_for_a_second_close_above_cost_area():
    closes = [10.0] * 20 + [10.8, 11.0]
    entry, _ = _run(closes, confirmation_bars=2)

    assert entry[-2, 0] == 0
    assert entry[-1, 0] == 1


def test_confirmed_breakout_rejects_an_overextended_confirmation_close():
    closes = [10.0] * 20 + [10.8, 13.0]
    entry, _ = _run(
        closes,
        confirmation_bars=2,
        max_breakout_extension_pct=5.0,
    )

    assert not entry.any()


def test_pullback_mode_waits_for_a_recovery_near_the_cost_area():
    closes = [10.0] * 20 + [10.8, 10.0, 11.0]
    values = np.asarray(closes, dtype=np.float32)[:, None]
    volume_ratio = np.full_like(values, 2.0)
    volume_ratio[-2:] = 1.0
    entry, _ = compute_chip_peak_entries(
        values * 1.01,
        values * 0.99,
        values,
        np.full_like(values, 20.0),
        volume_ratio,
        min_history_bars=20,
        position_window=20,
        volume_ratio_min=1.5,
        concentration_max_pct=25.0,
        peak_share_min_pct=5.0,
        peak_position_max_pct=60.0,
        max_initial_residual_pct=10.0,
        max_breakout_extension_pct=3.0,
        pullback_window_bars=10,
        pullback_touch_buffer_pct=2.0,
        pullback_volume_ratio_max=1.2,
    )

    assert not entry[:-1].any()
    assert entry[-1, 0] == 1


def test_market_breadth_mask_uses_only_valid_assets_on_each_date():
    close = np.asarray([[11.0, 9.0, np.nan], [11.0, 12.0, 9.0]], dtype=np.float32)
    ma60 = np.asarray([[10.0, 10.0, np.nan], [10.0, 10.0, 10.0]], dtype=np.float32)

    mask = compute_market_breadth_mask(close, ma60, 0.5)

    np.testing.assert_array_equal(mask, np.asarray([True, True]))


def test_builtin_registry_loads_chip_peak_strategy_without_errors():
    builtin_dir = Path(__file__).resolve().parents[2] / "app" / "strategy" / "seeds"
    engine = StrategyEngine(strategy_dirs=[builtin_dir])

    assert engine.has("chip_peak_breakout")
    assert engine.load_errors() == []

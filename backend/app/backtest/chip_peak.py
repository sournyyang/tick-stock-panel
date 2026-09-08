"""Causal chip-distribution calculations for matrix backtests.

The kernel uses a fixed logarithmic price grid. Its bin boundaries therefore
do not depend on a future high or low from the requested backtest window.
"""
from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit, prange
except ImportError:  # pragma: no cover - exercised only on unsupported platforms
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def decorator(func):
            return func

        return decorator

    prange = range


_PRICE_MIN = 0.25
_PRICE_MAX = 512.0
_PRICE_BINS = 384
_LOG_MIN = math.log(_PRICE_MIN)
_LOG_STEP = (math.log(_PRICE_MAX) - _LOG_MIN) / _PRICE_BINS


def compute_market_breadth_mask(
    close: np.ndarray,
    moving_average: np.ndarray,
    min_fraction: float,
) -> np.ndarray:
    """Return dates where enough valid assets close above their moving average."""
    if close.shape != moving_average.shape or close.ndim != 2:
        raise ValueError("market breadth inputs must be equally shaped 2D matrices")
    valid = np.isfinite(close) & np.isfinite(moving_average)
    valid_count = valid.sum(axis=1)
    above_count = (valid & (close > moving_average)).sum(axis=1)
    breadth = np.zeros(close.shape[0], dtype=np.float32)
    np.divide(above_count, valid_count, out=breadth, where=valid_count > 0)
    return breadth >= min(max(float(min_fraction), 0.0), 1.0)


@njit(cache=True)
def _price_bin(price: float) -> int:
    if price <= _PRICE_MIN:
        return 0
    if price >= _PRICE_MAX:
        return _PRICE_BINS - 1
    return min(max(int((math.log(price) - _LOG_MIN) / _LOG_STEP), 0), _PRICE_BINS - 1)


@njit(cache=True)
def _bin_mid(bin_id: int) -> float:
    return math.exp(_LOG_MIN + (bin_id + 0.5) * _LOG_STEP)


@njit(cache=True)
def _profile_metrics(diff: np.ndarray) -> tuple[float, float, float, float]:
    """Return POC, COST(15), COST(85), and local peak share."""
    hist = np.empty(_PRICE_BINS, dtype=np.float64)
    running = 0.0
    total = 0.0
    poc_id = 0
    poc_value = -1.0
    for bin_id in range(_PRICE_BINS):
        running += diff[bin_id]
        value = max(running, 0.0)
        hist[bin_id] = value
        total += value
        if value > poc_value:
            poc_value = value
            poc_id = bin_id

    if total <= 0.0:
        return math.nan, math.nan, math.nan, math.nan

    cost15_id = 0
    cost85_id = _PRICE_BINS - 1
    cumulative = 0.0
    found15 = False
    for bin_id in range(_PRICE_BINS):
        cumulative += hist[bin_id]
        if not found15 and cumulative >= total * 0.15:
            cost15_id = bin_id
            found15 = True
        if cumulative >= total * 0.85:
            cost85_id = bin_id
            break

    peak_mass = 0.0
    for bin_id in range(max(0, poc_id - 1), min(_PRICE_BINS, poc_id + 2)):
        peak_mass += hist[bin_id]
    return (
        _bin_mid(poc_id),
        _bin_mid(cost15_id),
        _bin_mid(cost85_id),
        peak_mass / total,
    )


@njit(cache=True, parallel=True)
def _chip_peak_signal_kernel(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    turnover_rate: np.ndarray,
    volume_ratio: np.ndarray,
    min_history_bars: int,
    position_window: int,
    volume_ratio_min: float,
    concentration_max_pct: float,
    peak_share_min_pct: float,
    peak_position_max_pct: float,
    max_initial_residual_pct: float,
    confirmation_bars: int,
    max_breakout_extension_pct: float,
    pullback_window_bars: int,
    pullback_touch_buffer_pct: float,
    pullback_volume_ratio_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    rows, assets = close.shape
    entry = np.zeros((rows, assets), dtype=np.uint8)
    score = np.zeros((rows, assets), dtype=np.float32)

    for asset_id in prange(assets):
        # A difference array makes each uniform daily high-low allocation O(1).
        diff = np.zeros(_PRICE_BINS + 1, dtype=np.float64)
        scale = 1.0
        residual = 1.0
        valid_bars = 0
        previous_close = math.nan
        confirmation_count = 0
        setup_volume_ratio = 0.0
        pullback_age = 0

        for row_id in range(rows):
            current_close = close[row_id, asset_id]
            current_high = high[row_id, asset_id]
            current_low = low[row_id, asset_id]
            if not (
                math.isfinite(current_close)
                and math.isfinite(current_high)
                and math.isfinite(current_low)
                and current_close > 0.0
                and current_high > 0.0
                and current_low > 0.0
            ):
                continue

            valid_bars += 1
            turnover = turnover_rate[row_id, asset_id]
            if not math.isfinite(turnover) or turnover < 0.0:
                turnover = 0.0
            turnover = min(turnover / 100.0, 1.0)
            decay = 1.0 - turnover
            residual *= decay

            if decay <= 1e-12:
                diff[:] = 0.0
                scale = 1.0
            else:
                scale *= decay
                if scale < 1e-8:
                    diff *= scale
                    scale = 1.0

            if turnover > 0.0:
                low_id = _price_bin(min(current_low, current_high))
                high_id = _price_bin(max(current_low, current_high))
                per_bin = (turnover / scale) / (high_id - low_id + 1)
                diff[low_id] += per_bin
                diff[high_id + 1] -= per_bin

            vol_ratio = volume_ratio[row_id, asset_id]
            initialized = (
                valid_bars >= min_history_bars
                and residual * 100.0 <= max_initial_residual_pct
                and math.isfinite(previous_close)
            )
            potential_breakout = (
                initialized
                and math.isfinite(vol_ratio)
                and vol_ratio >= volume_ratio_min
                and current_close > previous_close
            )
            if (
                not potential_breakout
                and confirmation_count == 0
                and pullback_age == 0
            ):
                previous_close = current_close
                continue

            poc, cost15, cost85, peak_share = _profile_metrics(diff)
            midpoint = (cost15 + cost85) * 0.5
            if not math.isfinite(midpoint) or midpoint <= 0.0:
                previous_close = current_close
                continue
            concentration_pct = (cost85 - cost15) / midpoint * 100.0

            window_start = max(0, row_id - position_window + 1)
            range_low = current_low
            range_high = current_high
            for history_id in range(window_start, row_id + 1):
                historical_low = low[history_id, asset_id]
                historical_high = high[history_id, asset_id]
                if math.isfinite(historical_low):
                    range_low = min(range_low, historical_low)
                if math.isfinite(historical_high):
                    range_high = max(range_high, historical_high)
            price_range = range_high - range_low
            peak_position_pct = (
                (poc - range_low) / price_range * 100.0
                if price_range > 0.0
                else 100.0
            )

            base_quality = (
                concentration_pct <= concentration_max_pct
                and peak_share * 100.0 >= peak_share_min_pct
                and peak_position_pct <= peak_position_max_pct
            )
            extension_pct = (current_close / cost85 - 1.0) * 100.0
            holds_breakout = (
                base_quality
                and current_close > cost85
                and extension_pct <= max_breakout_extension_pct
            )
            breakout = (
                base_quality
                and potential_breakout
                and previous_close <= cost85
                and current_close > cost85
            )

            signal = False
            score_volume_ratio = vol_ratio
            if pullback_window_bars > 0:
                if pullback_age > 0:
                    pullback_age += 1
                    pullback_touch = (
                        current_low
                        <= cost85 * (1.0 + pullback_touch_buffer_pct / 100.0)
                    )
                    pullback_recovered = (
                        base_quality
                        and pullback_touch
                        and current_close >= cost85
                        and current_close > previous_close
                        and extension_pct <= max_breakout_extension_pct
                        and math.isfinite(vol_ratio)
                        and vol_ratio <= pullback_volume_ratio_max
                    )
                    if pullback_recovered:
                        signal = True
                        pullback_age = 0
                    elif current_close < cost15 or pullback_age > pullback_window_bars:
                        pullback_age = 0
                elif breakout:
                    pullback_age = 1
            elif confirmation_bars <= 1:
                signal = holds_breakout and breakout
            elif confirmation_count > 0:
                if holds_breakout:
                    confirmation_count += 1
                    score_volume_ratio = setup_volume_ratio
                    if math.isfinite(vol_ratio):
                        score_volume_ratio = max(setup_volume_ratio, vol_ratio)
                    if confirmation_count >= confirmation_bars:
                        signal = True
                        confirmation_count = 0
                else:
                    confirmation_count = 0
            elif breakout:
                confirmation_count = 1
                setup_volume_ratio = vol_ratio

            if signal:
                entry[row_id, asset_id] = 1
                score[row_id, asset_id] = np.float32(
                    max(concentration_max_pct - concentration_pct, 0.0)
                    + peak_share * 100.0
                    + min(score_volume_ratio, 5.0) * 5.0
                    + max(peak_position_max_pct - peak_position_pct, 0.0) * 0.25
                )
            previous_close = current_close

    return entry, score


def compute_chip_peak_entries(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    turnover_rate: np.ndarray,
    volume_ratio: np.ndarray,
    *,
    min_history_bars: int,
    position_window: int,
    volume_ratio_min: float,
    concentration_max_pct: float,
    peak_share_min_pct: float,
    peak_position_max_pct: float,
    max_initial_residual_pct: float,
    confirmation_bars: int = 1,
    max_breakout_extension_pct: float = 1_000.0,
    pullback_window_bars: int = 0,
    pullback_touch_buffer_pct: float = 0.0,
    pullback_volume_ratio_max: float = 1_000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute causal close-of-day entries and their cross-sectional scores."""
    shape = close.shape
    for name, values in {
        "high": high,
        "low": low,
        "turnover_rate": turnover_rate,
        "volume_ratio": volume_ratio,
    }.items():
        if values.shape != shape:
            raise ValueError(f"{name} shape {values.shape} does not match close {shape}")
    return _chip_peak_signal_kernel(
        np.asarray(high, dtype=np.float32),
        np.asarray(low, dtype=np.float32),
        np.asarray(close, dtype=np.float32),
        np.asarray(turnover_rate, dtype=np.float32),
        np.asarray(volume_ratio, dtype=np.float32),
        max(20, int(min_history_bars)),
        max(20, int(position_window)),
        max(0.0, float(volume_ratio_min)),
        max(0.0, float(concentration_max_pct)),
        max(0.0, float(peak_share_min_pct)),
        min(max(float(peak_position_max_pct), 0.0), 100.0),
        min(max(float(max_initial_residual_pct), 0.0), 100.0),
        max(1, int(confirmation_bars)),
        max(0.0, float(max_breakout_extension_pct)),
        max(0, int(pullback_window_bars)),
        max(0.0, float(pullback_touch_buffer_pct)),
        max(0.0, float(pullback_volume_ratio_max)),
    )

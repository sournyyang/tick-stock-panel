"""Replay one pre-close daily strategy decision per market session."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, time
from types import MappingProxyType

import numpy as np
import polars as pl

from app.backtest.matrix import (
    MatrixStrategyPipeline,
    build_market_matrix_from_signals,
    make_signal_matrix,
    slice_market_data_matrix,
    valid_shift,
)
from app.price_limits import numpy_limit_price
from app.services.minute_adjust import compute_ratio_pairs

STRATEGY_ID = "custom_triple_volume_two60_body25"
SIGNAL_TIME = "14:40"
LAST_INPUT_BAR = time(14, 39)
FILL_BAR = time(14, 42)


def _aggregate_minutes(frame: pl.DataFrame, day: date) -> pl.DataFrame:
    if frame.is_empty():
        raise ValueError(f"{day} 缺少本地分钟K，无法回放当天尾盘")
    clock = pl.col("datetime").dt.time()
    before_signal = clock <= LAST_INPUT_BAR
    regular_input = (
        clock.is_between(time(9, 31), time(11, 30))
        | clock.is_between(time(13, 1), LAST_INPUT_BAR)
    )
    fill_bar = clock == FILL_BAR
    valid_price = pl.all_horizontal(
        pl.col(c).is_finite() & (pl.col(c) > 0) for c in ("open", "high", "low", "close")
    ) & pl.col("volume").is_finite() & (pl.col("volume") >= 0)
    return (
        frame.filter(
            (pl.col("datetime").dt.date() == day)
            & (clock.is_between(time(9, 30), time(11, 30)) | clock.is_between(time(13, 1), time(15)))
            & (pl.col("datetime").dt.second() == 0)
            & (pl.col("datetime").dt.microsecond() == 0)
        )
        .sort("datetime")
        .group_by("symbol")
        .agg(
            pl.col("datetime").n_unique().alias("unique_minutes"),
            pl.len().alias("rows"),
            valid_price.fill_null(False).all().alias("valid"),
            (pl.col("high") >= pl.max_horizontal("open", "close", "low")).all().alias("high_valid"),
            (pl.col("low") <= pl.min_horizontal("open", "close", "high")).all().alias("low_valid"),
            pl.col("volume").sum().alias("day_volume"),
            pl.col("close").filter(clock == time(15)).last().alias("final_close"),
            pl.col("high").filter(before_signal).max(),
            pl.col("low").filter(before_signal).min(),
            pl.col("close").filter(before_signal).last(),
            pl.col("volume").filter(before_signal).sum(),
            regular_input.sum().alias("input_minutes"),
            pl.col("open").filter(fill_bar & (pl.col("volume") > 0)).first().alias("fill_open"),
        )
    )


def _snapshot_row(market, row: int, frame: pl.DataFrame, day: date, data_dir):
    current_raw = market.fields["raw_close"][row].astype(np.float64)
    current_volume = market.volume[row].astype(np.float64)
    expected = np.isfinite(current_raw) & np.isfinite(current_volume) & (current_volume > 0)
    aggregated = _aggregate_minutes(frame, day)
    pairs = compute_ratio_pairs(frame, data_dir).select("symbol", "_ratio")
    ordered = (
        pl.DataFrame({"symbol": market.symbols})
        .join(aggregated, on="symbol", how="left")
        .join(pairs, on="symbol", how="left")
    )
    data = {name: ordered[name].to_numpy() for name in ordered.columns if name != "symbol"}
    # 241 clock slots include the separate opening-auction row. A store without
    # that row can still have 240, provided the daily volume and close reconcile.
    verified = (
        ordered["valid"].fill_null(False).to_numpy()
        & ordered["high_valid"].fill_null(False).to_numpy()
        & ordered["low_valid"].fill_null(False).to_numpy()
        & (data["rows"] == data["unique_minutes"])
        & (data["input_minutes"] == 219)
        & (np.abs(data["day_volume"] - current_volume) <= np.maximum(0.02, current_volume * 0.005))
        & (np.abs(data["final_close"] - current_raw) <= 0.011)
    )
    missing = int(np.count_nonzero(expected & ~verified))
    if missing or not expected.any():
        raise ValueError(f"{day} 分钟K覆盖不足，{missing}/{int(expected.sum())} 只有成交的股票未通过分钟量价核对")
    # Use the repo's corporate-action projection. The official close only
    # validates that the historical minute and daily datasets use the same basis.
    factor = data["_ratio"]
    with np.errstate(divide="ignore", invalid="ignore"):
        checked_factor = market.close[row].astype(np.float64) / current_raw
    invalid_factor = expected & (~np.isfinite(factor) | (factor <= 0)
                                 | ~np.isclose(factor, checked_factor, rtol=2e-5, atol=1e-7))
    if invalid_factor.any():
        raise ValueError(f"{day} 有{int(invalid_factor.sum())}只股票的分钟与日线复权口径不一致")
    observed = verified & np.isfinite(data["volume"]) & (data["volume"] > 0)
    partial = {
        name: np.where(observed, data[name] * factor, np.nan).astype(np.float32)
        for name in ("high", "low", "close")
    }
    # The official session open is public by 14:40. Keep it even when the first
    # stored minute contains a different first print.
    partial["open"] = np.where(observed, market.open[row], np.nan)
    partial["high"] = np.maximum(partial["high"], partial["open"])
    partial["low"] = np.minimum(partial["low"], partial["open"])
    partial["volume"] = np.where(observed, data["volume"], np.nan).astype(np.float32)
    raw_close = np.where(observed, data["close"], np.nan).astype(np.float32)
    return partial, raw_close, data["fill_open"], factor, int(expected.sum())


def _with_partial_bar(market, row, partial, raw_close):
    history = slice_market_data_matrix(market, 0, row + 1)

    def replace_last(array, last):
        prior = array.copy()
        prior[-1] = last
        prior.flags.writeable = False
        return prior

    visible = np.isfinite(partial["close"])
    safe_fields = {name: history.fields[name] for name in ("price_limit_pct", "listing_date")
                   if name in history.fields}
    safe_fields["raw_close"] = replace_last(history.fields["raw_close"], raw_close)
    return replace(
        history,
        **{name: replace_last(getattr(history, name), value) for name, value in partial.items()},
        tradable=replace_last(history.tradable, visible),
        limit_up_locked=replace_last(history.limit_up_locked, False),
        limit_down_locked=replace_last(history.limit_down_locked, False),
        fields=MappingProxyType(safe_fields),
        _valid_bars=None,
    )


def build_tail_execution(repo, market, strategy, params, pipeline_config, start, stop,
                         *, entry_allowed=None, evidence=None, cancel_event=None):
    """Keep historical daily bars intact and use one partial row per decision."""
    from app.backtest.engine import TailSessionFill

    if evidence is not None:
        from app.backtest.evidence import fingerprint_matrix

        evidence.datasets["daily_matrix"] = fingerprint_matrix(market)
    daily = slice_market_data_matrix(market, start, stop)
    prior_raw = valid_shift(market.fields["raw_close"], 1, bar_index=market.valid_bars)
    with np.errstate(invalid="ignore", divide="ignore"):
        prior_factor = valid_shift(market.close / market.fields["raw_close"], 1, bar_index=market.valid_bars)
    entry = np.zeros(daily.shape, dtype=np.uint8)
    exit_ = np.zeros_like(entry)
    scores = np.zeros(daily.shape, dtype=np.float32)
    entry_codes = np.full(daily.shape, -1, dtype=np.int16)
    exit_codes = np.full_like(entry_codes, -1)
    floors = np.full(daily.shape, np.nan, dtype=np.float32)
    prices = np.full_like(floors, np.nan)
    signal_close = np.full_like(floors, np.nan)
    buy_blocked = np.ones(daily.shape, dtype=bool)
    sell_blocked = np.ones_like(buy_blocked)
    coverage = []
    entry_ids = exit_ids = ()
    for j, row in enumerate(range(start, stop)):
        if cancel_event is not None and cancel_event.is_set():
            raise ValueError("cancelled")
        day = date.fromisoformat(market.timestamp_labels[row][:10])
        frame = repo.get_minute_by_dates(list(market.symbols), [day], adjusted=False)
        partial, raw_close, raw_fill, factor, confirmed = _snapshot_row(market, row, frame, day, repo.store.data_dir)
        history = _with_partial_bar(market, row, partial, raw_close)
        signals = MatrixStrategyPipeline().run(strategy, history, params, pipeline_config)
        if signals.entry_close_floor is None:
            raise ValueError("尾盘回放需要策略提供买点绑定防守线")
        entry[j] = signals.entry[-1] if entry_allowed is None or entry_allowed[row] else 0
        exit_[j] = signals.exit[-1]
        scores[j] = signals.score[-1]
        entry_codes[j] = signals.entry_signal_code[-1]
        exit_codes[j] = signals.exit_signal_code[-1]
        floors[j] = signals.entry_close_floor[-1]
        entry_ids, exit_ids = signals.entry_signal_ids, signals.exit_signal_ids
        signal_close[j] = partial["close"]
        pct = market.fields["price_limit_pct"][row]
        # Existing opening-limit convention conservatively excludes unknown
        # reference prices and factor-change sessions on both sides.
        basis = prior_raw[row]
        valid_reference = np.isfinite(basis) & np.isfinite(pct) & np.isfinite(prior_factor[row])
        valid_reference &= np.isclose(factor, prior_factor[row], rtol=2e-5, atol=1e-7)
        prices[j] = np.where(valid_reference, raw_fill * factor, np.nan)
        for up, blocked in ((True, buy_blocked), (False, sell_blocked)):
            limit = numpy_limit_price(
                np.where(valid_reference, basis, 1.0), np.where(valid_reference, pct, 0), up=up,
            )
            locked = raw_fill >= limit - 0.0049 if up else raw_fill <= limit + 0.0049
            blocked[j] = valid_reference & locked
        coverage.append({"date": str(day), "verified_symbols": confirmed})
        if evidence is not None:
            evidence.record_frame(f"tail_minutes_{day}", frame)
    directions = make_signal_matrix(
        daily.shape, entry=entry, exit=exit_, score=scores, entry_signal_code=entry_codes,
        exit_signal_code=exit_codes, entry_signal_ids=entry_ids, exit_signal_ids=exit_ids,
    )
    # The paired mode has its own execution-minute restrictions. The legacy
    # matrix builder's whole-day opening checks do not apply to this schedule.
    execution_daily = replace(daily, fields=MappingProxyType({
        k: v for k, v in daily.fields.items() if k != "price_limit_pct"
    }))
    matrix = build_market_matrix_from_signals(execution_daily, directions)
    matrix = replace(matrix, entry_close_floor=floors, limit_up_locked=buy_blocked,
                     limit_down_locked=sell_blocked)
    return matrix, TailSessionFill(prices, signal_close), coverage

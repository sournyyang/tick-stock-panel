"""Sequoia-X HighTightFlag: strong advance followed by tight, low-volume action."""

import numpy as np

from app.backtest.matrix import (
    MarketDataMatrix,
    SignalMatrix,
    make_signal_matrix,
    valid_rolling_max,
    valid_rolling_mean,
    valid_rolling_min,
    valid_shift,
)

META = {
    "id": "sequoia_high_tight_flag",
    "name": "Sequoia 高窄旗形",
    "description": (
        "40日高低比大于1.6, 近10日高低比小于1.15且维持在40日高点的80%以上, "
        "当日成交量低于此前20日均量的60%。源码只识别整理形态, 不包含突破确认。"
    ),
    "tags": ["Sequoia-X", "形态", "缩量"],
    "asset_types": ["stock"],
    "timeframes": ["1d"],
    "params": [],
    "scoring": {},
    "order_by": None,
    "descending": True,
    "limit": 100,
}

BASIC_FILTER = {
    "price_min": None,
    "price_max": None,
    "market_cap_min": None,
    "float_cap_min": None,
    "float_cap_max": None,
    "amount_min": None,
    "amount_max": None,
    "turnover_min": None,
    "turnover_max": None,
    "exclude_st": False,
    "exclude_new_days": 0,
    "boards": [],
}

EXECUTION_BACKEND = "matrix_native"
ENTRY_SIGNALS = ["signal_sequoia_high_tight_flag"]
EXIT_SIGNALS: list[str] = []
MAX_HOLD_DAYS = 5


class SequoiaHighTightFlagMatrixStrategy:
    def required_fields(self) -> frozenset[str]:
        return frozenset({"high", "low", "close", "volume"})

    def required_warmup_bars(self, params: dict) -> int:
        del params
        return 40

    def compute_signals(self, market: MarketDataMatrix, params: dict) -> SignalMatrix:
        del params
        close_valid = np.isfinite(market.close)
        high_valid = close_valid & np.isfinite(market.high)
        low_valid = close_valid & np.isfinite(market.low)
        volume_valid = close_valid & np.isfinite(market.volume)

        high40 = valid_rolling_max(market.high, high_valid, 40)
        low40 = valid_rolling_min(market.low, low_valid, 40)
        high10 = valid_rolling_max(market.high, high_valid, 10)
        low10 = valid_rolling_min(market.low, low_valid, 10)
        previous_volume = valid_shift(market.volume, 1, volume_valid)
        previous_volume_mean20 = valid_rolling_mean(
            previous_volume,
            np.isfinite(previous_volume),
            20,
        )

        momentum_ratio = np.full(market.shape, np.nan, dtype=np.float32)
        consolidation_ratio = np.full(market.shape, np.nan, dtype=np.float32)
        volume_ratio = np.full(market.shape, np.nan, dtype=np.float32)
        np.divide(high40, low40, out=momentum_ratio, where=np.isfinite(low40) & (low40 > 0))
        np.divide(high10, low10, out=consolidation_ratio, where=np.isfinite(low10) & (low10 > 0))
        np.divide(
            market.volume,
            previous_volume_mean20,
            out=volume_ratio,
            where=np.isfinite(previous_volume_mean20) & (previous_volume_mean20 > 0),
        )

        entry = momentum_ratio > np.float32(1.6)
        entry &= consolidation_ratio < np.float32(1.15)
        entry &= low10 >= high40 * np.float32(0.8)
        entry &= volume_ratio < np.float32(0.6)
        entry &= close_valid & np.isfinite(previous_volume_mean20)

        # The source returns an unsorted list. Rank portfolio contention by the
        # source's strong-momentum ratio without changing the entry population.
        score = np.where(entry, momentum_ratio, np.float32(0.0)).astype(
            np.float32,
            copy=False,
        )
        return make_signal_matrix(
            market.shape,
            entry=entry.astype(np.uint8),
            score=score,
            entry_signal_code=np.where(entry, 0, -1).astype(np.int16),
            entry_signal_ids=("signal_sequoia_high_tight_flag",),
        )


MATRIX_STRATEGY = SequoiaHighTightFlagMatrixStrategy()

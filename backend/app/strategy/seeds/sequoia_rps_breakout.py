"""Sequoia-X RpsBreakout: 120-day relative-strength percentile near the high."""

import numpy as np

from app.backtest.matrix import (
    MarketDataMatrix,
    SignalMatrix,
    make_signal_matrix,
    matrix_feature,
    valid_rolling_max,
)

META = {
    "id": "sequoia_rps_breakout",
    "name": "Sequoia RPS 强势接近新高",
    "description": (
        "120日收益处于当日全市场前10%, 且收盘价不低于含当日在内的120日最高价90%。"
        "这是源码实际规则, 并非严格的新高突破。"
    ),
    "tags": ["Sequoia-X", "RPS", "相对强度"],
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
ENTRY_SIGNALS = ["signal_sequoia_rps_breakout"]
EXIT_SIGNALS: list[str] = []
MAX_HOLD_DAYS = 5


def _percentile_rank_rows(values: np.ndarray) -> np.ndarray:
    """Match pandas rank(pct=True) with average ranks for finite values in each row."""
    result = np.zeros(values.shape, dtype=np.float32)
    for time_id in range(values.shape[0]):
        asset_ids = np.flatnonzero(np.isfinite(values[time_id]))
        if asset_ids.size == 0:
            continue
        row_values = values[time_id, asset_ids]
        order = np.argsort(row_values, kind="stable")
        sorted_values = row_values[order]
        start = 0
        while start < order.size:
            stop = start + 1
            while stop < order.size and sorted_values[stop] == sorted_values[start]:
                stop += 1
            average_rank = ((start + 1) + stop) / 2.0
            result[time_id, asset_ids[order[start:stop]]] = np.float32(
                average_rank / order.size * 100.0
            )
            start = stop
    return result


class SequoiaRpsBreakoutMatrixStrategy:
    def required_fields(self) -> frozenset[str]:
        return frozenset({"high", "close"})

    def required_warmup_bars(self, params: dict) -> int:
        del params
        return 121

    def compute_signals(self, market: MarketDataMatrix, params: dict) -> SignalMatrix:
        del params
        close_valid = np.isfinite(market.close)
        momentum120 = matrix_feature(market, "momentum_120d")
        rps = _percentile_rank_rows(momentum120)
        rolling_high120 = valid_rolling_max(
            market.high,
            close_valid & np.isfinite(market.high),
            120,
        )

        entry = rps >= np.float32(90.0)
        entry &= market.close >= rolling_high120 * np.float32(0.90)
        entry &= close_valid & np.isfinite(rolling_high120)

        score = np.where(entry, rps, np.float32(0.0)).astype(np.float32, copy=False)
        return make_signal_matrix(
            market.shape,
            entry=entry.astype(np.uint8),
            score=score,
            entry_signal_code=np.where(entry, 0, -1).astype(np.int16),
            entry_signal_ids=("signal_sequoia_rps_breakout",),
        )


MATRIX_STRATEGY = SequoiaRpsBreakoutMatrixStrategy()

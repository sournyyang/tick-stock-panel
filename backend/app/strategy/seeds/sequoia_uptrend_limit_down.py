"""Sequoia-X UptrendLimitDown with exchange-aware limit-down detection."""

import numpy as np

from app.backtest.matrix import (
    MarketDataMatrix,
    SignalMatrix,
    make_signal_matrix,
    matrix_feature,
    valid_rolling_mean,
    valid_shift,
)

META = {
    "id": "sequoia_uptrend_limit_down",
    "name": "Sequoia 上升趋势跌停",
    "description": (
        "昨日 MA20 高于 MA60, 今日收盘跌停且成交量超过含当日在内的20日均量2倍。"
        "源码没有反包确认; 跌停判定改用逐日交易所价格限制。"
    ),
    "tags": ["Sequoia-X", "跌停", "趋势"],
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
ENTRY_SIGNALS = ["signal_sequoia_uptrend_limit_down"]
EXIT_SIGNALS: list[str] = []
MAX_HOLD_DAYS = 5


class SequoiaUptrendLimitDownMatrixStrategy:
    def required_fields(self) -> frozenset[str]:
        return frozenset({"close", "volume"})

    def required_warmup_bars(self, params: dict) -> int:
        del params
        return 61

    def compute_signals(self, market: MarketDataMatrix, params: dict) -> SignalMatrix:
        del params
        close_valid = np.isfinite(market.close)
        volume_valid = close_valid & np.isfinite(market.volume)
        previous_ma20 = valid_shift(
            matrix_feature(market, "ma20"),
            1,
            close_valid,
        )
        previous_ma60 = valid_shift(
            matrix_feature(market, "ma60"),
            1,
            close_valid,
        )
        volume_ma20 = valid_rolling_mean(market.volume, volume_valid, 20)
        volume_ratio = np.full(market.shape, np.nan, dtype=np.float32)
        np.divide(
            market.volume,
            volume_ma20,
            out=volume_ratio,
            where=np.isfinite(volume_ma20) & (volume_ma20 > 0),
        )

        entry = previous_ma20 > previous_ma60
        entry &= market.limit_down_locked != 0
        entry &= volume_ratio > np.float32(2.0)
        entry &= close_valid & np.isfinite(previous_ma60)

        score = np.where(entry, volume_ratio, np.float32(0.0)).astype(
            np.float32,
            copy=False,
        )
        return make_signal_matrix(
            market.shape,
            entry=entry.astype(np.uint8),
            score=score,
            entry_signal_code=np.where(entry, 0, -1).astype(np.int16),
            entry_signal_ids=("signal_sequoia_uptrend_limit_down",),
        )


MATRIX_STRATEGY = SequoiaUptrendLimitDownMatrixStrategy()

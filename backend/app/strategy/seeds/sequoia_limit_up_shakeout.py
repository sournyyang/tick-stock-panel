"""Sequoia-X LimitUpShakeout with exchange-aware prior-day limit detection."""

import numpy as np

from app.backtest.matrix import (
    MarketDataMatrix,
    SignalMatrix,
    make_signal_matrix,
    valid_shift,
)

META = {
    "id": "sequoia_limit_up_shakeout",
    "name": "Sequoia 涨停洗盘",
    "description": (
        "昨日收盘涨停, 今日放量超过昨日2倍并收阴, 且最低价不低于昨日收盘。"
        "涨停判定改用逐日交易所价格限制, 避免源码固定9.5%阈值误判。"
    ),
    "tags": ["Sequoia-X", "涨停", "洗盘"],
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
ENTRY_SIGNALS = ["signal_sequoia_limit_up_shakeout"]
EXIT_SIGNALS: list[str] = []
MAX_HOLD_DAYS = 5


class SequoiaLimitUpShakeoutMatrixStrategy:
    def required_fields(self) -> frozenset[str]:
        return frozenset({"open", "low", "close", "volume"})

    def required_warmup_bars(self, params: dict) -> int:
        del params
        return 3

    def compute_signals(self, market: MarketDataMatrix, params: dict) -> SignalMatrix:
        del params
        close_valid = np.isfinite(market.close)
        previous_close = valid_shift(market.close, 1, close_valid)
        previous_volume = valid_shift(market.volume, 1, close_valid & np.isfinite(market.volume))
        previous_limit_up = valid_shift(
            market.limit_up_locked.astype(np.float32),
            1,
            close_valid,
        )

        volume_ratio = np.full(market.shape, np.nan, dtype=np.float32)
        np.divide(
            market.volume,
            previous_volume,
            out=volume_ratio,
            where=np.isfinite(previous_volume) & (previous_volume > 0),
        )
        entry = previous_limit_up == np.float32(1.0)
        entry &= market.close < market.open
        entry &= volume_ratio > np.float32(2.0)
        entry &= market.low >= previous_close
        entry &= close_valid & np.isfinite(previous_close)

        score = np.where(entry, volume_ratio, np.float32(0.0)).astype(
            np.float32,
            copy=False,
        )
        return make_signal_matrix(
            market.shape,
            entry=entry.astype(np.uint8),
            score=score,
            entry_signal_code=np.where(entry, 0, -1).astype(np.int16),
            entry_signal_ids=("signal_sequoia_limit_up_shakeout",),
        )


MATRIX_STRATEGY = SequoiaLimitUpShakeoutMatrixStrategy()

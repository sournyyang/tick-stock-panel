"""Sequoia-X TurtleTrade: 20-day high breakout with liquidity and bullish filters."""

import numpy as np

from app.backtest.matrix import (
    MarketDataMatrix,
    SignalMatrix,
    make_signal_matrix,
    matrix_feature,
    valid_rolling_max,
    valid_shift,
)

META = {
    "id": "sequoia_turtle_trade",
    "name": "Sequoia 海龟突破",
    "description": (
        "收盘突破此前20个交易日最高价, 成交额过亿, 且当日收阳并高于昨收; "
        "按当日涨幅排序。规则移植自 Sequoia-X TurtleTrade。"
    ),
    "tags": ["Sequoia-X", "突破", "海龟"],
    "asset_types": ["stock"],
    "timeframes": ["1d"],
    "params": [],
    "scoring": {},
    "order_by": None,
    "descending": True,
    "limit": 100,
}

# Sequoia-X scans every locally stored stock and only applies the amount gate below.
# Clear TickFlow's default stock filters so the migrated entry population stays comparable.
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
ENTRY_SIGNALS = ["signal_sequoia_turtle_trade"]
EXIT_SIGNALS: list[str] = []
MAX_HOLD_DAYS = 5


class SequoiaTurtleTradeMatrixStrategy:
    def required_fields(self) -> frozenset[str]:
        return frozenset({"open", "high", "close", "amount"})

    def required_warmup_bars(self, params: dict) -> int:
        del params
        return 21

    def compute_signals(self, market: MarketDataMatrix, params: dict) -> SignalMatrix:
        del params
        close_valid = np.isfinite(market.close)
        previous_high = valid_shift(market.high, 1, close_valid)
        prior_high_20 = valid_rolling_max(
            previous_high,
            np.isfinite(previous_high),
            20,
        )
        previous_close = valid_shift(market.close, 1, close_valid)
        amount = market.field("amount")

        entry = market.close > prior_high_20
        entry &= amount > np.float32(100_000_000.0)
        entry &= market.close > market.open
        entry &= market.close > previous_close
        entry &= close_valid & np.isfinite(amount)

        # The README promises gain ordering. The source implementation instead queries
        # today's float cap, which cannot be reproduced causally for historical dates.
        score = matrix_feature(market, "change_pct") * np.float32(100.0)
        score = np.where(entry, score, np.float32(0.0)).astype(np.float32, copy=False)
        return make_signal_matrix(
            market.shape,
            entry=entry.astype(np.uint8),
            score=score,
            entry_signal_code=np.where(entry, 0, -1).astype(np.int16),
            entry_signal_ids=("signal_sequoia_turtle_trade",),
        )


MATRIX_STRATEGY = SequoiaTurtleTradeMatrixStrategy()

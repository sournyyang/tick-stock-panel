"""Sequoia-X MaVolume: MA5/MA20 golden cross confirmed by 20-day volume."""

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
    "id": "sequoia_ma_volume",
    "name": "Sequoia 均线放量",
    "description": (
        "MA5 由下向上穿越 MA20, 且当日成交量大于含当日在内的20日均量1.5倍。"
        "规则移植自 Sequoia-X MaVolume。"
    ),
    "tags": ["Sequoia-X", "均线", "放量"],
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
ENTRY_SIGNALS = ["signal_sequoia_ma_volume"]
EXIT_SIGNALS: list[str] = []
MAX_HOLD_DAYS = 5


class SequoiaMaVolumeMatrixStrategy:
    def required_fields(self) -> frozenset[str]:
        return frozenset({"close", "volume"})

    def required_warmup_bars(self, params: dict) -> int:
        del params
        return 21

    def compute_signals(self, market: MarketDataMatrix, params: dict) -> SignalMatrix:
        del params
        close_valid = np.isfinite(market.close)
        volume_valid = close_valid & np.isfinite(market.volume)
        ma5 = matrix_feature(market, "ma5")
        ma20 = matrix_feature(market, "ma20")
        previous_ma5 = valid_shift(ma5, 1, np.isfinite(ma5))
        previous_ma20 = valid_shift(ma20, 1, np.isfinite(ma20))
        volume_ma20 = valid_rolling_mean(market.volume, volume_valid, 20)

        entry = previous_ma5 < previous_ma20
        entry &= ma5 > ma20
        entry &= market.volume > volume_ma20 * np.float32(1.5)
        entry &= close_valid & np.isfinite(volume_ma20)

        # Sequoia-X does not rank MaVolume hits. Volume expansion is the closest
        # causal signal-strength measure for TickFlow's ten-position portfolio.
        score = np.full(market.shape, 0.0, dtype=np.float32)
        np.divide(
            market.volume,
            volume_ma20,
            out=score,
            where=entry & np.isfinite(volume_ma20) & (volume_ma20 > 0),
        )
        return make_signal_matrix(
            market.shape,
            entry=entry.astype(np.uint8),
            score=score,
            entry_signal_code=np.where(entry, 0, -1).astype(np.int16),
            entry_signal_ids=("signal_sequoia_ma_volume",),
        )


MATRIX_STRATEGY = SequoiaMaVolumeMatrixStrategy()

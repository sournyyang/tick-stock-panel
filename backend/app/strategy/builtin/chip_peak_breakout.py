"""低位筹码峰突破回踩 — 放量突破后等待缩量回踩成本区。"""

import numpy as np

from app.backtest.matrix import (
    MarketDataMatrix,
    SignalMatrix,
    compute_chip_peak_entries,
    compute_market_breadth_mask,
    make_signal_matrix,
    matrix_feature,
)

META = {
    "id": "chip_peak_breakout",
    "name": "低位筹码峰回踩·强势环境(探索V4)",
    "description": "低位筹码峰回踩承接,且全市场至少半数股票站上MA60",
    "tags": ["筹码峰", "缩量回踩", "市场宽度", "探索V4"],
    "version": "4.0.0",
    "asset_types": ["stock"],
    "timeframes": ["1d"],
    "params": [
        {
            "id": "min_history_bars",
            "label": "最少筹码预热交易日",
            "type": "int",
            "default": 120,
            "min": 60,
            "max": 250,
            "step": 10,
        },
        {
            "id": "position_window",
            "label": "峰位观察窗口",
            "type": "int",
            "default": 120,
            "min": 60,
            "max": 250,
            "step": 10,
        },
        {
            "id": "volume_ratio_min",
            "label": "突破日最低量比",
            "type": "float",
            "default": 1.5,
            "min": 1.0,
            "max": 3.0,
            "step": 0.1,
        },
        {
            "id": "concentration_max_pct",
            "label": "70%成本区最大宽度%",
            "type": "float",
            "default": 20.0,
            "min": 5.0,
            "max": 35.0,
            "step": 1.0,
        },
        {
            "id": "peak_share_min_pct",
            "label": "主峰附近最少筹码%",
            "type": "float",
            "default": 15.0,
            "min": 5.0,
            "max": 40.0,
            "step": 1.0,
        },
        {
            "id": "peak_position_max_pct",
            "label": "主峰最高位置%",
            "type": "float",
            "default": 45.0,
            "min": 20.0,
            "max": 70.0,
            "step": 5.0,
        },
        {
            "id": "max_initial_residual_pct",
            "label": "初始未知筹码上限%",
            "type": "float",
            "default": 10.0,
            "min": 1.0,
            "max": 30.0,
            "step": 1.0,
        },
    ],
    # 策略内 score 由集中度、主峰占比、峰位和突破量能共同确定。
    "scoring": {},
    "order_by": "score",
    "descending": True,
    "limit": 100,
}

EXECUTION_BACKEND = "matrix_native"
ENTRY_SIGNALS = ["signal_chip_peak_breakout"]
EXIT_SIGNALS = ["signal_chip_peak_fail_ma20"]
STOP_LOSS = -0.08
MAX_HOLD_DAYS = 20


class ChipPeakBreakoutMatrixStrategy:
    def required_fields(self) -> frozenset[str]:
        return frozenset({"high", "low", "close", "volume", "turnover_rate"})

    def required_warmup_bars(self, params: dict) -> int:
        return max(
            int(params.get("min_history_bars", 120)),
            int(params.get("position_window", 120)),
        )

    def compute_signals(self, market: MarketDataMatrix, params: dict) -> SignalMatrix:
        entry, score = compute_chip_peak_entries(
            market.high,
            market.low,
            market.close,
            matrix_feature(market, "turnover_rate"),
            matrix_feature(market, "vol_ratio_5d"),
            min_history_bars=int(params.get("min_history_bars", 120)),
            position_window=int(params.get("position_window", 120)),
            volume_ratio_min=float(params.get("volume_ratio_min", 1.5)),
            concentration_max_pct=float(params.get("concentration_max_pct", 20.0)),
            peak_share_min_pct=float(params.get("peak_share_min_pct", 15.0)),
            peak_position_max_pct=float(params.get("peak_position_max_pct", 45.0)),
            max_initial_residual_pct=float(params.get("max_initial_residual_pct", 10.0)),
            confirmation_bars=1,
            max_breakout_extension_pct=3.0,
            pullback_window_bars=10,
            pullback_touch_buffer_pct=2.0,
            pullback_volume_ratio_max=1.2,
        )
        breadth_mask = compute_market_breadth_mask(
            market.close,
            matrix_feature(market, "ma60"),
            0.5,
        )
        entry = entry & breadth_mask[:, None]
        score = np.where(entry != 0, score, 0.0).astype(np.float32)
        ma20 = matrix_feature(market, "ma20")
        exit_ = np.isfinite(ma20) & (market.close < ma20)
        return make_signal_matrix(
            market.shape,
            entry=entry,
            exit=exit_.astype(np.uint8),
            score=score,
            entry_signal_code=np.where(entry != 0, 0, -1).astype(np.int16),
            exit_signal_code=np.where(exit_, 0, -1).astype(np.int16),
            entry_signal_ids=("signal_chip_peak_breakout",),
            exit_signal_ids=("signal_chip_peak_fail_ma20",),
        )


MATRIX_STRATEGY = ChipPeakBreakoutMatrixStrategy()

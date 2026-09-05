"""试盘线突破: 适度放量上影阳线后, 缩量并以实体阳线突破试盘收盘价。"""

import numpy as np

from app.backtest.matrix import (
    MarketDataMatrix,
    SignalMatrix,
    make_signal_matrix,
    valid_shift,
)

META = {
    "id": "trial_line_breakout",
    "name": "试盘线战法",
    "description": "三至六倍量上影阳线后缩量并首次有效突破; 过滤弱试盘、弱突破、阶段低位和短期追高信号",
    "tags": ["试盘线", "量价", "突破"],
    "asset_types": ["stock"],
    "timeframes": ["1d"],
    "version": "1.2.0",
    "default_entry_fill": "close_t",
    "default_exit_fill": "close_t",
    "basic_filter": {
        "price_min": 3,
        "price_max": 300,
        "market_cap_min": 10e8,
        "amount_min": 0.2e8,
        "exclude_st": True,
        "exclude_new_days": 30,
        "boards": ["沪主板", "深主板", "创业板"],
    },
    "params": [
        {
            "id": "trial_volume_multiple",
            "label": "试盘量/前一日量下限",
            "type": "float",
            "default": 3.0,
            "min": 1.0,
            "max": 10.0,
            "step": 0.1,
        },
        {
            "id": "max_trial_volume_multiple",
            "label": "试盘量/前一日量上限",
            "type": "float",
            "default": 6.0,
            "min": 3.0,
            "max": 30.0,
            "step": 0.5,
        },
        {
            "id": "upper_shadow_body_ratio",
            "label": "上影线/实体最小比例",
            "type": "float",
            "default": 0.5,
            "min": 0.0,
            "max": 5.0,
            "step": 0.1,
        },
        {
            "id": "min_trial_close_location_pct",
            "label": "试盘收盘在振幅中的最低位置(%)",
            "type": "float",
            "default": 40.0,
            "min": 0.0,
            "max": 100.0,
            "step": 1.0,
        },
        {
            "id": "min_pretrial_low_distance_pct",
            "label": "试盘前价格高于20日低点(%)",
            "type": "float",
            "default": 3.0,
            "min": 0.0,
            "max": 50.0,
            "step": 1.0,
        },
        {
            "id": "max_pretrial_20d_gain_pct",
            "label": "试盘前20日最大涨幅(%)",
            "type": "float",
            "default": 10.0,
            "min": 0.0,
            "max": 300.0,
            "step": 1.0,
        },
        {
            "id": "entry_window_days",
            "label": "突破等待天数",
            "type": "int",
            "default": 5,
            "min": 1,
            "max": 30,
            "step": 1,
        },
        {
            "id": "recent_limit_window_days",
            "label": "试盘前涨停统计天数",
            "type": "int",
            "default": 10,
            "min": 1,
            "max": 20,
            "step": 1,
        },
        {
            "id": "max_recent_limit_ups",
            "label": "试盘前最多涨停数",
            "type": "int",
            "default": 1,
            "min": 0,
            "max": 5,
            "step": 1,
        },
        {
            "id": "recent_limit_down_window_days",
            "label": "试盘前跌停统计天数",
            "type": "int",
            "default": 5,
            "min": 1,
            "max": 20,
            "step": 1,
        },
        {
            "id": "max_recent_limit_downs",
            "label": "试盘前最多跌停数",
            "type": "int",
            "default": 0,
            "min": 0,
            "max": 5,
            "step": 1,
        },
        {
            "id": "max_post_trial_volume_pct",
            "label": "洗盘均量/试盘量上限(%)",
            "type": "float",
            "default": 50.0,
            "min": 1.0,
            "max": 100.0,
            "step": 1.0,
        },
        {
            "id": "max_post_trial_high_pct",
            "label": "试盘后最高允许涨幅(%)",
            "type": "float",
            "default": 10.0,
            "min": 1.0,
            "max": 50.0,
            "step": 1.0,
        },
        {
            "id": "min_breakout_margin_pct",
            "label": "买入收盘高于试盘收盘(%)",
            "type": "float",
            "default": 0.5,
            "min": 0.0,
            "max": 20.0,
            "step": 0.1,
        },
        {
            "id": "exit_volume_multiple",
            "label": "卖出量/试盘量",
            "type": "float",
            "default": 3.0,
            "min": 1.0,
            "max": 10.0,
            "step": 0.1,
        },
        {
            "id": "take_profit_pct",
            "label": "收盘止盈(%)",
            "type": "float",
            "default": 15.0,
            "min": 1.0,
            "max": 100.0,
            "step": 1.0,
        },
        {
            "id": "close_stop_loss_pct",
            "label": "收盘止损(%)",
            "type": "float",
            "default": 6.0,
            "min": 1.0,
            "max": 50.0,
            "step": 1.0,
        },
    ],
    "scoring": {},
    "order_by": "score",
    "descending": True,
    "limit": 100,
}

EXECUTION_BACKEND = "matrix_native"
ENTRY_SIGNALS = ["signal_trial_close_breakout"]
EXIT_SIGNALS = [
    "signal_trial_volume_bearish",
    "signal_profit_target",
    "signal_close_stop_loss",
]
# 本策略在信号矩阵中按收盘价确认止损, 避免通用成交引擎按盘中最低价提前退出。
STOP_LOSS = None
MAX_HOLD_DAYS = 5
ALERTS = []


class TrialLineBreakoutMatrixStrategy:
    def required_fields(self) -> frozenset[str]:
        return frozenset({"open", "high", "close", "volume"})

    def required_warmup_bars(self, params: dict) -> int:
        return (
            max(
                int(params.get("recent_limit_window_days", 10)),
                int(params.get("recent_limit_down_window_days", 5)),
                20,
                1,
            )
            + 1
        )

    def compute_signals(self, market: MarketDataMatrix, params: dict) -> SignalMatrix:
        trial_volume_multiple = max(float(params.get("trial_volume_multiple", 3.0)), 0.0)
        max_trial_volume_multiple = max(
            float(params.get("max_trial_volume_multiple", 6.0)),
            trial_volume_multiple,
        )
        upper_shadow_body_ratio = max(float(params.get("upper_shadow_body_ratio", 0.5)), 0.0)
        min_trial_close_location_ratio = (
            max(float(params.get("min_trial_close_location_pct", 40.0)), 0.0) / 100.0
        )
        min_pretrial_low_distance_ratio = (
            max(float(params.get("min_pretrial_low_distance_pct", 3.0)), 0.0) / 100.0
        )
        max_pretrial_20d_gain_ratio = (
            max(float(params.get("max_pretrial_20d_gain_pct", 10.0)), 0.0) / 100.0
        )
        entry_window_days = max(int(params.get("entry_window_days", 5)), 1)
        recent_limit_window_days = max(int(params.get("recent_limit_window_days", 10)), 1)
        max_recent_limit_ups = max(int(params.get("max_recent_limit_ups", 1)), 0)
        recent_limit_down_window_days = max(
            int(params.get("recent_limit_down_window_days", 5)), 1
        )
        max_recent_limit_downs = max(int(params.get("max_recent_limit_downs", 0)), 0)
        max_post_trial_volume_ratio = (
            max(float(params.get("max_post_trial_volume_pct", 50.0)), 0.0) / 100.0
        )
        max_post_trial_high_ratio = (
            max(float(params.get("max_post_trial_high_pct", 10.0)), 0.0) / 100.0
        )
        min_breakout_margin_ratio = (
            max(float(params.get("min_breakout_margin_pct", 0.5)), 0.0) / 100.0
        )
        exit_volume_multiple = max(float(params.get("exit_volume_multiple", 3.0)), 0.0)
        take_profit_ratio = max(float(params.get("take_profit_pct", 15.0)), 0.0) / 100.0
        close_stop_loss_ratio = (
            max(float(params.get("close_stop_loss_pct", 6.0)), 0.0) / 100.0
        )

        previous_volume = valid_shift(
            market.volume,
            1,
            bar_index=market.valid_bars,
        )
        previous_close = valid_shift(
            market.close,
            1,
            bar_index=market.valid_bars,
        )
        bullish = market.close > market.open
        body = market.close - market.open
        upper_shadow = market.high - market.close
        candle_range = market.high - market.low
        trial_close_location = np.divide(
            market.close - market.low,
            candle_range,
            out=np.zeros(market.shape, dtype=np.float32),
            where=candle_range > 0,
        )
        valid_close = np.isfinite(market.close)
        limit_up_values = market.limit_up_locked.astype(np.float32)
        limit_down_values = market.limit_down_locked.astype(np.float32)
        recent_limit_up_count = np.zeros(market.shape, dtype=np.float32)
        recent_limit_down_count = np.zeros(market.shape, dtype=np.float32)
        pretrial_20d_low = np.full(market.shape, np.inf, dtype=np.float32)
        for offset in range(1, recent_limit_window_days + 1):
            shifted_limit_up = valid_shift(
                limit_up_values,
                offset,
                valid_mask=valid_close,
                bar_index=market.valid_bars,
            )
            recent_limit_up_count += np.nan_to_num(shifted_limit_up, nan=0.0)
        for offset in range(1, recent_limit_down_window_days + 1):
            shifted_limit_down = valid_shift(
                limit_down_values,
                offset,
                valid_mask=valid_close,
                bar_index=market.valid_bars,
            )
            recent_limit_down_count += np.nan_to_num(shifted_limit_down, nan=0.0)
        for offset in range(1, 21):
            shifted_low = valid_shift(
                market.low,
                offset,
                bar_index=market.valid_bars,
            )
            np.minimum(
                pretrial_20d_low,
                np.where(np.isfinite(shifted_low), shifted_low, np.inf),
                out=pretrial_20d_low,
            )
        pretrial_20d_close = valid_shift(
            market.close,
            20,
            bar_index=market.valid_bars,
        )
        enough_above_recent_low = (
            ~np.isfinite(pretrial_20d_close)
            | (min_pretrial_low_distance_ratio <= 0)
            | (
                np.isfinite(pretrial_20d_low)
                & (pretrial_20d_low > 0)
                & (
                    previous_close
                    >= pretrial_20d_low * (1.0 + min_pretrial_low_distance_ratio)
                )
            )
        )
        not_pretrial_overextended = (
            ~np.isfinite(pretrial_20d_close)
            | (pretrial_20d_close <= 0)
            | (
                previous_close
                <= pretrial_20d_close * (1.0 + max_pretrial_20d_gain_ratio)
            )
        )

        trial = (
            bullish
            & np.isfinite(previous_close)
            & (market.close > previous_close)
            & np.isfinite(previous_volume)
            & (previous_volume > 0)
            & (market.volume >= previous_volume * trial_volume_multiple)
            & (market.volume <= previous_volume * max_trial_volume_multiple)
            & (upper_shadow > 0)
            & (upper_shadow >= body * upper_shadow_body_ratio)
            & (trial_close_location >= min_trial_close_location_ratio)
            & enough_above_recent_low
            & not_pretrial_overextended
            & (recent_limit_up_count <= max_recent_limit_ups)
            & (recent_limit_down_count <= max_recent_limit_downs)
        )

        entry = np.zeros(market.shape, dtype=np.uint8)
        exit_ = np.zeros(market.shape, dtype=np.uint8)
        score = np.zeros(market.shape, dtype=np.float32)
        entry_code = np.full(market.shape, -1, dtype=np.int16)
        exit_code = np.full(market.shape, -1, dtype=np.int16)

        asset_count = market.shape[1]
        waiting = np.zeros(asset_count, dtype=bool)
        waiting_age = np.zeros(asset_count, dtype=np.int16)
        trial_close = np.full(asset_count, np.nan, dtype=np.float32)
        trial_volume = np.full(asset_count, np.nan, dtype=np.float32)
        trial_score = np.zeros(asset_count, dtype=np.float32)
        pullback_volume_sum = np.zeros(asset_count, dtype=np.float64)
        pullback_bar_count = np.zeros(asset_count, dtype=np.int16)
        holding = np.zeros(asset_count, dtype=bool)
        entry_price = np.full(asset_count, np.nan, dtype=np.float32)
        benchmark_volume = np.full(asset_count, np.nan, dtype=np.float32)
        holding_age = np.zeros(asset_count, dtype=np.int16)

        for time_id in range(market.shape[0]):
            valid = (
                np.isfinite(market.open[time_id])
                & np.isfinite(market.high[time_id])
                & np.isfinite(market.close[time_id])
                & np.isfinite(market.volume[time_id])
            )

            active = holding & valid
            holding_age[active] += 1
            profit_exit = active & (
                market.close[time_id] / entry_price - 1.0
                >= take_profit_ratio - 1e-6
            )
            volume_exit = (
                active
                & (market.close[time_id] < market.open[time_id])
                & (market.volume[time_id] >= benchmark_volume * exit_volume_multiple)
            )
            close_stop_exit = active & (
                market.close[time_id]
                <= entry_price * (1.0 - close_stop_loss_ratio) + 1e-6
            )
            max_hold_exit = active & (holding_age >= MAX_HOLD_DAYS)
            signal_exit = profit_exit | volume_exit | close_stop_exit
            should_reset = signal_exit | max_hold_exit
            if np.any(signal_exit):
                exit_[time_id, signal_exit] = 1
                exit_code[time_id, volume_exit] = 0
                exit_code[time_id, profit_exit] = 1
                exit_code[time_id, close_stop_exit] = 2
            if np.any(should_reset):
                holding[should_reset] = False
                holding_age[should_reset] = 0
                entry_price[should_reset] = np.nan
                benchmark_volume[should_reset] = np.nan

            can_scan = valid & ~holding & ~should_reset
            aging = waiting & can_scan
            waiting_age[aging] += 1
            expired = waiting & (waiting_age > entry_window_days)
            waiting[expired] = False
            pullback_volume_sum[expired] = 0.0
            pullback_bar_count[expired] = 0

            current_trial = trial[time_id] & can_scan
            if np.any(current_trial):
                waiting[current_trial] = True
                waiting_age[current_trial] = 0
                trial_close[current_trial] = market.close[time_id, current_trial]
                trial_volume[current_trial] = market.volume[time_id, current_trial]
                trial_score[current_trial] = (
                    previous_volume[time_id, current_trial]
                    / market.volume[time_id, current_trial]
                ).astype(np.float32)
                pullback_volume_sum[current_trial] = 0.0
                pullback_bar_count[current_trial] = 0

            setup_ready = waiting & (waiting_age >= 1) & (waiting_age <= entry_window_days)
            direct_contraction = (
                setup_ready
                & (waiting_age == 1)
                & (
                    market.volume[time_id]
                    <= trial_volume * max_post_trial_volume_ratio + 1e-6
                )
            )
            delayed_contraction = (
                setup_ready
                & (waiting_age >= 2)
                & (pullback_bar_count > 0)
                & (
                    pullback_volume_sum
                    / np.maximum(pullback_bar_count, 1)
                    <= trial_volume * max_post_trial_volume_ratio + 1e-6
                )
            )
            volume_contracted = direct_contraction | delayed_contraction
            overextended = setup_ready & (
                market.high[time_id]
                > trial_close * (1.0 + max_post_trial_high_ratio) + 1e-6
            )
            first_breakout = setup_ready & (market.close[time_id] > trial_close)
            strong_breakout = setup_ready & (
                market.close[time_id]
                >= trial_close * (1.0 + min_breakout_margin_ratio) - 1e-6
            )
            should_enter = (
                first_breakout
                & strong_breakout
                & (waiting_age >= 1)
                & (waiting_age <= entry_window_days)
                & bullish[time_id]
                & volume_contracted
                & ~market.limit_up_locked[time_id].astype(bool)
                & ~overextended
                & can_scan
            )
            invalid_setup = overextended | (first_breakout & ~should_enter)
            waiting[invalid_setup] = False
            pullback_volume_sum[invalid_setup] = 0.0
            pullback_bar_count[invalid_setup] = 0
            if np.any(should_enter):
                entry[time_id, should_enter] = 1
                entry_code[time_id, should_enter] = 0
                score[time_id, should_enter] = trial_score[should_enter]
                holding[should_enter] = True
                holding_age[should_enter] = 0
                entry_price[should_enter] = market.close[time_id, should_enter]
                benchmark_volume[should_enter] = trial_volume[should_enter]
                waiting[should_enter] = False
                pullback_volume_sum[should_enter] = 0.0
                pullback_bar_count[should_enter] = 0

            record_pullback = waiting & (waiting_age >= 1) & valid
            pullback_volume_sum[record_pullback] += market.volume[
                time_id, record_pullback
            ]
            pullback_bar_count[record_pullback] += 1

        return make_signal_matrix(
            market.shape,
            entry=entry,
            exit=exit_,
            score=score,
            entry_signal_code=entry_code,
            exit_signal_code=exit_code,
            entry_signal_ids=("signal_trial_close_breakout",),
            exit_signal_ids=(
                "signal_trial_volume_bearish",
                "signal_profit_target",
                "signal_close_stop_loss",
            ),
        )


MATRIX_STRATEGY = TrialLineBreakoutMatrixStrategy()

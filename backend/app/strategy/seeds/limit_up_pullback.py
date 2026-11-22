"""涨停后倍量观察、缩量双阴、止跌阳线确认的日线研究策略。"""

import numpy as np

from app.backtest.matrix import (
    MarketDataMatrix,
    SignalMatrix,
    make_signal_matrix,
    numpy_limit_price,
    rolling_mean,
    shift,
    valid_shift,
)

META = {
    "id": "limit_up_pullback",
    "name": "涨停回调买入法",
    "version": "1.1.0",
    "description": (
        "关键词量化研究版: 涨停当日或次日倍量观察, 随后连续两根缩量下跌阴线, "
        "默认再收上涨且低点不下移的阳线, 也可选择收复昨收、放量确认或突破昨高。"
        "收盘确认, 默认次日开盘买入, "
        "买入后再持有5个交易日收盘退出, 不预设止盈止损。"
        "默认沪深主板非ST, 锚点至少有60根有效历史K线; 排除一字涨停和除权日锚点。"
        "尚无稳定盈利证据, 费用和成交约束使用回测页面设置。"
    ),
    "tags": ["涨停", "回调", "缩量", "研究"],
    "asset_types": ["stock"],
    "timeframes": ["1d"],
    "default_entry_fill": "open_t+1",
    "default_exit_fill": "close_t",
    "params": [
        {
            "id": "volume_stage",
            "label": "倍量观察日 (A当日/B次日)",
            "type": "select",
            "default": "涨停当日",
            "options": ["涨停当日", "涨停次日"],
        },
        {
            "id": "volume_reference",
            "label": "倍量比较基准",
            "type": "select",
            "default": "前一日",
            "options": ["前一日", "前5日均量"],
        },
        {
            "id": "confirmation",
            "label": "阳线确认方式",
            "type": "select",
            "default": "低点不下移",
            "options": ["低点不下移", "收复昨收", "放量收复昨收", "收盘突破昨高"],
        },
        {
            "id": "use_volume_filter",
            "label": "启用倍量要求 (关闭可做对照)",
            "type": "bool",
            "default": True,
        },
        {
            "id": "volume_multiple",
            "label": "观察日成交量/前一日成交量下限",
            "type": "float",
            "default": 3.0,
            "min": 1.0,
            "max": 10.0,
            "step": 0.1,
        },
    ],
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
    "exclude_st": True,
    "exclude_new_days": 0,
    "boards": ["沪主板", "深主板"],
}

EXECUTION_BACKEND = "matrix_native"
ENTRY_SIGNALS = ["signal_limit_up_pullback"]
EXIT_SIGNALS: list[str] = []
MAX_HOLD_DAYS = 5


class LimitUpPullbackMatrixStrategy:
    def required_fields(self) -> frozenset[str]:
        return frozenset({
            "open", "high", "low", "close", "volume",
            "raw_close", "raw_high", "raw_low", "price_limit_pct",
        })

    def required_warmup_bars(self, params: dict) -> int:
        del params
        return 64

    def compute_signals(self, market: MarketDataMatrix, params: dict) -> SignalMatrix:
        stage = params.get("volume_stage", "涨停当日")
        if stage not in ("涨停当日", "涨停次日"):
            raise ValueError("倍量观察日必须为涨停当日或涨停次日")
        confirmation = params.get("confirmation", "低点不下移")
        if confirmation not in ("低点不下移", "收复昨收", "放量收复昨收", "收盘突破昨高"):
            raise ValueError("不支持的阳线确认方式")
        volume_reference = params.get("volume_reference", "前一日")
        if volume_reference not in ("前一日", "前5日均量"):
            raise ValueError("不支持的倍量比较基准")
        multiple = float(params.get("volume_multiple", 3.0))
        if not np.isfinite(multiple) or not 1.0 <= multiple <= 10.0:
            raise ValueError("放量倍数必须为1至10之间的有限数值")

        raw_high = market.field("raw_high")
        raw_low = market.field("raw_low")
        raw_close = market.field("raw_close")
        limit_pct = market.field("price_limit_pct")
        close = market.close
        volume = market.volume
        valid = (
            np.isfinite(market.open) & np.isfinite(market.high)
            & np.isfinite(market.low) & np.isfinite(close) & np.isfinite(volume)
            & (market.open > 0) & (market.low > 0) & (close > 0) & (volume > 0)
        )

        # 涨停限价使用原始价; 缺少权威除权参考价时排除因子变化日, 避免误判。
        observed = np.isfinite(close)
        previous_raw = valid_shift(raw_close, 1, observed).astype(np.float64)
        expected_limit = numpy_limit_price(
            previous_raw, np.where(np.isfinite(limit_pct), limit_pct, 0.0), up=True,
        )
        factor = np.full(market.shape, np.nan, dtype=np.float64)
        np.divide(close, raw_close, out=factor, where=np.isfinite(raw_close) & (raw_close > 0))
        previous_factor = valid_shift(factor, 1, observed)
        anchor = valid & np.isfinite(raw_high) & np.isfinite(raw_low)
        anchor &= np.isfinite(limit_pct) & (limit_pct > 0)
        anchor &= np.abs(raw_close - expected_limit) <= 0.00501
        anchor &= np.abs(factor - previous_factor) <= 1e-6
        anchor &= raw_high - raw_low > 0.005
        anchor &= np.cumsum(valid, axis=0) >= 60

        previous_close = shift(close)
        bearish = valid & (close < market.open) & (close < previous_close)
        bullish = valid & (close > market.open) & (close > previous_close)
        if confirmation == "低点不下移":
            bullish &= market.low >= shift(market.low)
        elif confirmation == "放量收复昨收":
            bullish &= volume > shift(volume)
        elif confirmation == "收盘突破昨高":
            bullish &= close > shift(market.high)
        shrinking = valid & (volume < shift(volume))

        anchor_lag = 3 if stage == "涨停当日" else 4
        entry = bullish & (shift(anchor, anchor_lag) == 1)
        entry &= (shift(bearish, 1) == 1) & (shift(bearish, 2) == 1)
        entry &= (shift(shrinking, 1) == 1) & (shift(shrinking, 2) == 1)
        # 形态按相邻市场交易日连接, 不把停牌/缺失K线压缩成连续双阴。
        for offset in range(1, anchor_lag + 1):
            entry &= shift(valid, offset) == 1
        if params.get("use_volume_filter", True):
            previous_volume = shift(volume)
            if volume_reference == "前5日均量":
                previous_volume = rolling_mean(previous_volume, 5)
            expanded = valid & (shift(valid) == 1) & (previous_volume > 0)
            expanded &= volume >= multiple * previous_volume
            # A的D0和B的D1均在确认阳线之前三个市场交易日。
            entry &= shift(expanded, 3) == 1

        return make_signal_matrix(
            market.shape,
            entry=entry.astype(np.uint8),
            entry_signal_code=np.where(entry, 0, -1).astype(np.int16),
            entry_signal_ids=("signal_limit_up_pullback",),
        )


MATRIX_STRATEGY = LimitUpPullbackMatrixStrategy()

from datetime import date, datetime, timedelta
from types import SimpleNamespace

import polars as pl

from app.backtest.engine import BacktestEngine


def test_minute_fill_reads_multiple_date_batches_without_reusing_batch_size():
    days = [date(2026, 1, 1) + timedelta(days=i) for i in range(101)]
    calls = []

    def read(symbols, dates, *, asset_type):
        calls.append(list(dates))
        return pl.DataFrame([
            {"symbol": "600000.SH", "datetime": datetime.combine(day, datetime.min.time()),
             "open": 10., "high": 11., "low": 9., "close": 10.5, "volume": 100., "amount": 1000.}
            for day in dates
        ])

    cache = BacktestEngine._load_minute_for_fills(
        SimpleNamespace(get_minute_by_dates=read), ["600000.SH"],
        {str(day) for day in days}, "stock",
    )
    assert [len(batch) for batch in calls] == [50, 50, 1]
    assert len(cache) == 101
    assert cache[("600000.SH", str(days[-1]))].shape == (1, 6)

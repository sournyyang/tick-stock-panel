from datetime import date, timedelta

import polars as pl
import pytest

from app.backtest.engine import BacktestEngine, MatcherConfig


def run(method, *, exit_fill="close_t", take_profit=None, gap=False, slots=1, many=False):
    symbols = ["A", "B", "C", "D", "E"] if many else ["A", "B"]
    rows, entries, exits = [], [], []
    for symbol in symbols:
        for day in range(5):
            opening, high, close = 10.0, 10.0, 10.0
            if symbol == "A" and day == 2:
                opening = 12.0 if gap else 10.0
                high = 20.0 if many else 12.0
                close = 20.0 if many else 11.0
            rows.append({"symbol": symbol, "date": date(2024, 1, 1)+timedelta(days=day),
                         "open": opening, "high": high, "low": 10.0, "close": close,
                         "volume": 100_000.0})
            entries.append((symbol == "A" and day == 0) or (symbol != "A" and day == 1))
            exits.append(symbol == "A" and day == (1 if exit_fill == "open_t+1" else 2))
    return getattr(BacktestEngine(repo=None), method)(
        pl.DataFrame(rows), pl.Series(entries), pl.Series(exits),
        MatcherConfig(entry_fill="open_t+1", exit_fill=exit_fill, take_profit_pct=take_profit,
                      max_positions=slots, initial_capital=100_000, fees_pct=0, slippage_bps=0))


@pytest.mark.parametrize("method", ["simulate_portfolio", "simulate_portfolio_legacy"])
def test_close_sale_cannot_free_slot_for_same_day_open_purchase(method):
    result = run(method)
    assert [t.symbol for t in result.trades] == ["A"]
    assert result.stats["execution"]["buy_no_slot"] == 1


@pytest.mark.parametrize("method", ["simulate_portfolio", "simulate_portfolio_legacy"])
def test_close_sale_profit_does_not_inflate_opening_position_budget(method):
    result = run(method, slots=5, many=True)
    new_buys = [t for t in result.trades if t.symbol != "A"]
    assert len(new_buys) == 4
    assert sum(t.entry_value for t in new_buys) <= 80_000


@pytest.mark.parametrize("method", ["simulate_portfolio", "simulate_portfolio_legacy"])
def test_open_exit_can_fund_same_open_rebalance(method):
    result = run(method, exit_fill="open_t+1")
    assert {t.symbol for t in result.trades} == {"A", "B"}
    trade = next(t for t in result.trades if t.symbol == "B")
    assert str(trade.entry_date) == "2024-01-03"


@pytest.mark.parametrize("method", ["simulate_portfolio", "simulate_portfolio_legacy"])
@pytest.mark.parametrize("gap", [False, True])
def test_take_profit_releases_opening_budget_only_if_opening_gap_already_hits_target(method, gap):
    result = run(method, take_profit=.15, gap=gap)
    symbols = {t.symbol for t in result.trades}
    assert symbols == ({"A", "B"} if gap else {"A"})

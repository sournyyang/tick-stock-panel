from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from app.backtest.engine import BacktestEngine, EntryCloseFloor, MatcherConfig, SimulationOptions
from app.backtest.matrix import build_market_matrix


def fixture(*, breach_day=2, second_signal=False, symbols=("A",), floor=9.0):
    rows, entries = [], []
    for symbol in symbols:
        for day in range(7):
            opening, high, low, close = 10.0, 10.5, 9.5, 10.0
            if symbol == "A" and day == breach_day:
                high, low, close = 10.5, 8.0, 8.5
            if symbol == "A" and day == breach_day + 1:
                opening, high, low, close = 8.0, 15.0, 8.0, 10.0
            rows.append(
                {
                    "symbol": symbol,
                    "date": date(2024, 1, 2) + timedelta(days=day),
                    "open": opening,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": 1000.0,
                }
            )
            entries.append(
                (symbol == "A" and (day == 0 or (second_signal and day == 1)))
                or (symbol == "B" and day == breach_day)
            )
    matrix = build_market_matrix(pl.DataFrame(rows), pl.Series(entries), None, entry_delay_bars=1)
    levels = np.full(matrix.shape, np.nan)
    levels[0, 0] = floor
    if second_signal:
        levels[1, 0] = 1.0
    if "B" in symbols:
        levels[breach_day, 1] = 7.0
    guard = EntryCloseFloor(levels=levels, open_sell_blocked=np.zeros(matrix.shape, dtype=bool))
    return matrix, guard


def run(matrix, guard, **overrides):
    cfg = MatcherConfig(
        entry_fill="open_t+1",
        exit_fill=overrides.pop("exit_fill", "close_t"),
        take_profit_pct=0.2,
        max_positions=1,
        initial_capital=100000,
        fees_pct=0,
        slippage_bps=0,
        **overrides,
    )
    return BacktestEngine(repo=None).simulate_market_matrix(
        matrix, cfg, entry_close_floor=guard, options=SimulationOptions(include_monte_carlo=False)
    )


def test_breach_exits_next_open_before_later_intraday_take_profit():
    matrix, guard = fixture()
    trade = run(matrix, guard).trades[0]
    assert trade.exit_reason == "structure_break"
    assert str(trade.entry_date) == "2024-01-03"
    assert str(trade.exit_signal_date) == "2024-01-04"
    assert str(trade.exit_date) == "2024-01-05"
    assert trade.exit_price == 8.0


@pytest.mark.parametrize("exit_fill", ["close_t", "open_t+1"])
def test_entry_day_close_breach_is_queued_and_sells_only_on_t_plus_one(exit_fill):
    matrix, guard = fixture(breach_day=1)
    trade = run(matrix, guard, exit_fill=exit_fill).trades[0]
    assert trade.exit_reason == "structure_break"
    assert str(trade.exit_date) == "2024-01-04"
    assert str(trade.exit_signal_date) == str(trade.entry_date)


def test_new_unfilled_signal_does_not_reset_existing_position_floor():
    matrix, guard = fixture(second_signal=True)
    trade = run(matrix, guard).trades[0]
    assert trade.exit_reason == "structure_break"
    assert str(trade.exit_date) == "2024-01-05"


@pytest.mark.parametrize("kind", ["limit_down", "suspended"])
@pytest.mark.parametrize("exit_fill", ["close_t", "open_t+1"])
def test_blocked_open_retries_next_open_and_retains_first_signal_date(kind, exit_fill):
    matrix, guard = fixture()
    if kind == "limit_down":
        blocked = guard.open_sell_blocked.copy()
        blocked[3, 0] = True
        guard = replace(guard, open_sell_blocked=blocked)
    else:
        tradable = matrix.tradable.copy()
        tradable[3, 0] = False
        matrix = replace(matrix, tradable=tradable)
    trade = run(matrix, guard, exit_fill=exit_fill).trades[0]
    assert trade.exit_reason == "structure_break"
    assert str(trade.exit_date) == "2024-01-06"
    assert str(trade.exit_signal_date) == "2024-01-04"
    assert trade.blocked_exit_days == 1


def test_structure_open_sale_can_fund_other_stock_at_same_open():
    matrix, guard = fixture(symbols=("A", "B"))
    result = run(matrix, guard)
    assert [t.symbol for t in result.trades] == ["A", "B"]
    assert str(result.trades[1].entry_date) == "2024-01-05"


def test_equal_close_does_not_trigger_and_default_disabled_keeps_old_result():
    matrix, guard = fixture(floor=8.5)
    assert run(matrix, guard).trades[0].exit_reason == "take_profit"
    assert run(matrix, None).trades[0].exit_reason == "take_profit"


def test_future_floor_values_cannot_affect_position():
    matrix, guard = fixture()
    changed = guard.levels.copy()
    changed[1:] = 1000
    assert run(matrix, guard).trades == run(matrix, replace(guard, levels=changed)).trades


def test_intraday_target_on_breach_day_precedes_closing_confirmation():
    matrix, guard = fixture()
    high = matrix.high.copy()
    high[2, 0] = 12.5
    trade = run(replace(matrix, high=high), guard).trades[0]
    assert trade.exit_reason == "take_profit"
    assert str(trade.exit_date) == "2024-01-04"


def test_pending_floor_exit_at_end_is_marked_without_a_fictitious_close_sale():
    matrix, guard = fixture()
    blocked = guard.open_sell_blocked.copy()
    blocked[3:, 0] = True
    result = run(matrix, replace(guard, open_sell_blocked=blocked))
    assert result.trades == []
    assert result.stats["pending_exit_positions"] == 1
    assert result.equity_curve[-1]["positions"] == 1


def test_intraday_low_alone_does_not_confirm_a_close_breach():
    matrix, guard = fixture()
    close = matrix.close.copy()
    close[2, 0] = 10.0
    trade = run(replace(matrix, close=close), guard).trades[0]
    assert trade.exit_reason == "take_profit"


def test_untradable_breach_row_cannot_generate_a_close_signal():
    matrix, guard = fixture()
    tradable = matrix.tradable.copy()
    tradable[2, 0] = False
    trade = run(replace(matrix, tradable=tradable), guard).trades[0]
    assert trade.exit_reason == "take_profit"


def test_invalid_shapes_missing_floor_and_unsupported_mode_fail_closed():
    matrix, guard = fixture()
    for bad in [
        replace(guard, levels=np.ones((1, 1))),
        replace(guard, levels=np.full(matrix.shape, np.nan)),
        replace(guard, open_sell_blocked=np.ones(matrix.shape)),
    ]:
        with pytest.raises(ValueError):
            run(matrix, bad)
    cfg = MatcherConfig(entry_fill="close_t", exit_fill="close_t")
    with pytest.raises(ValueError):
        BacktestEngine(repo=None).simulate_market_matrix(matrix, cfg, entry_close_floor=guard)


def test_close_floor_with_next_open_regular_exits_keeps_bound_floor_and_priority():
    matrix, guard = fixture(second_signal=True)
    # A later signal's floor must not replace the original filled position's floor.
    exits = matrix.exit.copy()
    signal_times = matrix.exit_signal_time.copy()
    exits[3, 0] = 1
    signal_times[3, 0] = 2
    matrix = replace(matrix, exit=exits, exit_signal_time=signal_times)
    trade = run(matrix, guard, exit_fill="open_t+1").trades[0]
    assert trade.exit_reason == "structure_break"
    assert str(trade.exit_signal_date) == "2024-01-04"
    assert str(trade.exit_date) == "2024-01-05"
    assert trade.exit_price == 8.0


def test_regular_next_open_exit_can_precede_close_floor_breach():
    matrix, guard = fixture(breach_day=4)
    exits = matrix.exit.copy()
    signal_times = matrix.exit_signal_time.copy()
    exits[3, 0] = 1
    signal_times[3, 0] = 2
    matrix = replace(matrix, exit=exits, exit_signal_time=signal_times)
    trade = run(matrix, guard, exit_fill="open_t+1").trades[0]
    assert trade.exit_reason == "signal"
    assert str(trade.exit_date) == "2024-01-05"


@pytest.mark.parametrize("breach_day,limit_down,signal_day,fill_day,fill_price", [
    (2, False, "2024-01-04", "2024-01-04", 8.5),
    (1, False, "2024-01-03", "2024-01-04", 10.0),
    (1, True, "2024-01-03", "2024-01-05", 10.0),
])
def test_bound_floor_at_close_enforces_t_plus_one_and_opening_slots(
    breach_day, limit_down, signal_day, fill_day, fill_price,
):
    matrix, guard = fixture(breach_day=breach_day, symbols=("A", "B"))
    down = matrix.limit_down_locked.copy()
    down[2, 0] = limit_down
    # The second stock tries to enter on the morning that the first sells at the close.
    entry = np.zeros_like(matrix.entry)
    times = matrix.entry_signal_time.copy()
    levels = guard.levels.copy()
    entry[1, 0] = 1
    sale_index = int(fill_day[-2:]) - 2
    entry[sale_index, 1] = 1
    times[sale_index, 1] = sale_index - 1
    levels[sale_index - 1, 1] = 9.0
    matrix = replace(matrix, entry=entry, entry_signal_time=times, limit_down_locked=down,
                     entry_close_floor=levels, opening_sell_blocked=guard.open_sell_blocked)
    config = MatcherConfig(entry_fill="open_t+1", exit_fill="close_t", max_positions=1)
    result = BacktestEngine(repo=None).simulate_market_matrix(matrix, config, bound_floor_at_close=True)
    trade, = result.trades
    assert trade.exit_reason == "structure_break"
    assert trade.exit_signal_date == signal_day
    assert trade.exit_date == fill_day
    assert trade.exit_price == fill_price
    assert result.stats["execution"]["buy_no_slot"] == 1

from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from app.backtest.engine import BacktestEngine, MatcherConfig, SimulationOptions
from app.backtest.matrix import (
    MatrixPipelineConfig,
    MatrixStrategyPipeline,
    apply_time_masks,
    build_market_data_matrix,
    build_market_matrix_from_signals,
    make_signal_matrix,
    slice_signal_matrix,
    validate_signal_matrix,
)


def fixture(*, breach=True):
    rows=[]
    for i in range(7):
        close=8.5 if breach and i==2 else 10.
        opening=8. if breach and i==3 else 10.
        rows.append({'symbol':'A','date':date(2024,1,2)+timedelta(days=i),'open':opening,'high':max(opening,close)+.5,'low':min(opening,close)-.5,'close':close,'volume':1000.,'raw_close':close,'price_limit_pct':.1})
    market=build_market_data_matrix(pl.DataFrame(rows),field_columns={'raw_close','price_limit_pct'})
    entry=np.zeros(market.shape,dtype=np.uint8)
    entry[:2,0]=1
    levels=np.full(market.shape,np.nan,dtype=np.float32)
    levels[:2,0]=[9.,1.]
    signals=make_signal_matrix(market.shape,entry=entry,entry_close_floor=levels)
    return market,signals


def matrix(market,signals):
    return build_market_matrix_from_signals(market,signals,entry_delay_bars=1,exit_delay_bars=1)


def simulate(execution):
    config=MatcherConfig(entry_fill='open_t+1',exit_fill='open_t+1',max_positions=1,fees_pct=0,slippage_bps=0)
    return BacktestEngine(repo=None).simulate_market_matrix(execution,config,options=SimulationOptions(include_monte_carlo=False))


def test_floor_survives_pipeline_time_mask_slice_and_binds_actual_entry():
    market,signals=fixture()
    class Fixed:
        def compute_signals(self,market,params):
            return signals
    piped=MatrixStrategyPipeline().run(Fixed(),market,{},MatrixPipelineConfig(basic_filter={},scoring={},order_by='score',descending=True))
    masked=apply_time_masks(piped,np.ones(7,dtype=bool),np.ones(7,dtype=bool))
    sliced=slice_signal_matrix(masked,0,7)
    np.testing.assert_array_equal(sliced.entry_close_floor,signals.entry_close_floor)
    assert not sliced.entry_close_floor.flags.writeable
    result=simulate(matrix(market,sliced))
    assert result.trades[0].exit_reason=='structure_break'
    assert str(result.trades[0].exit_signal_date)=='2024-01-04'
    assert str(result.trades[0].exit_date)=='2024-01-05'
    assert result.trades[0].exit_price==8.


def test_delayed_entry_after_missing_market_session_is_cancelled():
    market,signals=fixture(breach=False)
    changes={}
    for field in ['open','high','low','close','volume']:
        values=np.array(getattr(market,field),copy=True)
        values[1,0]=np.nan
        changes[field]=values
    market=replace(market,**changes,_valid_bars=None)
    entry=signals.entry.copy()
    entry[1,0]=0
    execution=matrix(market,replace(signals,entry=readonly(entry)))
    assert not execution.entry.any()


def readonly(values):
    values.flags.writeable=False
    return values


def test_open_at_limit_is_blocked_even_if_day_later_trades_below_limit():
    market,signals=fixture(breach=False)
    opening=np.array(market.open,copy=True)
    opening[1,0]=11.
    execution=matrix(replace(market,open=readonly(opening)),signals)
    assert not execution.entry[1,0]


def test_floor_contract_rejects_wrong_fills_missing_raw_context_and_invalid_level():
    market,signals=fixture()
    with pytest.raises(ValueError,match='next-open'):
        build_market_matrix_from_signals(market,signals)
    with pytest.raises(ValueError,match='raw_close'):
        matrix(replace(market,fields={}),signals)
    levels=signals.entry_close_floor.copy()
    levels[0,0]=np.nan
    with pytest.raises(ValueError,match='floor'):
        validate_signal_matrix(replace(signals,entry_close_floor=readonly(levels)),market.shape)


def test_end_of_sample_marks_open_position_without_last_open_forced_sale():
    market,signals=fixture(breach=False)
    result=simulate(matrix(market,signals))
    assert not result.trades
    assert result.equity_curve[-1]['positions']==1


def test_independent_mode_rejects_bound_position_floor_instead_of_ignoring_it():
    market,signals=fixture()
    with pytest.raises(ValueError,match='组合'):
        BacktestEngine(repo=None).simulate_independent_market_matrix(matrix(market,signals),2,MatcherConfig())


def test_composite_merge_does_not_silently_drop_bound_floor():
    from app.strategy.composite import merge_signal_matrices
    market,signals=fixture()
    with pytest.raises(ValueError,match='防守线'):
        merge_signal_matrices(market.shape,[signals],[('child',1.)],'union',1,10)


def test_future_floor_changes_are_included_in_causality_audit():
    from app.backtest.evidence import audit_matrix
    market,_signals=fixture(breach=False)
    def evaluate(data):
        entry=np.ones(data.shape,dtype=np.uint8)
        floor=np.full(data.shape,float(data.close[-1,0])*.9,dtype=np.float32)
        return make_signal_matrix(data.shape,entry=entry,entry_close_floor=floor)
    report=audit_matrix(market,evaluate(market),evaluate,start=market.timestamp_labels[0],end=market.timestamp_labels[-1])
    assert report['status']=='failed'
    assert any(example['field']=='entry_close_floor' for check in report['checks'] for example in check.get('examples',[]))

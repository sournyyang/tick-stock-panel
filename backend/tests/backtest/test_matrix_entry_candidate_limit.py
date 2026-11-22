from datetime import date

import numpy as np
import polars as pl
import pytest

from app.backtest.matrix import (
    MatrixPipelineConfig,
    MatrixStrategyPipeline,
    build_market_data_matrix,
    make_signal_matrix,
)


def market():
    return build_market_data_matrix(pl.DataFrame({
        'symbol':['000001.SZ','600000.SH','600001.SH'],'date':[date(2026,1,5)]*3,
        'open':[10.]*3,'high':[11.]*3,'low':[9.]*3,'close':[10.]*3,'volume':[100.]*3,
        'amount':[50.,200.,300.],
    }),field_columns={'amount'})


class Limited:
    def __init__(self,scores=(3.,2.,1.),limit=1):
        self.scores=scores
        self.limit=limit

    def entry_candidate_limit(self,params):
        return self.limit

    def compute_signals(self,market,params):
        return make_signal_matrix(market.shape,entry=np.ones(market.shape,dtype=np.uint8),exit=np.ones(market.shape,dtype=np.uint8),
                                  score=np.array([self.scores],dtype=np.float32),entry_close_floor=np.full(market.shape,9.,dtype=np.float32))


def run(strategy,**changes):
    config=MatrixPipelineConfig(basic_filter={},scoring={},order_by='score',descending=True,**changes)
    return MatrixStrategyPipeline().run(strategy,market(),{},config)


def test_daily_cap_applies_after_effective_filter_pool_and_scoring():
    strategy=Limited()
    config=MatrixPipelineConfig(basic_filter={'amount_min':100},scoring={},order_by='score',descending=True)
    actual=MatrixStrategyPipeline().run(strategy,market(),{},config)
    assert actual.entry.tolist()==[[0,1,0]]
    assert actual.exit.tolist()==[[1,1,1]]
    np.testing.assert_array_equal(actual.entry_close_floor,[[9.,9.,9.]])
    assert not actual.entry.flags.writeable and not actual.entry_close_floor.flags.writeable
    pooled=run(strategy,asset_mask=np.array([False,False,True]))
    assert pooled.entry.tolist()==[[0,0,1]]
    scored=MatrixStrategyPipeline().run(strategy,market(),{},MatrixPipelineConfig(basic_filter={},scoring={},order_by='amount',descending=False))
    assert scored.entry.tolist()==[[1,0,0]]


def test_boundary_score_ties_are_excluded_as_a_group_without_index_fallback():
    assert run(Limited((3.,3.,1.))).entry.tolist()==[[0,0,0]]
    assert run(Limited((3.,2.,2.),limit=2)).entry.tolist()==[[1,0,0]]
    assert run(Limited((3.,2.,2.),limit=3)).entry.tolist()==[[1,1,1]]
    assert run(Limited(limit=None)).entry.tolist()==[[1,1,1]]


@pytest.mark.parametrize('limit',[0,-1,True,1.5,'1'])
def test_invalid_selection_limits_fail_closed(limit):
    with pytest.raises((TypeError,ValueError),match='entry_candidate_limit'):
        run(Limited(limit=limit))


def test_optional_limit_preserves_ordinary_strategies_and_cached_signal_arrays():
    class Ordinary:
        def compute_signals(self,market,params):
            self.original=Limited().compute_signals(market,params)
            return self.original
    ordinary=Ordinary()
    assert run(ordinary).entry.tolist()==[[1,1,1]]
    np.testing.assert_array_equal(ordinary.original.entry,[[1,1,1]])

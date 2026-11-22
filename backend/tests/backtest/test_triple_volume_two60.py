import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np

from app.api.strategy import _strategy_detail
from app.backtest.matrix import (
    MatrixPipelineConfig,
    MatrixStrategyPipeline,
    slice_market_data_matrix,
)
from app.strategy.ai_generator import AIStrategyGenerator
from app.strategy.engine import StrategyEngine

SOURCE=Path(__file__).resolve().parents[3]/'examples/strategies/custom_triple_volume_two60_exit8.py'


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture():
    helper=load(Path(__file__).with_name('test_triple_volume_defense.py'),'two60_fixture')
    market=helper.fixture()
    close=market.close.copy()
    close[27]=10.55  # body3.43%: selected strategy passes, old4% recipe does not.
    fields=dict(market.fields)
    fields['raw_close']=close.copy()
    close.flags.writeable=False
    fields['raw_close'].flags.writeable=False
    return replace(market,close=close,fields=fields,_valid_bars=None)


def pipeline(market,**changes):
    module=load(SOURCE,'two60_portable')
    cfg=MatrixPipelineConfig(basic_filter=module.BASIC_FILTER,scoring={},order_by='score',descending=True,**changes)
    return MatrixStrategyPipeline().run(module.MATRIX_STRATEGY,market,{},cfg)


def test_safe_loader_exposes_8_day_exit_and_editable_portfolio_defaults(tmp_path):
    code=SOURCE.read_text()
    AIStrategyGenerator._validate_safety(code)
    AIStrategyGenerator._validate_meta_semantics(code,AIStrategyGenerator._extract_meta(code))
    path=tmp_path/'strategies/custom'/SOURCE.name
    path.parent.mkdir(parents=True)
    path.write_text(code)
    engine=StrategyEngine([path.parent])
    strategy=engine.get('custom_triple_volume_two60_exit8')
    detail=_strategy_detail(strategy)
    assert strategy.max_hold_days==8 and strategy.stop_loss is None
    assert detail['backtest_defaults']=={'initial_capital':100000,'max_positions':2,'max_exposure_pct':.6,'position_sizing':'equal','commission_pct':.0002,'stamp_tax_pct':.001,'slippage_bps':12}
    assert detail['backtest_requirements']=={'entry_fill':'open_t+1','exit_fill':'open_t+1','mode':'position','minute_fill':False}
    assert strategy.matrix_strategy.entry_candidate_limit({})==1


def test_three_percent_signal_and_daily_top1_after_asset_pool():
    market=fixture()
    old=load(SOURCE.with_name('custom_triple_volume_ma3_defense.py'),'old_four_percent')
    assert not old.MATRIX_STRATEGY.compute_signals(market,{}).entry[27].any()
    actual=pipeline(market)
    np.testing.assert_array_equal(actual.entry[27],[0,1])
    np.testing.assert_allclose(actual.entry_close_floor[27],market.close[27]*.97)
    selected=pipeline(market,asset_mask=np.array([True,False]))
    np.testing.assert_array_equal(selected.entry[27],[1,0])


def test_prefixes_permutations_and_complete_economic_ties():
    market=fixture()
    full=pipeline(market)
    for end in [26,27,28,32,45]:
        part=pipeline(slice_market_data_matrix(market,0,end))
        for name in ['entry','exit','score','entry_close_floor']:
            np.testing.assert_array_equal(getattr(full,name)[:end],getattr(part,name))
    arrays={name:np.array(getattr(market,name)[:,::-1],copy=True) for name in ['open','high','low','close','volume','tradable','limit_up_locked','limit_down_locked']}
    swapped=replace(market,**arrays,fields={n:v[:,::-1] for n,v in market.fields.items()},symbols=market.symbols[::-1],names=market.names[::-1],_valid_bars=None)
    shuffled=pipeline(swapped)
    for name in ['entry','exit','score','entry_close_floor']:
        np.testing.assert_array_equal(getattr(full,name),getattr(shuffled,name)[:,::-1])
    volume=market.volume.copy()
    volume[:,1]=volume[:,0]
    assert not pipeline(replace(market,volume=volume,_valid_bars=None)).entry.any()

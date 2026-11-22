import importlib.util
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from app.backtest.matrix import build_market_data_matrix, slice_market_data_matrix
from app.strategy.ai_generator import AIStrategyGenerator
from app.strategy.engine import StrategyEngine

SOURCE=Path(__file__).resolve().parents[3]/'examples/strategies/custom_triple_volume_ma3_defense.py'


def strategy():
    spec=importlib.util.spec_from_file_location('triple_volume_example',SOURCE)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MATRIX_STRATEGY


def fixture():
    values=np.tile([9.9,10.1,9.8,10.,100.],(45,1))
    values[20]=[10.,10.8,9.8,10.6,350.]
    values[21]=[10.6,11.,10.4,10.65,200.]
    values[22:27]=[10.3,10.7,10.2,10.4,70.]
    values[27]=[10.2,10.8,10.1,10.7,120.]
    rows=[]
    for a,symbol in enumerate(['000001.SZ','600000.SH']):
        for i,r in enumerate(values):
            rows.append({'symbol':symbol,'date':date(2022,1,1)+timedelta(days=i),'open':r[0],'high':r[1],'low':r[2],'close':r[3],'volume':r[4]*(a+1),'raw_close':r[3]})
    return build_market_data_matrix(pl.DataFrame(rows),field_columns={'raw_close','price_limit_pct'})


def test_portable_strategy_passes_safe_loader_and_defaults(tmp_path):
    code=SOURCE.read_text()
    AIStrategyGenerator._validate_safety(code)
    meta=AIStrategyGenerator._extract_meta(code)
    AIStrategyGenerator._validate_meta_semantics(code,meta)
    target=tmp_path/'strategies/custom'/SOURCE.name
    target.parent.mkdir(parents=True)
    target.write_text(code)
    engine=StrategyEngine([target.parent])
    loaded=engine.get('custom_triple_volume_ma3_defense')
    assert loaded.execution_backend=='matrix_native' and loaded.stop_loss is None
    assert loaded.meta['default_exit_fill']=='open_t+1'
    from app.api.strategy import _strategy_detail
    assert _strategy_detail(loaded)['backtest_requirements']=={
        'entry_fill':'open_t+1','exit_fill':'open_t+1','mode':'position','minute_fill':False,
    }
    assert not loaded.meta.get('research_only')


def test_setup_score_floor_and_prefixes_do_not_depend_on_future_length():
    market=fixture()
    signals=strategy().compute_signals(market,{})
    np.testing.assert_array_equal(np.flatnonzero(signals.entry[:,0]),[27])
    assert signals.score[27,1]>signals.score[27,0]
    np.testing.assert_allclose(signals.entry_close_floor[27],market.close[27]*.97)
    for end in [26,27,28,32,45]:
        prefix=strategy().compute_signals(slice_market_data_matrix(market,0,end),{})
        for name in ['entry','exit','score','entry_close_floor']:
            np.testing.assert_array_equal(getattr(prefix,name),getattr(signals,name)[:end])


def test_column_permutation_and_identical_economic_vectors():
    market=fixture()
    signals=strategy().compute_signals(market,{})
    changes={name:np.array(getattr(market,name)[:,::-1],copy=True) for name in ['open','high','low','close','volume','tradable','limit_up_locked','limit_down_locked']}
    fields={name:values[:,::-1] for name,values in market.fields.items()}
    swapped=replace(market,**changes,fields=fields,symbols=market.symbols[::-1],names=market.names[::-1],_valid_bars=None)
    shuffled=strategy().compute_signals(swapped,{})
    for name in ['entry','exit','score','entry_close_floor']:
        np.testing.assert_array_equal(getattr(signals,name),getattr(shuffled,name)[:,::-1])
    volume=market.volume.copy()
    volume[:,1]=volume[:,0]
    tied=strategy().compute_signals(replace(market,volume=volume,_valid_bars=None),{})
    assert not tied.entry.any()

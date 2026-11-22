from datetime import date
from types import SimpleNamespace

import pytest

from app.backtest.strategy import StrategyBacktestConfig, StrategyBacktestService
from app.strategy.engine import _normalize_backtest_requirements

REQUIREMENTS={'entry_fill':'open_t+1','exit_fill':'open_t+1','mode':'position','minute_fill':False}


@pytest.mark.parametrize('changes,label',[
    ({'entry_fill':'close_t'},'建仓口径'),
    ({'exit_fill':'close_t'},'清仓口径'),
    ({'mode':'full'},'仓位模拟'),
    ({'minute_fill':True},'分钟成交'),
])
def test_incompatible_execution_returns_short_error_before_loading_market(changes,label):
    strategy=SimpleNamespace(
        meta={'id':'bound','name':'收盘防守','asset_types':['stock'],'timeframes':['1d'],'backtest_requirements':REQUIREMENTS},
        execution_backend='matrix_native',
    )
    class NoMarket:
        def load_market_data_matrix_for_backtest(self,*args,**kwargs):
            pytest.fail('invalid execution reached market loading')
    service=StrategyBacktestService(NoMarket(),SimpleNamespace(get=lambda _sid:strategy))
    config=StrategyBacktestConfig(strategy_id='bound',symbols=None,start=date(2026,1,1),end=date(2026,1,5),entry_fill='open_t+1',exit_fill='open_t+1')
    for field,value in changes.items():
        setattr(config,field,value)
    result=service.run(config)
    assert result.error and label in result.error
    assert 'Traceback' not in result.error and '/app/' not in result.error


@pytest.mark.parametrize('raw',[{'unknown':True},{'entry_fill':'today'},{'minute_fill':'false'},[]])
def test_bad_requirement_definitions_fail_closed(raw):
    with pytest.raises(ValueError,match='backtest_requirements'):
        _normalize_backtest_requirements(raw)


def test_optional_requirements_preserve_defaults_and_copy_values():
    assert _normalize_backtest_requirements(None)=={}
    actual=_normalize_backtest_requirements(REQUIREMENTS)
    assert actual==REQUIREMENTS and actual is not REQUIREMENTS

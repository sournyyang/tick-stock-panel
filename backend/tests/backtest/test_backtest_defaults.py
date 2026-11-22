import pytest

from app.strategy.engine import _normalize_backtest_defaults


def test_portfolio_defaults_are_optional_and_copied():
    assert _normalize_backtest_defaults(None)=={}
    raw={'initial_capital':100000,'max_positions':2,'max_exposure_pct':.6,'position_sizing':'equal','commission_pct':.0002,'stamp_tax_pct':.001,'slippage_bps':12}
    actual=_normalize_backtest_defaults(raw)
    assert actual==raw and actual is not raw


@pytest.mark.parametrize('raw',[
    [],{'extra':1},{'max_positions':True},{'max_positions':1.5},{'max_positions':0},
    {'initial_capital':0},{'max_exposure_pct':1.1},{'slippage_bps':-1},{'slippage_bps':float('inf')},
    {'commission_pct':float('nan')},{'stamp_tax_pct':-.01},{'position_sizing':'policy'},
])
def test_invalid_portfolio_defaults_are_rejected(raw):
    with pytest.raises(ValueError,match='backtest_defaults'):
        _normalize_backtest_defaults(raw)

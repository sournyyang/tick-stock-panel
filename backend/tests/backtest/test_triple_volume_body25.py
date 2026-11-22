import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from app.backtest.matrix import (
    MatrixPipelineConfig,
    MatrixStrategyPipeline,
    slice_market_data_matrix,
)
from app.strategy.ai_generator import AIStrategyGenerator
from app.strategy.engine import StrategyEngine

SOURCE = Path(__file__).resolve().parents[3] / 'examples/strategies/custom_triple_volume_two60_body25.py'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(body=.028):
    helper = load(Path(__file__).with_name('test_triple_volume_two60.py'), 'body25_fixture')
    market = helper.fixture()
    close = market.close.copy()
    close[27] = market.open[27] * (1 + body)
    fields = {**market.fields, 'raw_close': close.copy()}
    return replace(market, close=close, fields=fields, _valid_bars=None)


def pipeline(market, path=SOURCE):
    module = load(path, 'body25_portable')
    return MatrixStrategyPipeline().run(module.MATRIX_STRATEGY, market, {}, MatrixPipelineConfig(
        basic_filter=module.BASIC_FILTER, scoring={}, order_by='score', descending=True))


def test_body25_safety_and_fixed_money_costs(tmp_path):
    code = SOURCE.read_text()
    AIStrategyGenerator._validate_safety(code)
    AIStrategyGenerator._validate_meta_semantics(code, AIStrategyGenerator._extract_meta(code))
    path = tmp_path / SOURCE.name
    path.write_text(code)
    strategy = StrategyEngine([tmp_path]).get('custom_triple_volume_two60_body25')
    assert strategy.max_hold_days == 8
    assert strategy.meta['backtest_defaults'] == {
        'initial_capital':100000, 'max_positions':2, 'max_exposure_pct':.6,
        'commission_pct':.0002, 'stamp_tax_pct':.001, 'slippage_bps':12, 'position_sizing':'equal'}
    assert strategy.meta['backtest_requirements'] == {
        'entry_fill':'open_t+1', 'exit_fill':'open_t+1', 'mode':'position', 'minute_fill':False}
    assert strategy.matrix_strategy.entry_candidate_limit({}) == 1


@pytest.mark.parametrize('body,expected', [(.028, True), (.024, False)])
def test_body_threshold_changes_real_entry_not_statistics(body, expected):
    market = fixture(body)
    selected = pipeline(market)
    assert bool(selected.entry[27].any()) is expected
    old = pipeline(market, SOURCE.with_name('custom_triple_volume_two60_exit8.py'))
    assert not old.entry[27].any()
    if expected:
        np.testing.assert_array_equal(selected.entry[27], [0, 1])
        np.testing.assert_allclose(selected.entry_close_floor[27], market.close[27] * .97)


def test_body25_prefixes_and_economic_ties():
    market = fixture()
    full = pipeline(market)
    for end in [27, 28, 32, 45]:
        partial = pipeline(slice_market_data_matrix(market, 0, end))
        for name in ['entry', 'exit', 'score', 'entry_close_floor']:
            np.testing.assert_array_equal(getattr(full, name)[:end], getattr(partial, name))
    volume = market.volume.copy()
    volume[:, 1] = volume[:, 0]
    assert not pipeline(replace(market, volume=volume, _valid_bars=None)).entry.any()

# tests/test_metrics_calculator_v2_5.py
# EOTS v2.5 - SENTRY-APPROVED CANONICAL SCRIPT
#
# This file contains comprehensive unit tests for the refactored
# MetricsCalculatorV2_5, ensuring its components function correctly
# and handle edge cases gracefully.

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date

# --- Mock Objects and Test Fixtures ---
# Mock the dependencies (ConfigManager, HistoricalDataManager)
class MockConfigManager:
    def get_setting(self, *keys, default=None):
        # Simplified lookup for testing
        config = {
            "system_settings": {"metric_calculation_phases_activation": {"run_metric_orchestration": True}},
            "strategy_settings": {"strike_col_name": "strike", "underlying_price_col_name": "price", "expiration_col_name": "expiration_days_from_epoch_calc", "contract_multiplier_col_name": "multiplier"},
            "data_processor_settings": {"normalization_clip_percentile": 99.0, "coefficients": {"dag_alpha":{}}}
        }
        val = config
        for key in keys:
            val = val.get(key, {})
        return val if val else default

class MockHistoricalDataManager:
    def get_ohlc_history_for_atr(self, symbol, num_days, current_date):
        # Return a predictable DataFrame for ATR calculation tests
        dates = pd.to_datetime([current_date - pd.Timedelta(days=i) for i in range(num_days)])
        return pd.DataFrame({
            'date': dates,
            'high': np.full(num_days, 102.0),
            'low': np.full(num_days, 98.0),
            'close': np.full(num_days, 100.0)
        })

@pytest.fixture
def metrics_calculator_instance():
    """Pytest fixture to create an instance of the MetricsCalculatorV2_5."""
    from core_analytics_engine.metrics_calculator_v2_5 import MetricsCalculatorV2_5
    mock_cm = MockConfigManager()
    mock_hdm = MockHistoricalDataManager()
    return MetricsCalculatorV2_5(mock_cm, mock_hdm)

@pytest.fixture
def sample_raw_options_df():
    """Pytest fixture to create a sample raw options DataFrame."""
    return pd.DataFrame({
        'strike': [95.0, 100.0, 105.0],
        'opt_kind': ['put', 'call', 'call'],
        'expiration_days_from_epoch_calc': [(date.today() - date(1970, 1, 1)).days] * 3, # All 0DTE
        'gxoi': [1e6, 2e6, 1.5e6],
        'dxoi': [-5e7, 8e7, 6e7],
        'vxoi': [1e5, 1.2e5, 1.1e5],
        'txoi': [-2e4, -1.8e4, -1.9e4],
        'charmxoi': [5e3, -4e3, -3e3],
        'vannaxoi': [-2e4, 3e4, 2.5e4],
        'vommaxoi': [1e4, 1.2e4, 1.1e4],
        'volatility': [0.3, 0.25, 0.28]
        # Add other necessary raw columns for testing all metric calculations
    })

@pytest.fixture
def sample_underlying_data():
    """Pytest fixture for sample underlying data."""
    return {
        "symbol": "TEST",
        "price": 100.1,
        "multiplier": 100.0,
        "call_gxoi": 3.5e6,
        "put_gxoi": 1e6,
        # Add other necessary underlying aggregate data
    }

# --- Unit Tests ---

def test_metrics_calculator_initialization(metrics_calculator_instance):
    """Test if the MetricsCalculatorV2_5 and its sub-components initialize correctly."""
    assert metrics_calculator_instance is not None
    assert hasattr(metrics_calculator_instance, '_structural_metrics')
    assert hasattr(metrics_calculator_instance, '_flow_metrics')
    assert hasattr(metrics_calculator_instance, '_volatility_metrics')
    assert hasattr(metrics_calculator_instance, '_time_decay_metrics')
    assert metrics_calculator_instance.config_manager is not None

def test_calculate_dte(metrics_calculator_instance, sample_raw_options_df):
    """Test the DTE calculation utility."""
    df = sample_raw_options_df.copy()
    today = date.today()
    dte_series = metrics_calculator_instance._calculate_dte(df, today)
    assert not dte_series.empty
    assert (dte_series == 0).all() # All contracts were set to today's date

def test_normalize_series(metrics_calculator_instance):
    """Test the series normalization utility."""
    series = pd.Series([-100, -50, 0, 50, 100])
    normalized = metrics_calculator_instance._normalize_series(series, "max_abs")
    assert np.isclose(normalized.max(), 1.0)
    assert np.isclose(normalized.min(), -1.0)
    assert np.isclose(normalized.iloc[2], 0.0)

def test_full_orchestration_cycle(metrics_calculator_instance, sample_raw_options_df, sample_underlying_data):
    """
    Test the main orchestration method `calculate_all_metrics` to ensure it runs
    without errors and produces outputs of the correct type and basic structure.
    """
    df_chain, df_strike, und_enriched = metrics_calculator_instance.calculate_all_metrics(
        options_df_raw=sample_raw_options_df,
        und_data_api_raw=sample_underlying_data,
        current_time_dt=datetime.now(),
        symbol="TEST"
    )

    # Validate output types
    assert isinstance(df_chain, pd.DataFrame)
    assert isinstance(df_strike, pd.DataFrame)
    assert isinstance(und_enriched, dict)

    # Validate that key metrics columns have been added
    assert 'dag_custom' in df_chain.columns
    assert 'mspi' in df_strike.columns
    assert 'nvp_strike' in df_strike.columns
    assert 'GIB_OI_based_Und' in und_enriched

    # Validate that data is not empty (for this specific test case)
    assert not df_chain.empty
    assert not df_strike.empty
    assert und_enriched is not None

def test_edge_case_empty_input(metrics_calculator_instance, sample_underlying_data):
    """Test the orchestrator's behavior with an empty options DataFrame."""
    empty_df = pd.DataFrame()
    df_chain, df_strike, und_enriched = metrics_calculator_instance.calculate_all_metrics(
        options_df_raw=empty_df,
        und_data_api_raw=sample_underlying_data,
        current_time_dt=datetime.now(),
        symbol="EMPTYTEST"
    )
    assert df_chain.empty
    assert df_strike.empty
    assert 'GIB_OI_based_Und' in und_enriched # Underlying metrics should still calculate
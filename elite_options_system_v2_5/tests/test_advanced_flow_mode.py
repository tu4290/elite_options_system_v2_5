# elite_options_system_v2_5/tests/test_advanced_flow_mode.py
import pytest
from datetime import datetime
import plotly.graph_objects as go
import pandas as pd # For pd.isna checks if needed, or float('nan')
import numpy as np # For float('nan')

from elite_options_system_v2_5.dashboard_application.modes.advanced_flow_mode import (
    create_vapi_fa_figure,
    create_dwfd_figure,
    create_tw_laf_figure
)
from elite_options_system_v2_5.data_models.eots_schemas_v2_5 import ProcessedUnderlyingAggregatesV2_5

@pytest.fixture
def sample_app_config():
    return {
        "visualization_settings": {
            "dashboard": {
                "default_graph_height": 350,
                "chart_specific_heights": {
                    "vapi_fa_z_score_chart": 300,
                    "dwfd_z_score_chart": 300,
                    "tw_laf_z_score_chart": 300,
                },
                "advanced_flow_chart_settings": {
                    "z_score_levels": [
                        {"name": "strong_negative", "threshold": -2.0, "color": "#d62728"},
                        {"name": "mild_negative", "threshold": -0.5, "color": "#ff9896"},
                        {"name": "neutral", "threshold": 0.5, "color": "#aec7e8"},
                        {"name": "mild_positive", "threshold": 2.0, "color": "#98df8a"},
                        {"name": "strong_positive", "threshold": None, "color": "#2ca02c"}
                    ],
                    "default_bar_color": "#cccccc",
                    "show_bar_threshold_lines_default": True
                }
            }
        },
        # Adding root level keys used by add_timestamp_annotation if needed by _create_z_score_figure
        "system_settings": {
            "logging_settings": {
                "log_level_plotly_utils": "INFO" # Example, adjust if specific values are read
            }
        }
    }

@pytest.fixture
def sample_underlying_data_factory():
    def _create_data(
        symbol="TEST",
        timestamp_dt=datetime(2023, 1, 1, 12, 0, 0),
        price_val=100.0,
        vapi_z=None,
        dwfd_z=None,
        twlaf_z=None
    ):
        # Based on ProcessedUnderlyingAggregatesV2_5 and its parent RawUnderlyingDataCombinedV2_5
        # Required fields: symbol, timestamp
        # Optional fields in RawUnderlyingDataCombinedV2_5 (like price) are given defaults here.
        return ProcessedUnderlyingAggregatesV2_5(
            symbol=symbol,
            timestamp=timestamp_dt,
            price=price_val, 
            # Initialize other potential fields from ProcessedUnderlyingAggregatesV2_5 to None or sensible defaults
            # if they are ever accessed by the tested functions or their helpers indirectly.
            # For now, focusing on those directly impacting the Z-score figures.
            vapi_fa_z_score_und=vapi_z,
            dwfd_z_score_und=dwfd_z,
            tw_laf_z_score_und=twlaf_z,
            # Add other numerous optional fields from the schema with None to ensure model validation passes
            day_open_price_und=None,
            day_high_price_und=None,
            day_low_price_und=None,
            prev_day_close_price_und=None,
            tradier_iv5_approx_smv_avg=None,
            u_volatility=None,
            total_call_oi_und=None,
            total_put_oi_und=None,
            total_call_vol_und=None,
            total_put_vol_und=None,
            gib_oi_based_und=None,
            td_gib_und=None,
            hp_eod_und=None,
            net_cust_delta_flow_und=None,
            net_cust_gamma_flow_und=None,
            net_cust_vega_flow_und=None,
            net_cust_theta_flow_und=None,
            net_value_flow_5m_und=None,
            net_vol_flow_5m_und=None,
            net_value_flow_15m_und=None,
            net_vol_flow_15m_und=None,
            net_value_flow_30m_und=None,
            net_vol_flow_30m_und=None,
            net_value_flow_60m_und=None,
            net_vol_flow_60m_und=None,
            vri_0dte_und_sum=None,
            vfi_0dte_und_sum=None,
            vvr_0dte_und_avg=None,
            vci_0dte_agg=None,
            arfi_overall_und_avg=None,
            a_mspi_und_summary_score=None,
            a_sai_und_avg=None,
            a_ssi_und_avg=None,
            vri_2_0_und_aggregate=None,
            ivsdh_surface_data=None,
            current_market_regime_v2_5=None,
            ticker_context_dict_v2_5=None,
            atr_und=None
        )
    return _create_data

# Tests for create_vapi_fa_figure
def test_create_vapi_fa_figure_valid_data(sample_app_config, sample_underlying_data_factory):
    data = sample_underlying_data_factory(symbol="VAPI_S", vapi_z=1.2)
    fig = create_vapi_fa_figure(data, sample_app_config)
    assert isinstance(fig, go.Figure)
    assert "VAPI-FA" in fig.layout.title.text
    assert "VAPI_S" in fig.layout.title.text
    assert fig.data[0].x[0] == 1.2
    assert fig.layout.height == 300 # From chart_specific_heights

def test_create_vapi_fa_figure_none_data(sample_app_config, sample_underlying_data_factory):
    data = sample_underlying_data_factory(symbol="VAPI_N", vapi_z=None)
    fig = create_vapi_fa_figure(data, sample_app_config)
    assert isinstance(fig, go.Figure)
    assert fig.layout.annotations # Check that annotations list exists
    assert len(fig.layout.annotations) > 0 # Basic check, specific text check is better
    # The create_empty_figure adds annotation at index 0 if reason is provided.
    # _create_z_score_figure calls create_empty_figure with a reason.
    assert "VAPI-FA Z-Score not available." in fig.layout.annotations[0].text
    assert "VAPI_N" in fig.layout.title.text

def test_create_vapi_fa_figure_nan_data(sample_app_config, sample_underlying_data_factory):
    data = sample_underlying_data_factory(symbol="VAPI_NAN", vapi_z=np.nan) # Using np.nan
    fig = create_vapi_fa_figure(data, sample_app_config)
    assert isinstance(fig, go.Figure)
    assert fig.layout.annotations
    assert len(fig.layout.annotations) > 0
    assert "VAPI-FA Z-Score not available." in fig.layout.annotations[0].text
    assert "VAPI_NAN" in fig.layout.title.text

# Tests for create_dwfd_figure
def test_create_dwfd_figure_valid_data(sample_app_config, sample_underlying_data_factory):
    data = sample_underlying_data_factory(symbol="DWFD_S", dwfd_z=-0.8)
    fig = create_dwfd_figure(data, sample_app_config)
    assert isinstance(fig, go.Figure)
    assert "DWFD" in fig.layout.title.text
    assert "DWFD_S" in fig.layout.title.text
    assert fig.data[0].x[0] == -0.8
    assert fig.layout.height == 300

def test_create_dwfd_figure_none_data(sample_app_config, sample_underlying_data_factory):
    data = sample_underlying_data_factory(symbol="DWFD_N", dwfd_z=None)
    fig = create_dwfd_figure(data, sample_app_config)
    assert isinstance(fig, go.Figure)
    assert fig.layout.annotations
    assert len(fig.layout.annotations) > 0
    assert "DWFD Z-Score not available." in fig.layout.annotations[0].text
    assert "DWFD_N" in fig.layout.title.text

def test_create_dwfd_figure_nan_data(sample_app_config, sample_underlying_data_factory):
    data = sample_underlying_data_factory(symbol="DWFD_NAN", dwfd_z=np.nan)
    fig = create_dwfd_figure(data, sample_app_config)
    assert isinstance(fig, go.Figure)
    assert fig.layout.annotations
    assert len(fig.layout.annotations) > 0
    assert "DWFD Z-Score not available." in fig.layout.annotations[0].text
    assert "DWFD_NAN" in fig.layout.title.text
    
# Tests for create_tw_laf_figure
def test_create_tw_laf_figure_valid_data(sample_app_config, sample_underlying_data_factory):
    data = sample_underlying_data_factory(symbol="TWLAF_S", twlaf_z=2.5)
    fig = create_tw_laf_figure(data, sample_app_config)
    assert isinstance(fig, go.Figure)
    assert "TW-LAF" in fig.layout.title.text
    assert "TWLAF_S" in fig.layout.title.text
    assert fig.data[0].x[0] == 2.5
    assert fig.layout.height == 300

def test_create_tw_laf_figure_none_data(sample_app_config, sample_underlying_data_factory):
    data = sample_underlying_data_factory(symbol="TWLAF_N", twlaf_z=None)
    fig = create_tw_laf_figure(data, sample_app_config)
    assert isinstance(fig, go.Figure)
    assert fig.layout.annotations
    assert len(fig.layout.annotations) > 0
    assert "TW-LAF Z-Score not available." in fig.layout.annotations[0].text
    assert "TWLAF_N" in fig.layout.title.text

def test_create_tw_laf_figure_nan_data(sample_app_config, sample_underlying_data_factory):
    data = sample_underlying_data_factory(symbol="TWLAF_NAN", twlaf_z=np.nan)
    fig = create_tw_laf_figure(data, sample_app_config)
    assert isinstance(fig, go.Figure)
    assert fig.layout.annotations
    assert len(fig.layout.annotations) > 0
    assert "TW-LAF Z-Score not available." in fig.layout.annotations[0].text
    assert "TWLAF_NAN" in fig.layout.title.text

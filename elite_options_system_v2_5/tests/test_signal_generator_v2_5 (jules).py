# tests/test_signal_generator_v2_5.py
# EOTS v2.5 - SENTRY-APPROVED CANONICAL SCRIPT
#
# This file contains unit tests for the SignalGeneratorV2_5.

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta 
from typing import List, Dict, Any, Callable

# EOTS V2.5 Imports - Assuming they are discoverable in the path
# If not, adjust sys.path or expect ImportError and skip tests.
try:
    from data_models.eots_schemas_v2_5 import (
        ProcessedDataBundleV2_5,
        ProcessedStrikeLevelMetricsV2_5,
        ProcessedUnderlyingAggregatesV2_5,
        ProcessedContractMetricsV2_5, 
        SignalPayloadV2_5,
        TickerContextV2_5, 
        MarketRegimeV2_5 
    )
    from core_analytics_engine.signal_generator_v2_5 import SignalGeneratorV2_5
    SCHEMAS_LOADED = True
except ImportError:
    SCHEMAS_LOADED = False
    # Define dummy classes if import fails, so tests can be skipped gracefully
    class ProcessedDataBundleV2_5: pass
    class ProcessedStrikeLevelMetricsV2_5: 
        def __init__(self, **kwargs): self.__dict__.update(kwargs)
    class ProcessedUnderlyingAggregatesV2_5: 
        def __init__(self, **kwargs): self.__dict__.update(kwargs)
    class ProcessedContractMetricsV2_5: pass
    class SignalPayloadV2_5: pass
    class TickerContextV2_5: 
        def __init__(self, is_0dte=False): self.is_0dte = is_0dte
        def dict(self): return {"is_0dte": self.is_0dte}

    class MarketRegimeV2_5: 
        REGIME_BULLISH_TREND = "REGIME_BULLISH_TREND"
        REGIME_BEARISH_TREND = "REGIME_BEARISH_TREND"
        REGIME_NEUTRAL_CONGESTION = "REGIME_NEUTRAL_CONGESTION"
        REGIME_EXTREME_BEARISH_PANIC = "REGIME_EXTREME_BEARISH_PANIC"
        REGIME_VOL_EXPANSION_IMMINENT_VRI0DTE_BULLISH = "REGIME_VOL_EXPANSION_IMMINENT_VRI0DTE_BULLISH"
        REGIME_VOL_EXPANSION_IMMINENT_VRI0DTE_BEARISH = "REGIME_VOL_EXPANSION_IMMINENT_VRI0DTE_BEARISH"
        REGIME_VOL_CONTRACTION_NEUTRAL = "REGIME_VOL_CONTRACTION_NEUTRAL"
        REGIME_VANNA_CASCADE_ALERT_BULLISH = "REGIME_VANNA_CASCADE_ALERT_BULLISH"
        REGIME_VANNA_CASCADE_ALERT_BEARISH = "REGIME_VANNA_CASCADE_ALERT_BEARISH"
        REGIME_EOD_HEDGING_PRESSURE_BUY = "REGIME_EOD_HEDGING_PRESSURE_BUY"
        REGIME_EOD_HEDGING_PRESSURE_SELL = "REGIME_EOD_HEDGING_PRESSURE_SELL"
        REGIME_TREND_EXHAUSTION_RISK_TOP = "REGIME_TREND_EXHAUSTION_RISK_TOP" # For Bubble
        REGIME_TREND_EXHAUSTION_RISK_BOTTOM = "REGIME_TREND_EXHAUSTION_RISK_BOTTOM"
        REGIME_CASCADE_RISK_CHARM = "REGIME_CASCADE_RISK_CHARM" 


    class SignalGeneratorV2_5: 
        def __init__(self, config_manager): self.config_manager = config_manager
        def generate_all_signals(self, bundle): return {"directional": [], "volatility": [], "time_decay": [], "complex_signals": [], "v2_5_enhanced": []}


# --- Mock Objects and Test Fixtures ---

class MockConfigManagerSignalGenerator:
    """A mock ConfigManager tailored for testing SignalGeneratorV2_5."""
    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict

    def get_setting(self, *keys: str, default: Any = None) -> Any:
        value = self._config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else: 
                return default
        return value
        
    def update_setting(self, keys_tuple: tuple, value: Any):
        d = self._config
        for key in keys_tuple[:-1]:
            d = d.setdefault(key, {})
        d[keys_tuple[-1]] = value


@pytest.fixture
def mock_config_manager_sg():
    default_config = {
        "system_settings": {
            "signal_activation": {
                "directional_signals": True, "volatility_signals": True,
                "time_decay_signals": True, "complex_signals": True,
                "v2_5_enhanced_signals": True,
            }
        },
        "strategy_settings": {
            "thresholds": {
                "mspi_bullish_confirmation_min_val": 0.6, "mspi_bearish_confirmation_min_val": -0.6, 
                "sai_high_conviction": 0.7, "sai_moderate_conviction": 0.5,
                "vapi_fa_strong_z_score": 2.0, "vapi_fa_moderate_z_score": 1.5,
                "dwfd_strong_z_score": 2.0, "dwfd_moderate_z_score": 1.5,
                "tw_laf_strong_z_score": 2.0, "tw_laf_moderate_z_score": 1.5,
                "vri_expansion_threshold": 0.3, "vri_contraction_threshold": -0.3, 
                "d_tdpi_pin_risk_min_abs_val_0dte": 1.5, 
                "default_min_strength_score": 0.3, "min_vol_confirmation_score": 0.5, 
                "sdag_mult_bullish_threshold": 1.1, "sdag_mult_bearish_threshold": 0.9,
                "sdag_dir_bullish_threshold": 0.5, "sdag_dir_bearish_threshold": -0.5,
                "sdag_conviction_min_aligned_components": 2, 
                "e_ctr_strong_threshold": 1.5, "e_tdfi_strong_threshold": 1.5, 
                "a_ssi_low_threshold": -0.5, "dwfd_divergence_threshold": -1.0, 
                "hp_eod_buy_pressure_threshold": 0.005, "hp_eod_sell_pressure_threshold": -0.005,
                "arfi_divergence_threshold": 0.5, 
                "vol_skew_shift_min_abs_skewfactor_global": 0.2, # Example
            },
            "signal_config": { 
                "mspi_sai_bullish": {"name": "MSPI_SAI_Bullish", "base_score": 0.7},
                "mspi_sai_bearish": {"name": "MSPI_SAI_Bearish", "base_score": -0.7},
                "vapi_fa_bullish_momentum": {"name": "VAPI-FA_Bullish_Momentum"},
                "vapi_fa_bearish_momentum": {"name": "VAPI-FA_Bearish_Momentum"},
                "sdag_conviction_bullish": {"name": "SDAG_Conviction_Bullish"},
                "sdag_conviction_bearish": {"name": "SDAG_Conviction_Bearish"},
                "vol_expansion_0dte": {"name": "Volatility_Expansion_0DTE"},
                "vol_expansion_vri2": {"name": "Volatility_Expansion_VRI2.0"},
                "vol_contraction": {"name": "Volatility_Contraction"},
                "pin_risk_0dte": {"name": "PinRisk_0DTE"},
                "charm_cascade_alert": {"name": "Charm_Cascade_Alert"},
                "structure_change_alert_low_ssi": {"name": "Structure_Change_Alert_Low_ASSI"},
                "flow_divergence_warning": {"name": "Flow_Divergence_Warning_ARFI"},
                "dwfd_smart_money_bullish": {"name": "DWFD_Smart_Money_Bullish"},
                "dwfd_smart_money_bearish": {"name": "DWFD_Smart_Money_Bearish"},
                "tw_laf_bullish_trend_confirm": {"name": "TW-LAF_Bullish_Trend_Confirmation"},
                "tw_laf_bearish_trend_confirm": {"name": "TW-LAF_Bearish_Trend_Confirmation"},
                "vanna_cascade_alert_bullish": {"name": "Vanna_Cascade_Alert_Bullish"},
                "vanna_cascade_alert_bearish": {"name": "Vanna_Cascade_Alert_Bearish"},
                "eod_hedging_flow_imminent_buying": {"name": "EOD_Hedging_Flow_Imminent_Buying"},
                "eod_hedging_flow_imminent_selling": {"name": "EOD_Hedging_Flow_Imminent_Selling"},
                "vol_skew_shift_alert": {"name": "Volatility_Skew_Shift_Alert"},
                "bubble_mispricing_warning": {"name": "Bubble_Mispricing_Warning"},
            }
        }
    }
    return MockConfigManagerSignalGenerator(default_config)

@pytest.fixture
def signal_generator_instance(mock_config_manager_sg):
    if not SCHEMAS_LOADED:
        pytest.skip("Schemas not loaded, cannot run SignalGenerator tests.")
    return SignalGeneratorV2_5(config_manager=mock_config_manager_sg)

@pytest.fixture
def processed_data_bundle_factory() -> Callable[..., ProcessedDataBundleV2_5]:
    def _factory(
        symbol: str = "TEST_STOCK",
        current_market_regime: str = MarketRegimeV2_5.REGIME_NEUTRAL_CONGESTION if SCHEMAS_LOADED else "REGIME_NEUTRAL_CONGESTION",
        is_0dte: bool = False,
        underlying_metrics: Dict[str, Any] = None,
        strike_metrics_list: List[Dict[str, Any]] = None, 
        contract_metrics_list: List[Dict[str, Any]] = None 
    ) -> ProcessedDataBundleV2_5:
        if not SCHEMAS_LOADED:
            pytest.skip("Schemas not loaded, cannot create ProcessedDataBundleV2_5.")

        default_underlying_metrics = {
            "vapi_fa_z_score_und": 0.0, "dwfd_z_score_und": 0.0, "tw_laf_z_score_und": 0.0,
            "vri_2_0_und_aggregate": 0.0, "arfi_overall_und_avg": 0.0, "hp_eod_und": 0.0,
            "gib_oi_based_und": 0.0, "td_gib_und": 0.0, "vci_0dte_agg": 0.0, "vri_0dte_und_sum": 0.0, "vfi_0dte_und_sum":0.0,
            "price": 100.0, "atm_iv": 0.20, "a_ssi_und_avg": 0.0,
            "skewfactor_global":0.0, 
        }
        if underlying_metrics:
            default_underlying_metrics.update(underlying_metrics)
        
        final_underlying_metrics = {**default_underlying_metrics}
        
        enriched_und_data_dict = {
            "symbol":symbol, "timestamp":datetime.now(timezone.utc),
            "current_market_regime_v2_5":current_market_regime,
            "ticker_context_dict_v2_5":TickerContextV2_5(is_0dte=is_0dte).dict(),
            **final_underlying_metrics
        }
        for field in ProcessedUnderlyingAggregatesV2_5.__fields__:
            if field not in enriched_und_data_dict:
                enriched_und_data_dict[field] = None 

        enriched_und_data = ProcessedUnderlyingAggregatesV2_5(**enriched_und_data_dict)


        processed_strikes = []
        if strike_metrics_list:
            for sm_dict in strike_metrics_list:
                if 'strike' not in sm_dict: sm_dict['strike'] = 100.0 
                for field in ProcessedStrikeLevelMetricsV2_5.__fields__:
                    if field not in sm_dict :
                        if ProcessedStrikeLevelMetricsV2_5.__fields__[field].outer_type_ is float:
                            sm_dict[field] = 0.0
                        elif ProcessedStrikeLevelMetricsV2_5.__fields__[field].outer_type_ is int:
                             sm_dict[field] = 0
                processed_strikes.append(ProcessedStrikeLevelMetricsV2_5(**sm_dict))
        
        processed_contracts = [] 

        return ProcessedDataBundleV2_5(
            options_df_raw=pd.DataFrame(), und_data_api_raw={}, 
            options_data_with_metrics=processed_contracts, 
            strike_level_data_with_metrics=processed_strikes,
            underlying_data_enriched=enriched_und_data,
            heatmap_data={}, 
            current_time_dt=datetime.now(timezone.utc),
            symbol=symbol, processing_timestamp=datetime.now(timezone.utc)
        )
    return _factory


# --- Initial Signal Tests (Phase 1) ---

def test_directional_mspi_sai_bullish_triggered(signal_generator_instance, processed_data_bundle_factory, mock_config_manager_sg):
    if not SCHEMAS_LOADED: pytest.skip("Schemas not loaded.")
    mspi_thresh = mock_config_manager_sg.get_setting("strategy_settings", "thresholds", "mspi_bullish_confirmation_min_val")
    sai_thresh = mock_config_manager_sg.get_setting("strategy_settings", "thresholds", "sai_high_conviction")
    signal_name_cfg = mock_config_manager_sg.get_setting("strategy_settings", "signal_config", "mspi_sai_bullish", "name")
    strike_metrics = [{"strike": 100.0, "a_mspi_und_summary_score": mspi_thresh + 0.1, "a_sai_und_avg": sai_thresh + 0.1}]
    bundle = processed_data_bundle_factory(strike_metrics_list=strike_metrics, current_market_regime=MarketRegimeV2_5.REGIME_NEUTRAL_CONGESTION)
    signals = signal_generator_instance.generate_all_signals(bundle)
    assert 'directional' in signals and len(signals['directional']) >= 1
    found_signal = next((s for s in signals['directional'] if s.signal_name == signal_name_cfg), None)
    assert found_signal is not None
    assert found_signal.direction == "Bullish"

def test_directional_mspi_sai_bullish_not_triggered_below_threshold(signal_generator_instance, processed_data_bundle_factory, mock_config_manager_sg):
    if not SCHEMAS_LOADED: pytest.skip("Schemas not loaded.")
    mspi_thresh = mock_config_manager_sg.get_setting("strategy_settings", "thresholds", "mspi_bullish_confirmation_min_val")
    sai_thresh = mock_config_manager_sg.get_setting("strategy_settings", "thresholds", "sai_high_conviction")
    signal_name_cfg = mock_config_manager_sg.get_setting("strategy_settings", "signal_config", "mspi_sai_bullish", "name")
    strike_metrics = [{"strike": 100.0, "a_mspi_und_summary_score": mspi_thresh - 0.1, "a_sai_und_avg": sai_thresh + 0.1}]
    bundle = processed_data_bundle_factory(strike_metrics_list=strike_metrics)
    signals = signal_generator_instance.generate_all_signals(bundle)
    assert not any(s.signal_name == signal_name_cfg for s in signals.get('directional', []))

def test_directional_mspi_sai_bearish_triggered(signal_generator_instance, processed_data_bundle_factory, mock_config_manager_sg):
    if not SCHEMAS_LOADED: pytest.skip("Schemas not loaded.")
    mspi_thresh_bearish = mock_config_manager_sg.get_setting("strategy_settings", "thresholds", "mspi_bearish_confirmation_min_val")
    sai_thresh = mock_config_manager_sg.get_setting("strategy_settings", "thresholds", "sai_high_conviction") 
    signal_name_cfg = mock_config_manager_sg.get_setting("strategy_settings", "signal_config", "mspi_sai_bearish", "name")
    strike_metrics = [{"strike": 100.0, "a_mspi_und_summary_score": mspi_thresh_bearish - 0.1, "a_sai_und_avg": -(sai_thresh + 0.1)}]
    bundle = processed_data_bundle_factory(strike_metrics_list=strike_metrics)
    signals = signal_generator_instance.generate_all_signals(bundle)
    assert 'directional' in signals and len(signals['directional']) >= 1
    assert any(s.signal_name == signal_name_cfg for s in signals['directional'])

def test_directional_mspi_sai_regime_influence_basic(signal_generator_instance, processed_data_bundle_factory, mock_config_manager_sg):
    if not SCHEMAS_LOADED: pytest.skip("Schemas not loaded.")
    mspi_thresh = mock_config_manager_sg.get_setting("strategy_settings", "thresholds", "mspi_bullish_confirmation_min_val")
    sai_thresh = mock_config_manager_sg.get_setting("strategy_settings", "thresholds", "sai_high_conviction")
    signal_name_cfg = mock_config_manager_sg.get_setting("strategy_settings", "signal_config", "mspi_sai_bullish", "name")
    strike_metrics = [{"strike": 100.0, "a_mspi_und_summary_score": mspi_thresh + 0.1, "a_sai_und_avg": sai_thresh + 0.1}]
    bundle = processed_data_bundle_factory(strike_metrics_list=strike_metrics, current_market_regime=MarketRegimeV2_5.REGIME_EXTREME_BEARISH_PANIC)
    signals = signal_generator_instance.generate_all_signals(bundle)
    assert any(s.signal_name == signal_name_cfg for s in signals.get('directional', []))

def test_vapi_fa_bullish_surge_triggered(signal_generator_instance, processed_data_bundle_factory, mock_config_manager_sg):
    if not SCHEMAS_LOADED: pytest.skip("Schemas not loaded.")
    z_score_thresh = mock_config_manager_sg.get_setting("strategy_settings", "thresholds", "vapi_fa_strong_z_score")
    signal_name_cfg = mock_config_manager_sg.get_setting("strategy_settings", "signal_config", "vapi_fa_bullish_momentum", "name")
    underlying_metrics = {"vapi_fa_z_score_und": z_score_thresh + 0.1}
    bundle = processed_data_bundle_factory(underlying_metrics=underlying_metrics)
    signals = signal_generator_instance.generate_all_signals(bundle)
    assert 'v2_5_enhanced' in signals and len(signals['v2_5_enhanced']) >=1
    assert any(s.signal_name == signal_name_cfg for s in signals['v2_5_enhanced'])

def test_vapi_fa_bearish_surge_triggered(signal_generator_instance, processed_data_bundle_factory, mock_config_manager_sg):
    if not SCHEMAS_LOADED: pytest.skip("Schemas not loaded.")
    z_score_thresh = mock_config_manager_sg.get_setting("strategy_settings", "thresholds", "vapi_fa_strong_z_score")
    signal_name_cfg = mock_config_manager_sg.get_setting("strategy_settings", "signal_config", "vapi_fa_bearish_momentum", "name")
    underlying_metrics = {"vapi_fa_z_score_und": -(z_score_thresh + 0.1)}
    bundle = processed_data_bundle_factory(underlying_metrics=underlying_metrics)
    signals = signal_generator_instance.generate_all_signals(bundle)
    assert 'v2_5_enhanced' in signals and len(signals['v2_5_enhanced']) >=1
    assert any(s.signal_name == signal_name_cfg for s in signals['v2_5_enhanced'])

def test_vapi_fa_surge_not_triggered_within_threshold(signal_generator_instance, processed_data_bundle_factory, mock_config_manager_sg):
    if not SCHEMAS_LOADED: pytest.skip("Schemas not loaded.")
    z_score_thresh = mock_config_manager_sg.get_setting("strategy_settings", "thresholds", "vapi_fa_strong_z_score")
    bull_signal_name = mock_config_manager_sg.get_setting("strategy_settings", "signal_config", "vapi_fa_bullish_momentum", "name")
    bear_signal_name = mock_config_manager_sg.get_setting("strategy_settings", "signal_config", "vapi_fa_bearish_momentum", "name")
    underlying_metrics = {"vapi_fa_z_score_und": z_score_thresh - 0.1} 
    bundle = processed_data_bundle_factory(underlying_metrics=underlying_metrics)
    signals = signal_generator_instance.generate_all_signals(bundle)
    assert not any(s.signal_name == bull_signal_name for s in signals.get('v2_5_enhanced', []))
    assert not any(s.signal_name == bear_signal_name for s in signals.get('v2_5_enhanced', []))

def test_signal_activation_flag_prevents_generation(mock_config_manager_sg, processed_data_bundle_factory): 
    if not SCHEMAS_LOADED: pytest.skip("Schemas not loaded.")
    mock_config_manager_sg.update_setting(("system_settings", "signal_activation", "directional_signals"), False)
    sg_instance_no_directional = SignalGeneratorV2_5(config_manager=mock_config_manager_sg) 
    mspi_thresh = mock_config_manager_sg.get_setting("strategy_settings", "thresholds", "mspi_bullish_confirmation_min_val")
    sai_thresh = mock_config_manager_sg.get_setting("strategy_settings", "thresholds", "sai_high_conviction")
    strike_metrics = [{"strike": 100.0, "a_mspi_und_summary_score": mspi_thresh + 0.1, "a_sai_und_avg": sai_thresh + 0.1}]
    bundle = processed_data_bundle_factory(strike_metrics_list=strike_metrics)
    signals = sg_instance_no_directional.generate_all_signals(bundle)
    assert 'directional' not in signals or not signals.get('directional', []) 
    mock_config_manager_sg.update_setting(("system_settings", "signal_activation", "directional_signals"), True)


# --- Phase 2 Tests ---

# Adaptive SDAG Conviction Signals
def test_sdag_conviction_bullish_triggered(signal_generator_instance, processed_data_bundle_factory, mock_config_manager_sg):
    if not SCHEMAS_LOADED: pytest.skip("Schemas not loaded.")
    
    sdag_mult_thresh = mock_config_manager_sg.get_setting("strategy_settings", "thresholds", "sdag_mult_bullish_threshold")
    sdag_dir_thresh = mock_config_manager_sg.get_setting("strategy_settings", "thresholds", "sdag_dir_bullish_threshold")
    signal_name_cfg = mock_config_manager_sg.get_setting("strategy_settings", "signal_config", "sdag_conviction_bullish", "name")

    strike_metrics = [{
        "strike": 100.0, "oi": 100, 
        "e_sdag_mult_strike": sdag_mult_thresh + 0.1, 
        "e_sdag_dir_strike": sdag_dir_thresh + 0.1,   
    }]
    bundle = processed_data_bundle_factory(strike_metrics_list=strike_metrics)
    signals = signal_generator_instance.generate_all_signals(bundle)

    assert 'directional' in signals, "Directional signals category not found."
    found_signal = next((s for s in signals['directional'] if s.signal_name == signal_name_cfg and s.strike_impacted == 100.0), None)
    assert found_signal is not None, f"{signal_name_cfg} for strike 100.0 not found."
    assert found_signal.direction == "Bullish"

def test_sdag_conviction_bearish_triggered(signal_generator_instance, processed_data_bundle_factory, mock_config_manager_sg):
    if not SCHEMAS_LOADED: pytest.skip("Schemas not loaded.")
    
    sdag_mult_thresh = mock_config_manager_sg.get_setting("strategy_settings", "thresholds", "sdag_mult_bearish_threshold") 
    sdag_dir_thresh = mock_config_manager_sg.get_setting("strategy_settings", "thresholds", "sdag_dir_bearish_threshold") 
    signal_name_cfg = mock_config_manager_sg.get_setting("strategy_settings", "signal_config", "sdag_conviction_bearish", "name")

    strike_metrics = [{
        "strike": 100.0, "oi": 100,
        "e_sdag_mult_strike": sdag_mult_thresh - 0.1, 
        "e_sdag_dir_strike": sdag_dir_thresh - 0.1,   
    }]
    bundle = processed_data_bundle_factory(strike_metrics_list=strike_metrics)
    signals = signal_generator_instance.generate_all_signals(bundle)

    assert 'directional' in signals, "Directional signals category not found."
    found_signal = next((s for s in signals['directional'] if s.signal_name == signal_name_cfg and s.strike_impacted == 100.0), None)
    assert found_signal is not None, f"{signal_name_cfg} for strike 100.0 not found."
    assert found_signal.direction == "Bearish"

def test_sdag_conviction_not_triggered_insufficient_alignment(signal_generator_instance, processed_data_bundle_factory, mock_config_manager_sg):
    if not SCHEMAS_LOADED: pytest.skip("Schemas not loaded.")
    
    sdag_mult_bull_thresh = mock_config_manager_sg.get_setting("strategy_settings", "thresholds", "sdag_mult_bullish_threshold")
    sdag_dir_bull_thresh = mock_config_manager_sg.get_setting("strategy_settings", "thresholds", "sdag_dir_bullish_threshold")
    bull_signal_name_cfg = mock_config_manager_sg.get_setting("strategy_settings", "signal_config", "sdag_conviction_bullish", "name")
    
    strike_metrics_1 = [{"strike": 100.0, "oi":100, "e_sdag_mult_strike": sdag_mult_bull_thresh + 0.1, "e_sdag_dir_strike": sdag_dir_bull_thresh - 0.1}] 
    bundle_1 = processed_data_bundle_factory(strike_metrics_list=strike_metrics_1)
    signals_1 = signal_generator_instance.generate_all_signals(bundle_1)
    assert not any(s.signal_name == bull_signal_name_cfg for s in signals_1.get('directional', []))
    
    strike_metrics_2 = [{"strike": 100.0, "oi":100, "e_sdag_mult_strike": sdag_mult_bull_thresh -0.1, "e_sdag_dir_strike": sdag_dir_bull_thresh + 0.1}] 
    bundle_2 = processed_data_bundle_factory(strike_metrics_list=strike_metrics_2)
    signals_2 = signal_generator_instance.generate_all_signals(bundle_2)
    assert not any(s.signal_name == bull_signal_name_cfg for s in signals_2.get('directional', []))

# Volatility Regime Signals
def test_vol_expansion_0dte_triggered(signal_generator_instance, processed_data_bundle_factory, mock_config_manager_sg):
    if not SCHEMAS_LOADED: pytest.skip("Schemas not loaded.")
    signal_name_cfg = mock_config_manager_sg.get_setting("strategy_settings", "signal_config", "vol_expansion_0dte", "name")
    
    bundle = processed_data_bundle_factory(current_market_regime=MarketRegimeV2_5.REGIME_VOL_EXPANSION_IMMINENT_VRI0DTE_BULLISH, is_0dte=True)
    signals = signal_generator_instance.generate_all_signals(bundle)
    
    assert 'volatility' in signals
    assert any(s.signal_name == signal_name_cfg for s in signals['volatility']), f"{signal_name_cfg} not found."

def test_vol_expansion_vri_2_0_driven_triggered(signal_generator_instance, processed_data_bundle_factory, mock_config_manager_sg):
    if not SCHEMAS_LOADED: pytest.skip("Schemas not loaded.")
    vri_exp_thresh = mock_config_manager_sg.get_setting("strategy_settings", "thresholds", "vri_expansion_threshold")
    signal_name_cfg = mock_config_manager_sg.get_setting("strategy_settings", "signal_config", "vol_expansion_vri2", "name")
    
    underlying_metrics = {"vri_2_0_und_aggregate": vri_exp_thresh + 0.1}
    bundle = processed_data_bundle_factory(underlying_metrics=underlying_metrics, current_market_regime=MarketRegimeV2_5.REGIME_NEUTRAL_CONGESTION)
    signals = signal_generator_instance.generate_all_signals(bundle)
    
    assert 'volatility' in signals
    assert any(s.signal_name == signal_name_cfg for s in signals['volatility']), f"{signal_name_cfg} not found."

def test_vol_contraction_triggered(signal_generator_instance, processed_data_bundle_factory, mock_config_manager_sg):
    if not SCHEMAS_LOADED: pytest.skip("Schemas not loaded.")
    vri_contr_thresh = mock_config_manager_sg.get_setting("strategy_settings", "thresholds", "vri_contraction_threshold")
    signal_name_cfg = mock_config_manager_sg.get_setting("strategy_settings", "signal_config", "vol_contraction", "name")

    underlying_metrics = {
        "vri_2_0_und_aggregate": vri_contr_thresh - 0.1,
        "a_ssi_und_avg": 0.6 
    }
    bundle = processed_data_bundle_factory(underlying_metrics=underlying_metrics, current_market_regime=MarketRegimeV2_5.REGIME_VOL_CONTRACTION_NEUTRAL)
    signals = signal_generator_instance.generate_all_signals(bundle)
    
    assert 'volatility' in signals
    assert any(s.signal_name == signal_name_cfg for s in signals['volatility']), f"{signal_name_cfg} not found."

# Enhanced Time Decay Signals
def test_pin_risk_0dte_triggered(signal_generator_instance, processed_data_bundle_factory, mock_config_manager_sg):
    if not SCHEMAS_LOADED: pytest.skip("Schemas not loaded.")
    pin_risk_thresh = mock_config_manager_sg.get_setting("strategy_settings", "thresholds", "d_tdpi_pin_risk_min_abs_val_0dte")
    signal_name_cfg = mock_config_manager_sg.get_setting("strategy_settings", "signal_config", "pin_risk_0dte", "name")

    strike_metrics = [{"strike": 100.0, "d_tdpi_strike_0dte": pin_risk_thresh + 0.1, "oi":50}] 
    underlying_metrics = {"vci_0dte_agg": 0.8, "price": 100.0} 
    
    bundle = processed_data_bundle_factory(
        strike_metrics_list=strike_metrics, 
        underlying_metrics=underlying_metrics,
        is_0dte=True 
    )
    signals = signal_generator_instance.generate_all_signals(bundle)
    
    assert 'time_decay' in signals
    found_signal = next((s for s in signals['time_decay'] if s.signal_name == signal_name_cfg and s.strike_impacted == 100.0), None)
    assert found_signal is not None, f"{signal_name_cfg} for strike 100.0 not found."

def test_charm_cascade_triggered(signal_generator_instance, processed_data_bundle_factory, mock_config_manager_sg):
    if not SCHEMAS_LOADED: pytest.skip("Schemas not loaded.")
    ctr_thresh = mock_config_manager_sg.get_setting("strategy_settings", "thresholds", "e_ctr_strong_threshold")
    tdfi_thresh = mock_config_manager_sg.get_setting("strategy_settings", "thresholds", "e_tdfi_strong_threshold")
    signal_name_cfg = mock_config_manager_sg.get_setting("strategy_settings", "signal_config", "charm_cascade_alert", "name")
    
    strike_metrics = [{
        "strike": 100.0, "oi":100,
        "e_ctr_strike": ctr_thresh + 0.1, 
        "e_tdfi_strike": tdfi_thresh + 0.1 
    }] 
    bundle = processed_data_bundle_factory(
        strike_metrics_list=strike_metrics,
        current_market_regime=MarketRegimeV2_5.REGIME_CASCADE_RISK_CHARM if SCHEMAS_LOADED else "REGIME_CASCADE_RISK_CHARM"
    )
    signals = signal_generator_instance.generate_all_signals(bundle)
    assert 'time_decay' in signals
    assert any(s.signal_name == signal_name_cfg for s in signals['time_decay'])

# Predictive Complex Signals
def test_structure_change_low_assi_triggered(signal_generator_instance, processed_data_bundle_factory, mock_config_manager_sg):
    if not SCHEMAS_LOADED: pytest.skip("Schemas not loaded.")
    assi_low_thresh = mock_config_manager_sg.get_setting("strategy_settings", "thresholds", "a_ssi_low_threshold")
    signal_name_cfg = mock_config_manager_sg.get_setting("strategy_settings", "signal_config", "structure_change_alert_low_ssi", "name")

    underlying_metrics = {
        "a_ssi_und_avg": assi_low_thresh - 0.1,
        "dwfd_z_score_und": -0.5 
    }
    bundle = processed_data_bundle_factory(underlying_metrics=underlying_metrics)
    signals = signal_generator_instance.generate_all_signals(bundle)
    
    assert 'complex_signals' in signals
    assert any(s.signal_name == signal_name_cfg for s in signals['complex_signals'])

def test_flow_divergence_arfi_triggered(signal_generator_instance, processed_data_bundle_factory, mock_config_manager_sg):
    if not SCHEMAS_LOADED: pytest.skip("Schemas not loaded.")
    arfi_div_thresh = mock_config_manager_sg.get_setting("strategy_settings", "thresholds", "arfi_divergence_threshold")
    signal_name_cfg = mock_config_manager_sg.get_setting("strategy_settings", "signal_config", "flow_divergence_warning", "name")
    
    underlying_metrics = {
        "arfi_overall_und_avg": arfi_div_thresh + 0.2, 
        "dwfd_z_score_und": -0.5, 
    }
    bundle = processed_data_bundle_factory(underlying_metrics=underlying_metrics, current_market_regime=MarketRegimeV2_5.REGIME_NEUTRAL_CONGESTION)
    signals = signal_generator_instance.generate_all_signals(bundle)

    assert 'complex_signals' in signals
    assert any(s.signal_name == signal_name_cfg for s in signals['complex_signals'])

# Other New v2.5 Signals (DWFD, TW-LAF based)
def test_dwfd_smart_money_bullish_triggered(signal_generator_instance, processed_data_bundle_factory, mock_config_manager_sg):
    if not SCHEMAS_LOADED: pytest.skip("Schemas not loaded.")
    dwfd_thresh = mock_config_manager_sg.get_setting("strategy_settings", "thresholds", "dwfd_strong_z_score")
    signal_name_cfg = mock_config_manager_sg.get_setting("strategy_settings", "signal_config", "dwfd_smart_money_bullish", "name")
    
    underlying_metrics = {"dwfd_z_score_und": dwfd_thresh + 0.1}
    bundle = processed_data_bundle_factory(underlying_metrics=underlying_metrics)
    signals = signal_generator_instance.generate_all_signals(bundle)
    
    assert 'v2_5_enhanced' in signals
    assert any(s.signal_name == signal_name_cfg for s in signals['v2_5_enhanced'])

def test_dwfd_smart_money_bearish_triggered(signal_generator_instance, processed_data_bundle_factory, mock_config_manager_sg):
    if not SCHEMAS_LOADED: pytest.skip("Schemas not loaded.")
    dwfd_thresh = mock_config_manager_sg.get_setting("strategy_settings", "thresholds", "dwfd_strong_z_score") 
    signal_name_cfg = mock_config_manager_sg.get_setting("strategy_settings", "signal_config", "dwfd_smart_money_bearish", "name")
    
    underlying_metrics = {"dwfd_z_score_und": -(dwfd_thresh + 0.1)}
    bundle = processed_data_bundle_factory(underlying_metrics=underlying_metrics)
    signals = signal_generator_instance.generate_all_signals(bundle)
    
    assert 'v2_5_enhanced' in signals
    assert any(s.signal_name == signal_name_cfg for s in signals['v2_5_enhanced'])

def test_tw_laf_trend_confirmation_bullish_triggered(signal_generator_instance, processed_data_bundle_factory, mock_config_manager_sg):
    if not SCHEMAS_LOADED: pytest.skip("Schemas not loaded.")
    tw_laf_thresh = mock_config_manager_sg.get_setting("strategy_settings", "thresholds", "tw_laf_strong_z_score")
    signal_name_cfg = mock_config_manager_sg.get_setting("strategy_settings", "signal_config", "tw_laf_bullish_trend_confirm", "name")
    
    underlying_metrics = {"tw_laf_z_score_und": tw_laf_thresh + 0.1}
    bundle = processed_data_bundle_factory(underlying_metrics=underlying_metrics, current_market_regime=MarketRegimeV2_5.REGIME_BULLISH_TREND)
    signals = signal_generator_instance.generate_all_signals(bundle)
    
    assert 'v2_5_enhanced' in signals
    assert any(s.signal_name == signal_name_cfg for s in signals['v2_5_enhanced'])

def test_tw_laf_trend_confirmation_bearish_triggered(signal_generator_instance, processed_data_bundle_factory, mock_config_manager_sg):
    if not SCHEMAS_LOADED: pytest.skip("Schemas not loaded.")
    tw_laf_thresh = mock_config_manager_sg.get_setting("strategy_settings", "thresholds", "tw_laf_strong_z_score") 
    signal_name_cfg = mock_config_manager_sg.get_setting("strategy_settings", "signal_config", "tw_laf_bearish_trend_confirm", "name")
    
    underlying_metrics = {"tw_laf_z_score_und": -(tw_laf_thresh + 0.1)}
    bundle = processed_data_bundle_factory(underlying_metrics=underlying_metrics, current_market_regime=MarketRegimeV2_5.REGIME_BEARISH_TREND)
    signals = signal_generator_instance.generate_all_signals(bundle)
    
    assert 'v2_5_enhanced' in signals
    assert any(s.signal_name == signal_name_cfg for s in signals['v2_5_enhanced'])

# Evolved v2.4 Signals
def test_vanna_cascade_alert_triggered(signal_generator_instance, processed_data_bundle_factory, mock_config_manager_sg):
    if not SCHEMAS_LOADED: pytest.skip("Schemas not loaded.")
    signal_name_cfg = mock_config_manager_sg.get_setting("strategy_settings", "signal_config", "vanna_cascade_alert_bullish", "name") # Test bullish case
    
    bundle = processed_data_bundle_factory(current_market_regime=MarketRegimeV2_5.REGIME_VANNA_CASCADE_ALERT_BULLISH)
    signals = signal_generator_instance.generate_all_signals(bundle)
    
    # This signal type might fall under 'complex_signals' or 'volatility' based on its nature
    assert 'complex_signals' in signals or 'volatility' in signals
    signal_found = False
    if signals.get('complex_signals') and any(s.signal_name == signal_name_cfg for s in signals['complex_signals']):
        signal_found = True
    if not signal_found and signals.get('volatility') and any(s.signal_name == signal_name_cfg for s in signals['volatility']):
        signal_found = True
    assert signal_found, f"{signal_name_cfg} not found."

def test_eod_hedging_flow_imminent_buying_triggered(signal_generator_instance, processed_data_bundle_factory, mock_config_manager_sg):
    if not SCHEMAS_LOADED: pytest.skip("Schemas not loaded.")
    signal_name_cfg = mock_config_manager_sg.get_setting("strategy_settings", "signal_config", "eod_hedging_flow_imminent_buying", "name")
    
    bundle = processed_data_bundle_factory(current_market_regime=MarketRegimeV2_5.REGIME_EOD_HEDGING_PRESSURE_BUY)
    signals = signal_generator_instance.generate_all_signals(bundle)
    
    # This signal type might fall under 'time_decay' or 'complex_signals'
    signal_found = False
    if signals.get('time_decay') and any(s.signal_name == signal_name_cfg for s in signals['time_decay']):
        signal_found = True
    if not signal_found and signals.get('complex_signals') and any(s.signal_name == signal_name_cfg for s in signals['complex_signals']):
        signal_found = True
    assert signal_found, f"{signal_name_cfg} not found."

def test_vol_skew_shift_alert_triggered(signal_generator_instance, processed_data_bundle_factory, mock_config_manager_sg):
    if not SCHEMAS_LOADED: pytest.skip("Schemas not loaded.")
    skew_thresh = mock_config_manager_sg.get_setting("strategy_settings", "thresholds", "vol_skew_shift_min_abs_skewfactor_global")
    signal_name_cfg = mock_config_manager_sg.get_setting("strategy_settings", "signal_config", "vol_skew_shift_alert", "name")

    underlying_metrics = {"skewfactor_global": skew_thresh + 0.1} # Positive shift
    bundle = processed_data_bundle_factory(underlying_metrics=underlying_metrics)
    signals = signal_generator_instance.generate_all_signals(bundle)
    assert 'volatility' in signals
    assert any(s.signal_name == signal_name_cfg for s in signals['volatility'])

    underlying_metrics_neg = {"skewfactor_global": -(skew_thresh + 0.1)} # Negative shift
    bundle_neg = processed_data_bundle_factory(underlying_metrics=underlying_metrics_neg)
    signals_neg = signal_generator_instance.generate_all_signals(bundle_neg)
    assert 'volatility' in signals_neg
    assert any(s.signal_name == signal_name_cfg for s in signals_neg['volatility'])


def test_bubble_mispricing_warning_triggered(signal_generator_instance, processed_data_bundle_factory, mock_config_manager_sg):
    if not SCHEMAS_LOADED: pytest.skip("Schemas not loaded.")
    signal_name_cfg = mock_config_manager_sg.get_setting("strategy_settings", "signal_config", "bubble_mispricing_warning", "name")
    
    # Mock a confluence of conditions. Exact thresholds depend on SG's internal logic for "price extension vs MSPI"
    # This test assumes the REGIME_TREND_EXHAUSTION_RISK_TOP is a key trigger if SG directly uses it.
    # Or, it checks a combination of metrics that would lead to this conclusion.
    underlying_metrics = {
        "a_mspi_und_summary_score": 0.5, # Example: Moderately bullish MSPI
        "arfi_overall_und_avg": -0.6,   # Example: Bearish ARFI (divergence with price if price is up)
        "dwfd_z_score_und": -1.5,       # Example: Bearish DWFD
        "tw_laf_z_score_und": -1.0        # Example: Bearish TW-LAF
    }
    bundle = processed_data_bundle_factory(
        underlying_metrics=underlying_metrics, 
        current_market_regime=MarketRegimeV2_5.REGIME_TREND_EXHAUSTION_RISK_TOP if SCHEMAS_LOADED else "REGIME_TREND_EXHAUSTION_RISK_TOP"
    )
    signals = signal_generator_instance.generate_all_signals(bundle)
    
    assert 'complex_signals' in signals
    assert any(s.signal_name == signal_name_cfg for s in signals['complex_signals'])

# --- End of file ---

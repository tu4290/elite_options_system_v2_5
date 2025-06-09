# tests/test_metrics_calculator_v2_5.py
# EOTS v2.5 - SENTRY-APPROVED CANONICAL SCRIPT
#
# This file contains comprehensive unit tests for the refactored
# MetricsCalculatorV2_5, ensuring its components function correctly
# and handle edge cases gracefully.

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from scipy.stats import norm as scipy_norm 


# Attempt to import schemas, handling potential ImportError for local testing vs. package structure
try:
    from elite_options_system_v2_5.eots_schemas_v2_5 import RawOptionsContractDataV2_5, RawUnderlyingDataCombinedV2_5, UnderlyingMarketDataApiV2_5, UnderlyingVolSurfaceApiV2_5, UnderlyingTDPIApiV2_5
    from elite_options_system_v2_5.eots_data_bundle_v2_5 import EOTSDataBundleV2_5
except ImportError:
    EOTSDataBundleV2_5 = None 
    RawOptionsContractDataV2_5 = None
    RawUnderlyingDataCombinedV2_5 = None
    UnderlyingMarketDataApiV2_5 = None
    UnderlyingVolSurfaceApiV2_5 = None
    UnderlyingTDPIApiV2_5 = None
    pass


# --- Mock Objects and Test Fixtures ---
class MockConfigManager:
    def __init__(self):
        self.config = {
            "system_settings": {
                "metric_calculation_phases_activation": {
                    "run_metric_orchestration": True,
                    "tier1_metrics": True,
                    "tier2_metrics": True,
                    "tier3_metrics": True,
                    "heatmap_metrics": True, 
                }
            },
            "strategy_settings": {
                "strike_col_name": "strike",
                "underlying_price_col_name": "underlying_price",
                "expiration_col_name": "expiration_date",
                "dte_col_name": "dte",
                "option_type_col_name": "opt_kind",
                "open_interest_col_name": "oi",
                "volume_col_name": "volume",
                "bid_col_name": "bid",
                "ask_col_name": "ask",
                "mid_price_col_name": "mid_price",
                "delta_col_name": "delta",
                "gamma_col_name": "gamma",
                "theta_col_name": "theta",
                "vega_col_name": "vega",
                "rho_col_name": "rho",
                "charm_col_name": "charm",
                "vanna_col_name": "vanna",
                "vomma_col_name": "vomma",
                "contract_multiplier_col_name": "multiplier",
                "net_premium_col_name": "net_premium",
                "signed_volume_col_name": "signed_volume",
                "customer_gamma_flow_col_name": "gxoi", 
                "customer_delta_flow_col_name": "dxoi", 
                "customer_vega_flow_col_name": "vxoi",   
                "customer_theta_flow_col_name": "txoi",  
                "customer_charm_flow_col_name": "charmxoi",
                "customer_vanna_flow_col_name": "vannaxoi",
                "customer_vomma_flow_col_name": "vommaxoi",
                "strike_stddev_col_name": "strike_stddev_dist_from_und", 
                "price_proximity_factor_col_name": "price_prox_factor", 
            },
            "data_processor_settings": {
                "normalization_clip_percentile": 99.5,
                "coefficients": {
                    "dag_alpha": {"default": 0.5, "spy": 0.6},
                    "tdpi_beta": {"default": 1.2}
                },
                "epc_settings": {
                    "epc_smoothing_window": 5
                }
            },
            "market_regime_engine_settings": {
                "time_of_day_definitions": {
                    "eod_trigger_time": "15:45:00",
                    "market_open_time": "09:30:00",
                    "market_close_time": "16:00:00"
                },
                "eod_reference_price_field": "mid_price" 
            },
            "enhanced_flow_metric_settings": {
                "vapi_fa": {"pvr_threshold": 0.1, "min_total_volume": 100, "fa_prev_svoi_col_name": "prev_svoi", "fa_default_multiplier": 1.0, "fa_max_multiplier": 3.0, "fa_min_multiplier": 0.5},
                "dwfd": {"smoothing_window": 5, "fvvd_historical_avg_col_name": "fvvd_hist_avg"},
                "tw_laf": {"laf_time_window_minutes": 30}
            },
            "adaptive_metric_params": {
                "adaptive_smoothing_alpha": 0.18, 
                "atr_period_for_adaptive_vol": 14, 
                "a_dag_params": {
                    "base_dag_col_name": "dag_raw", 
                    "regime_alpha_multipliers": {
                        "default": 1.0, "bullish_strong": 1.2, "bullish_moderate": 1.1,
                        "bearish_strong": 0.8, "bearish_moderate": 0.9, "neutral": 1.0, 
                        "low_vol": 1.05, "high_vol": 0.95
                    },
                    "dte_sensitivity_factors": { 
                        "short_term_dte_threshold": 5, "short_term_factor": 1.1,
                        "medium_term_factor": 1.0,
                        "long_term_dte_threshold": 30, "long_term_factor": 0.9
                    },
                    "vol_context_multiplier_thresholds": { 
                        "high_vol_entry": 0.7, "high_vol_factor": 0.9, 
                        "low_vol_entry": -0.7, "low_vol_factor": 1.1,  
                        "default_factor": 1.0 
                    },
                    "min_oi_threshold_for_adaptation": 10 
                },
                "e_sdag_params": { 
                    "base_sdag_col_name": "sdag_raw", 
                    "regime_alpha_multipliers": { 
                        "default": 1.0, "bullish_strong": 1.15, "bullish_moderate": 1.05,
                        "bearish_strong": 0.85, "bearish_moderate": 0.95, "neutral": 1.0
                    },
                    "moneyness_factors": { 
                        "itm_deep": {"factor": 0.9, "delta_threshold": 0.75}, 
                        "itm": {"factor": 0.95, "delta_threshold": 0.60},      
                        "atm": {"factor": 1.1, "delta_threshold": 0.40},       
                        "otm": {"factor": 0.95, "delta_threshold": 0.20},      
                        "otm_deep": {"factor": 0.9, "delta_threshold": 0.0}   
                    },
                    "min_total_strike_oi_for_adaptation": 20 
                },
                "d_tdpi_params": {
                    "base_tdpi_col_name": "tdpi_value", 
                    "regime_adjustment_factors": { 
                        "default": 1.0, "bullish_strong": 1.2, "bearish_strong": 0.8, "neutral": 1.0
                    },
                    "momentum_context_factor_map": { 
                        "strong_up": 1.15, "moderate_up": 1.05, "neutral": 1.0, 
                        "moderate_down": 0.95, "strong_down": 0.85
                    },
                    "sentiment_context_factor_map": { 
                        "very_bullish": 1.2, "bullish": 1.1, "neutral": 1.0, 
                        "bearish": 0.9, "very_bearish": 0.8 
                    }
                },
                "vri_2_0_params": { 
                    "vri_smoothing_window": 5, 
                    "base_vxoi_call_col_name": "aggregate_call_vxoi", 
                    "base_vxoi_put_col_name": "aggregate_put_vxoi",   
                    "regime_impact_factors": { 
                         "default": 1.0, "high_vol": 1.2, "low_vol": 0.8, "neutral": 1.0
                    }
                }
            },
            "metrics_settings": {
                "common_epsilon": 1e-9,
                "arfi_epsilon": 1e-7 
            },
            "heatmap_generation_settings":{ 
                "sgdhp_params": {
                    "dxoi_col_name": "dxoi_strike", 
                    "recent_flow_confirmation_col_name": "mock_flow_conf_factor", 
                    "price_prox_factor_col_name" : "price_prox_factor",
                    "output_col_name": "SGDHP_Score" 
                },
                "ugch_greek_weights": { 
                    "gamma_weight": 0.4, "delta_weight": 0.3, "vega_weight": 0.2, "theta_weight": 0.1,
                    "gxoi_col_name": "gxoi_strike", 
                    "dxoi_col_name": "dxoi_strike", 
                    "vxoi_col_name": "vxoi_strike", 
                    "txoi_col_name": "txoi_strike", 
                    "output_col_name": "UGCH_Score" 
                },
                "ivsdh_params":{ 
                    "vanna_oi_col": "vannaxoi", # These are contract-level $OI greeks from options_df
                    "vomma_oi_col": "vommaxoi", 
                    "vega_oi_col": "vxoi",     
                    "charm_oi_col": "charmxoi",  
                    "default_dte_sens_factor": 0.5, # Example, can be a map too {<dte_bucket>:<factor>}
                    "output_pivot_table_name": "IVSDH_Surface" 
                }
            }
        }

    def get_setting(self, *keys, default=None):
        val = self.config
        try:
            for key in keys:
                val = val[key]
            return val
        except KeyError:
            if len(keys) == 1 and isinstance(keys[0], (list, tuple)):
                path = keys[0]
                val_alt = self.config
                try:
                    for k_alt in path:
                        val_alt = val_alt[k_alt]
                    return val_alt
                except KeyError:
                    return default 
            return default

    def update_setting(self, keys, value):
        d = self.config
        if not isinstance(keys, (list, tuple)):
            keys = [keys] 
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value

class MockHistoricalDataManager:
    def __init__(self, ohlc_data=None):
        self.ohlc_data = ohlc_data if ohlc_data is not None else {}

    def get_ohlc_history_for_atr(self, symbol, num_days, current_date_dt):
        current_date = current_date_dt.date() 

        if symbol in self.ohlc_data:
            df = self.ohlc_data[symbol]
            return df.head(num_days) 
        else:
            dates = pd.to_datetime([current_date - timedelta(days=i) for i in range(num_days)])
            return pd.DataFrame({
                'date': dates, 
                'high': np.full(num_days, 102.0),
                'low': np.full(num_days, 98.0),
                'close': np.full(num_days, 100.0)
            })

    def add_symbol_ohlc_data(self, symbol, df_ohlc):
        self.ohlc_data[symbol] = df_ohlc

@pytest.fixture
def mock_config_manager_v2_5():
    return MockConfigManager()

@pytest.fixture
def mock_historical_data_manager_v2_5():
    return MockHistoricalDataManager()

@pytest.fixture
def metrics_calculator_v2_5_instance(mock_config_manager_v2_5, mock_historical_data_manager_v2_5):
    from core_analytics_engine.metrics_calculator_v2_5 import MetricsCalculatorV2_5
    return MetricsCalculatorV2_5(mock_config_manager_v2_5, mock_historical_data_manager_v2_5)

@pytest.fixture
def sample_raw_options_contract_data_v2_5_factory(mock_config_manager_v2_5):
    def _factory(underlying_price=None, strike_price=100.0, dte=0, opt_type="call", oi=10, volume=5,
                 bid=None, ask=None, mid_price=None, delta=None, gamma=None, theta=None, vega=None, rho=None,
                 charm=None, vanna=None, vomma=None, net_premium=None, signed_volume=None, prev_svoi=None): 
        if RawOptionsContractDataV2_5 is None:
             pytest.skip("RawOptionsContractDataV2_5 schema not loaded.")
        _underlying_price = underlying_price if underlying_price is not None else 100.5 
        expiration = date.today() + timedelta(days=dte)
        _delta = delta if delta is not None else (0.5 if opt_type == "call" else -0.5)
        _gamma = gamma if gamma is not None else 0.1
        _theta = theta if theta is not None else -0.05
        _vega = vega if vega is not None else 0.2
        _rho = rho if rho is not None else (0.01 if opt_type == "call" else -0.01)
        _charm = charm if charm is not None else 0.005
        _vanna = vanna if vanna is not None else 0.015
        _vomma = vomma if vomma is not None else 0.007
        _bid = bid if bid is not None else (1.0 if opt_type == "call" else 0.8)
        _ask = ask if ask is not None else (1.2 if opt_type == "call" else 1.0)
        _mid_price = mid_price if mid_price is not None else (_bid + _ask) / 2
        
        _signed_volume = signed_volume if signed_volume is not None else volume
        if net_premium is None: 
            _net_premium = _signed_volume * _mid_price * mock_config_manager_v2_5.get_setting("strategy_settings", "contract_multiplier_col_name", default=100)
        else:
            _net_premium = net_premium


        contract_dict = {
            "meta_data":{"contract_id": f"TEST_{opt_type.upper()}_{strike_price}_{expiration.strftime('%Y%m%d')}_{np.random.randint(1000,9999)}", "symbol_root": "TEST"},
            "quote_unixtime_ms":int(datetime.now().timestamp() * 1000),
            "expiration_date":expiration, "strike":strike_price, "opt_kind":opt_type, "oi":oi, "volume":volume,
            "bid":_bid, "ask":_ask, "mid_price":_mid_price, "underlying_price":_underlying_price, "delta":_delta, "gamma":_gamma,
            "theta":_theta, "vega":_vega, "rho":_rho, "charm":_charm, "vanna":_vanna, "vomma":_vomma, 
            "net_premium":_net_premium, "signed_volume":_signed_volume, 
        }
        if prev_svoi is not None: 
            contract_dict['prev_svoi'] = prev_svoi 
            
        return RawOptionsContractDataV2_5(**contract_dict)
    return _factory

@pytest.fixture
def sample_options_df_raw(mock_config_manager_v2_5, sample_raw_options_contract_data_v2_5_factory): 
    if RawOptionsContractDataV2_5 is None: 
        pytest.skip("RawOptionsContractDataV2_5 schema not loaded, cannot create sample_options_df_raw.")
    contracts_data = [
        sample_raw_options_contract_data_v2_5_factory(underlying_price=100.5, strike_price=95.0, dte=0, opt_type="call", oi=10, volume=5, signed_volume=5, mid_price=1.1, delta=0.8, gamma=0.05, vega=0.1, charm=0.005, vanna=0.01, vomma=0.002, prev_svoi=450).dict(),
        sample_raw_options_contract_data_v2_5_factory(underlying_price=100.5, strike_price=105.0, dte=0, opt_type="put", oi=8, volume=10, signed_volume=-8, mid_price=0.9, delta=-0.7, gamma=0.06, vega=0.12, charm=-0.004, vanna=-0.008, vomma=0.003, prev_svoi=-700).dict(),
        sample_raw_options_contract_data_v2_5_factory(underlying_price=100.5, strike_price=105.0, dte=0, opt_type="call", oi=12, volume=7, signed_volume=7, mid_price=1.1, delta=0.3, gamma=0.07, vega=0.09, charm=0.003, vanna=0.007, vomma=0.001, prev_svoi=750).dict(),
        sample_raw_options_contract_data_v2_5_factory(underlying_price=100.5, strike_price=95.0, dte=0, opt_type="put", oi=15, volume=3, signed_volume=-2, mid_price=0.9, delta=-0.2, gamma=0.04, vega=0.07, charm=-0.002, vanna=-0.005, vomma=0.0015, prev_svoi=-350).dict(),
        sample_raw_options_contract_data_v2_5_factory(underlying_price=100.5, strike_price=100.0, dte=1, opt_type="call", oi=20, volume=15, signed_volume=10, mid_price=1.1, delta=0.55, gamma=0.09, vega=0.22, charm=0.006, vanna=0.012, vomma=0.0025, prev_svoi=1500).dict(),
        sample_raw_options_contract_data_v2_5_factory(underlying_price=100.5, strike_price=100.0, dte=1, opt_type="put", oi=18, volume=12, signed_volume=-10, mid_price=0.9, delta=-0.45, gamma=0.08, vega=0.20, charm=-0.005, vanna=-0.01, vomma=0.0022, prev_svoi=-1800).dict(),
        sample_raw_options_contract_data_v2_5_factory(underlying_price=100.5, strike_price=110.0, dte=1, opt_type="call", oi=5, volume=0, signed_volume=0, mid_price=1.1, delta=0.2, gamma=0.03, vega=0.05, charm=0.001, vanna=0.002, vomma=0.0005, prev_svoi=0).dict(),
    ]
    df = pd.DataFrame(contracts_data)
    cfg_strat = mock_config_manager_v2_5.get_setting("strategy_settings")
    cfg_enhanced_flow = mock_config_manager_v2_5.get_setting("enhanced_flow_metric_settings")

    multiplier_col_name = cfg_strat.get("contract_multiplier_col_name", "multiplier") 
    df[multiplier_col_name] = 100 
    exp_col = cfg_strat['expiration_col_name'] 
    dte_col = cfg_strat['dte_col_name'] 
    df[exp_col] = pd.to_datetime(df[exp_col]) 
    df[dte_col] = (df[exp_col].dt.date - date.today()).dt.days
    und_price_col = cfg_strat['underlying_price_col_name'] 
    oi_col = cfg_strat['open_interest_col_name'] 
    gamma_col = cfg_strat['gamma_col_name'] 
    delta_col = cfg_strat['delta_col_name'] 
    vega_col = cfg_strat['vega_col_name'] 
    theta_col = cfg_strat['theta_col_name'] 
    charm_col = cfg_strat['charm_col_name']
    vanna_col = cfg_strat['vanna_col_name']
    vomma_col = cfg_strat['vomma_col_name']
    cust_gamma_flow_col = cfg_strat['customer_gamma_flow_col_name'] 
    cust_delta_flow_col = cfg_strat['customer_delta_flow_col_name'] 
    cust_vega_flow_col = cfg_strat['customer_vega_flow_col_name']   
    cust_theta_flow_col = cfg_strat['customer_theta_flow_col_name']  
    cust_charm_flow_col = cfg_strat['customer_charm_flow_col_name']
    cust_vanna_flow_col = cfg_strat['customer_vanna_flow_col_name']
    cust_vomma_flow_col = cfg_strat['customer_vomma_flow_col_name']
    actual_multiplier_value = df[multiplier_col_name].iloc[0] 
    df[cust_gamma_flow_col] = df[gamma_col] * df[oi_col] * (df[und_price_col] ** 2) / 100 * actual_multiplier_value
    df[cust_delta_flow_col] = df[delta_col] * df[oi_col] * df[und_price_col] * actual_multiplier_value
    df[cust_vega_flow_col] = df[vega_col] * df[oi_col] * actual_multiplier_value 
    df[cust_theta_flow_col] = df[theta_col] * df[oi_col] * actual_multiplier_value
    df[cust_charm_flow_col] = df[charm_col] * df[oi_col] * actual_multiplier_value
    df[cust_vanna_flow_col] = df[vanna_col] * df[oi_col] * actual_multiplier_value
    df[cust_vomma_flow_col] = df[vomma_col] * df[oi_col] * actual_multiplier_value
    
    fvvd_hist_col = cfg_enhanced_flow.get("dwfd",{}).get("fvvd_historical_avg_col_name", "fvvd_hist_avg")
    df[fvvd_hist_col] = 0.0001 
    
    prev_svoi_col = cfg_enhanced_flow.get("vapi_fa",{}).get("fa_prev_svoi_col_name", "prev_svoi")
    if prev_svoi_col not in df.columns: 
        df[prev_svoi_col] = 0 
    
    return df

@pytest.fixture
def sample_underlying_data_api_raw(sample_options_df_raw, mock_config_manager_v2_5): 
    if RawUnderlyingDataCombinedV2_5 is None: 
         pytest.skip("RawUnderlyingDataCombinedV2_5 schema not loaded.")

    cfg_strat = mock_config_manager_v2_5.get_setting("strategy_settings")
    opt_kind_col = cfg_strat['option_type_col_name']
    oi_col = cfg_strat['open_interest_col_name']
    vol_col = cfg_strat['volume_col_name']
    und_price_col = cfg_strat['underlying_price_col_name']
    multiplier_col = cfg_strat.get("contract_multiplier_col_name", "multiplier")
    vxoi_col = cfg_strat['customer_vega_flow_col_name']


    total_call_oi_val = sample_options_df_raw[sample_options_df_raw[opt_kind_col] == 'call'][oi_col].sum() if not sample_options_df_raw.empty else 0
    total_put_oi_val = sample_options_df_raw[sample_options_df_raw[opt_kind_col] == 'put'][oi_col].sum() if not sample_options_df_raw.empty else 0
    total_call_volume_val = sample_options_df_raw[sample_options_df_raw[opt_kind_col] == 'call'][vol_col].sum() if not sample_options_df_raw.empty else 0
    total_put_volume_val = sample_options_df_raw[sample_options_df_raw[opt_kind_col] == 'put'][vol_col].sum() if not sample_options_df_raw.empty else 0
    agg_call_gxoi = sample_options_df_raw[sample_options_df_raw[opt_kind_col] == 'call'][cfg_strat['customer_gamma_flow_col_name']].sum() if not sample_options_df_raw.empty else 0
    agg_put_gxoi = sample_options_df_raw[sample_options_df_raw[opt_kind_col] == 'put'][cfg_strat['customer_gamma_flow_col_name']].sum() if not sample_options_df_raw.empty else 0
    agg_call_dxoi = sample_options_df_raw[sample_options_df_raw[opt_kind_col] == 'call'][cfg_strat['customer_delta_flow_col_name']].sum() if not sample_options_df_raw.empty else 0
    agg_put_dxoi = sample_options_df_raw[sample_options_df_raw[opt_kind_col] == 'put'][cfg_strat['customer_delta_flow_col_name']].sum() if not sample_options_df_raw.empty else 0
    
    agg_call_vxoi = sample_options_df_raw[sample_options_df_raw[opt_kind_col] == 'call'][vxoi_col].sum() if not sample_options_df_raw.empty else 0
    agg_put_vxoi = sample_options_df_raw[sample_options_df_raw[opt_kind_col] == 'put'][vxoi_col].sum() if not sample_options_df_raw.empty else 0
    
    underlying_price_val = sample_options_df_raw[und_price_col].iloc[0] if not sample_options_df_raw.empty else 100.5
    multiplier_val = sample_options_df_raw[multiplier_col].iloc[0] if not sample_options_df_raw.empty and multiplier_col in sample_options_df_raw.columns else 100
    eod_ref_price_val = 100.2 

    tdpi_val_raw = 5000000.0 
    if UnderlyingTDPIApiV2_5:
        tdpi_data_obj = UnderlyingTDPIApiV2_5(symbol="TEST", tdpi_value=tdpi_val_raw, tdpi_normalized=0.5)
    else: 
        tdpi_data_obj = {"symbol":"TEST", "tdpi_value": tdpi_val_raw, "tdpi_normalized": 0.5}


    base_data_dict = RawUnderlyingDataCombinedV2_5(
        meta_data={"symbol_root": "TEST", "data_source": "mock_api"},
        market_data=UnderlyingMarketDataApiV2_5(
            symbol="TEST", price=underlying_price_val, bid=underlying_price_val - 0.1, ask=underlying_price_val + 0.1,
            volume=1000000, quote_unixtime_ms=int(datetime.now().timestamp() * 1000)
        ) if UnderlyingMarketDataApiV2_5 else None,
        vol_surface_data=UnderlyingVolSurfaceApiV2_5(
            symbol="TEST", atm_iv=0.20, atm_call_iv=0.20, atm_put_iv=0.20, iv_skew_slope=0.05, 
            call_put_iv_spread=0.001, raw_iv_surface_data={"strikes": [90,100,110], "ivs": [0.22,0.20,0.21]}
        ) if UnderlyingVolSurfaceApiV2_5 else None,
        tdpi_data=tdpi_data_obj,
        total_call_oi=total_call_oi_val, total_put_oi=total_put_oi_val,
        total_call_volume=total_call_volume_val, total_put_volume=total_put_volume_val,
        total_oi=total_call_oi_val + total_put_oi_val, total_volume=total_call_volume_val + total_put_volume_val,
        aggregate_call_gxoi=agg_call_gxoi, aggregate_put_gxoi=agg_put_gxoi,
        aggregate_call_dxoi=agg_call_dxoi, aggregate_put_dxoi=agg_put_dxoi,
        aggregate_call_vxoi=agg_call_vxoi, 
        aggregate_put_vxoi=agg_put_vxoi,   
        atr_value=2.5, multiplier=multiplier_val
    ).dict()
    
    base_data_dict['eod_ref_price'] = eod_ref_price_val 
    base_data_dict['current_market_regime_v2_5'] = "neutral" 
    base_data_dict['vri_2_0_und_aggregate'] = 0.0 
    if base_data_dict.get('meta_data'):
        base_data_dict['meta_data']['current_market_regime_v2_5'] = "neutral"
    else:
        base_data_dict['meta_data'] = {"current_market_regime_v2_5": "neutral"}
    
    base_data_dict['st_momentum_score'] = "neutral" 
    base_data_dict['lt_sentiment_score'] = "neutral" 

    return base_data_dict

# --- Fixtures for Adaptive Metrics Tests ---

@pytest.fixture
def options_df_for_adaptive_tests(mock_config_manager_v2_5, sample_raw_options_contract_data_v2_5_factory):
    if RawOptionsContractDataV2_5 is None:
        pytest.skip("RawOptionsContractDataV2_5 schema not loaded.")
    contracts_data = [
        sample_raw_options_contract_data_v2_5_factory(dte=3, opt_type="call", strike_price=100, delta=0.5, gamma=0.10, oi=50, underlying_price=100).dict(), 
        sample_raw_options_contract_data_v2_5_factory(dte=5, opt_type="put", strike_price=95, delta=-0.3, gamma=0.08, oi=40, underlying_price=100).dict(),  
        sample_raw_options_contract_data_v2_5_factory(dte=15, opt_type="call", strike_price=105, delta=0.4, gamma=0.09, oi=60, underlying_price=100).dict(), 
        sample_raw_options_contract_data_v2_5_factory(dte=30, opt_type="put", strike_price=90, delta=-0.2, gamma=0.05, oi=30, underlying_price=100).dict(), 
        sample_raw_options_contract_data_v2_5_factory(dte=45, opt_type="call", strike_price=110, delta=0.3, gamma=0.06, oi=20, underlying_price=100).dict(), 
        sample_raw_options_contract_data_v2_5_factory(dte=10, opt_type="call", strike_price=102, delta=0.45, gamma=0.07, oi=5, underlying_price=100).dict(),
    ]
    df = pd.DataFrame(contracts_data)
    cfg_strat = mock_config_manager_v2_5.get_setting("strategy_settings")
    cfg_adapt_adag = mock_config_manager_v2_5.get_setting("adaptive_metric_params", "a_dag_params")
    multiplier_col_name = cfg_strat.get("contract_multiplier_col_name", "multiplier")
    df[multiplier_col_name] = 100 
    exp_col = cfg_strat['expiration_col_name']
    dte_col = cfg_strat['dte_col_name']
    df[exp_col] = pd.to_datetime(df['expiration_date'])
    df[dte_col] = (df[exp_col].dt.date - date.today()).dt.days 
    base_dag_col = cfg_adapt_adag.get("base_dag_col_name", "dag_raw")
    df[base_dag_col] = df[cfg_strat['gamma_col_name']] * 100 
    return df

# --- Tier 2: Adaptive Metrics Tests ---

def test_calculate_a_dag_baseline(metrics_calculator_v2_5_instance, options_df_for_adaptive_tests, sample_underlying_data_api_raw, mock_config_manager_v2_5):
    if EOTSDataBundleV2_5 is None: pytest.skip("Schemas not loaded")

    current_dt = datetime.now()
    und_data = sample_underlying_data_api_raw.copy()
    und_data['current_market_regime_v2_5'] = "neutral" 
    if und_data.get('meta_data'):
        und_data['meta_data']['current_market_regime_v2_5'] = "neutral"
    else:
        und_data['meta_data'] = {"current_market_regime_v2_5": "neutral"}
    und_data['vri_2_0_und_aggregate'] = 0.0 

    cfg_adapt_adag = mock_config_manager_v2_5.get_setting("adaptive_metric_params", "a_dag_params")
    base_dag_col = cfg_adapt_adag.get("base_dag_col_name", "dag_raw")
    adapted_dag_metric_name = "A_DAG_Und" 

    assert base_dag_col in options_df_for_adaptive_tests.columns, f"Base DAG column '{base_dag_col}' not in options_df_for_adaptive_tests."

    bundle = EOTSDataBundleV2_5(options_df_raw=options_df_for_adaptive_tests, 
                                und_data_api_raw=und_data, 
                                current_time_dt=current_dt, symbol="TESTADAGBASE")
    
    processed_bundle = metrics_calculator_v2_5_instance.calculate_all_metrics(bundle)
    a_dag_output_und = processed_bundle.und_data_enriched.get(adapted_dag_metric_name) 
    
    assert a_dag_output_und is not None, f"{adapted_dag_metric_name} not found in und_data_enriched."

    min_oi_threshold = cfg_adapt_adag.get("min_oi_threshold_for_adaptation", 0)
    cfg_strat = mock_config_manager_v2_5.get_setting("strategy_settings")
    oi_col = cfg_strat['open_interest_col_name']
    dte_col_cfg = cfg_strat['dte_col_name'] 
    
    df_calc = options_df_for_adaptive_tests.copy()
    
    regime_mult = cfg_adapt_adag['regime_alpha_multipliers'].get("neutral", 1.0) 
    vol_mult = cfg_adapt_adag['vol_context_multiplier_thresholds'].get("default_factor", 1.0)
    
    dte_params = cfg_adapt_adag['dte_sensitivity_factors']
    def get_dte_factor(dte_val):
        if dte_val <= dte_params['short_term_dte_threshold']:
            return dte_params.get('short_term_factor', 1.0)
        elif dte_val >= dte_params['long_term_dte_threshold']:
            return dte_params.get('long_term_factor', 1.0)
        return dte_params.get('medium_term_factor', 1.0)
        
    df_calc['dte_factor'] = df_calc[dte_col_cfg].apply(get_dte_factor)
    df_calc['adaptation_multiplier'] = 1.0
    eligible_mask = df_calc[oi_col] >= min_oi_threshold
    df_calc.loc[eligible_mask, 'adaptation_multiplier'] = regime_mult * vol_mult * df_calc.loc[eligible_mask, 'dte_factor']
    df_calc['adapted_dag_value_for_sum'] = df_calc[base_dag_col] * df_calc['adaptation_multiplier']
    expected_a_dag_und = df_calc['adapted_dag_value_for_sum'].sum()

    assert np.isclose(a_dag_output_und, expected_a_dag_und, atol=1e-5), \
        f"Expected baseline {adapted_dag_metric_name} {expected_a_dag_und}, got {a_dag_output_und}"

# Test A-DAG Adaptation to Market Regime
def test_calculate_a_dag_adaptive_regime(metrics_calculator_v2_5_instance, options_df_for_adaptive_tests, sample_underlying_data_api_raw, mock_config_manager_v2_5):
    if EOTSDataBundleV2_5 is None: pytest.skip("Schemas not loaded")

    current_dt = datetime.now()
    adapted_dag_metric_name = "A_DAG_Und" 
    
    und_data_neutral = sample_underlying_data_api_raw.copy()
    und_data_neutral['current_market_regime_v2_5'] = "neutral"
    if und_data_neutral.get('meta_data'): und_data_neutral['meta_data']['current_market_regime_v2_5'] = "neutral"
    else: und_data_neutral['meta_data'] = {"current_market_regime_v2_5": "neutral"}
    und_data_neutral['vri_2_0_und_aggregate'] = 0.0

    bundle_neutral = EOTSDataBundleV2_5(options_df_raw=options_df_for_adaptive_tests, 
                                        und_data_api_raw=und_data_neutral, 
                                        current_time_dt=current_dt, symbol="TESTADAGNEUTRAL")
    processed_bundle_neutral = metrics_calculator_v2_5_instance.calculate_all_metrics(bundle_neutral)
    a_dag_neutral = processed_bundle_neutral.und_data_enriched.get(adapted_dag_metric_name)
    assert a_dag_neutral is not None, f"{adapted_dag_metric_name} (neutral) not found."

    target_regime = "bullish_strong"
    und_data_bullish = sample_underlying_data_api_raw.copy()
    und_data_bullish['current_market_regime_v2_5'] = target_regime
    if und_data_bullish.get('meta_data'): und_data_bullish['meta_data']['current_market_regime_v2_5'] = target_regime
    else: und_data_bullish['meta_data'] = {"current_market_regime_v2_5": target_regime}
    und_data_bullish['vri_2_0_und_aggregate'] = 0.0 

    bundle_bullish = EOTSDataBundleV2_5(options_df_raw=options_df_for_adaptive_tests, 
                                        und_data_api_raw=und_data_bullish, 
                                        current_time_dt=current_dt, symbol="TESTADAGBULL")
    processed_bundle_bullish = metrics_calculator_v2_5_instance.calculate_all_metrics(bundle_bullish)
    a_dag_bullish = processed_bundle_bullish.und_data_enriched.get(adapted_dag_metric_name)
    assert a_dag_bullish is not None, f"{adapted_dag_metric_name} (bullish) not found."
    
    cfg_adapt_adag = mock_config_manager_v2_5.get_setting("adaptive_metric_params", "a_dag_params")
    base_dag_col = cfg_adapt_adag.get("base_dag_col_name", "dag_raw")
    min_oi_threshold = cfg_adapt_adag.get("min_oi_threshold_for_adaptation", 0)
    cfg_strat = mock_config_manager_v2_5.get_setting("strategy_settings")
    oi_col = cfg_strat['open_interest_col_name']
    dte_col_cfg = cfg_strat['dte_col_name']

    df_calc = options_df_for_adaptive_tests.copy()
    
    regime_mult_bullish = cfg_adapt_adag['regime_alpha_multipliers'].get(target_regime, 1.0)
    assert regime_mult_bullish != 1.0, "Bullish strong multiplier should not be 1.0 for a meaningful test."
    vol_mult = cfg_adapt_adag['vol_context_multiplier_thresholds'].get("default_factor", 1.0) 

    dte_params = cfg_adapt_adag['dte_sensitivity_factors']
    def get_dte_factor(dte_val):
        if dte_val <= dte_params['short_term_dte_threshold']: return dte_params.get('short_term_factor',1.0)
        elif dte_val >= dte_params['long_term_dte_threshold']: return dte_params.get('long_term_factor',1.0)
        return dte_params.get('medium_term_factor',1.0)
    df_calc['dte_factor'] = df_calc[dte_col_cfg].apply(get_dte_factor)
    
    df_calc['adaptation_multiplier'] = 1.0
    eligible_mask = df_calc[oi_col] >= min_oi_threshold
    df_calc.loc[eligible_mask, 'adaptation_multiplier'] = regime_mult_bullish * vol_mult * df_calc.loc[eligible_mask, 'dte_factor']
    
    df_calc['adapted_dag_value_for_sum'] = df_calc[base_dag_col] * df_calc['adaptation_multiplier']
    expected_a_dag_bullish = df_calc['adapted_dag_value_for_sum'].sum()

    assert np.isclose(a_dag_bullish, expected_a_dag_bullish, atol=1e-5), \
        f"Expected {target_regime} {adapted_dag_metric_name} {expected_a_dag_bullish}, got {a_dag_bullish}"
    
    if options_df_for_adaptive_tests[options_df_for_adaptive_tests[oi_col] >= min_oi_threshold].shape[0] > 0 :
        assert not np.isclose(a_dag_bullish, a_dag_neutral), \
            f"A-DAG did not change for {target_regime} regime. Neutral: {a_dag_neutral}, Bullish: {a_dag_bullish}"

# Test A-DAG Adaptation to DTE
def test_calculate_a_dag_adaptive_dte(metrics_calculator_v2_5_instance, sample_raw_options_contract_data_v2_5_factory, sample_underlying_data_api_raw, mock_config_manager_v2_5):
    if EOTSDataBundleV2_5 is None: pytest.skip("Schemas not loaded")

    cfg_strat = mock_config_manager_v2_5.get_setting("strategy_settings")
    cfg_adapt_adag = mock_config_manager_v2_5.get_setting("adaptive_metric_params", "a_dag_params")
    base_dag_col = cfg_adapt_adag.get("base_dag_col_name", "dag_raw")
    adapted_dag_metric_name = "A_DAG_Und"
    min_oi_threshold = cfg_adapt_adag.get("min_oi_threshold_for_adaptation", 1) 

    contracts_data = [
        sample_raw_options_contract_data_v2_5_factory(dte=3, opt_type="call", strike_price=100, gamma=0.10, oi=min_oi_threshold + 10).dict(), 
        sample_raw_options_contract_data_v2_5_factory(dte=45, opt_type="put", strike_price=100, gamma=0.06, oi=min_oi_threshold + 10).dict(), 
    ]
    options_df_dte_test = pd.DataFrame(contracts_data)
    options_df_dte_test[cfg_strat.get("contract_multiplier_col_name", "multiplier")] = 100
    options_df_dte_test[cfg_strat['expiration_col_name']] = pd.to_datetime(options_df_dte_test['expiration_date'])
    options_df_dte_test[cfg_strat['dte_col_name']] = (options_df_dte_test[cfg_strat['expiration_col_name']].dt.date - date.today()).dt.days
    options_df_dte_test[base_dag_col] = options_df_dte_test[cfg_strat['gamma_col_name']] * 100

    current_dt = datetime.now()
    und_data = sample_underlying_data_api_raw.copy()
    und_data['current_market_regime_v2_5'] = "neutral" 
    if und_data.get('meta_data'): und_data['meta_data']['current_market_regime_v2_5'] = "neutral"
    else: und_data['meta_data'] = {"current_market_regime_v2_5": "neutral"}
    und_data['vri_2_0_und_aggregate'] = 0.0 

    bundle = EOTSDataBundleV2_5(options_df_raw=options_df_dte_test, 
                                und_data_api_raw=und_data, 
                                current_time_dt=current_dt, symbol="TESTADAGDTE")
    
    processed_bundle = metrics_calculator_v2_5_instance.calculate_all_metrics(bundle)
    a_dag_output = processed_bundle.und_data_enriched.get(adapted_dag_metric_name)
    assert a_dag_output is not None, f"{adapted_dag_metric_name} not found for DTE test."

    df_calc = options_df_dte_test.copy()
    regime_mult = cfg_adapt_adag['regime_alpha_multipliers'].get("neutral", 1.0)
    vol_mult = cfg_adapt_adag['vol_context_multiplier_thresholds'].get("default_factor", 1.0)
    dte_params = cfg_adapt_adag['dte_sensitivity_factors']
    
    def get_dte_factor(dte_val):
        if dte_val <= dte_params['short_term_dte_threshold']: return dte_params.get('short_term_factor',1.0)
        elif dte_val >= dte_params['long_term_dte_threshold']: return dte_params.get('long_term_factor',1.0)
        return dte_params.get('medium_term_factor',1.0)
    df_calc['dte_factor'] = df_calc[cfg_strat['dte_col_name']].apply(get_dte_factor)
    
    df_calc['adaptation_multiplier'] = regime_mult * vol_mult * df_calc['dte_factor'] 
    
    df_calc['adapted_dag_value_for_sum'] = df_calc[base_dag_col] * df_calc['adaptation_multiplier']
    expected_a_dag = df_calc['adapted_dag_value_for_sum'].sum()
    
    assert np.isclose(a_dag_output, expected_a_dag, atol=1e-5), \
        f"Expected DTE-adapted A_DAG_Und {expected_a_dag}, got {a_dag_output}"

# Test A-DAG Adaptation to Volatility Context
def test_calculate_a_dag_adaptive_vol_context(metrics_calculator_v2_5_instance, options_df_for_adaptive_tests, sample_underlying_data_api_raw, mock_config_manager_v2_5):
    if EOTSDataBundleV2_5 is None: pytest.skip("Schemas not loaded")

    current_dt = datetime.now()
    adapted_dag_metric_name = "A_DAG_Und"

    und_data_neutral_vol = sample_underlying_data_api_raw.copy()
    und_data_neutral_vol['current_market_regime_v2_5'] = "neutral"
    if und_data_neutral_vol.get('meta_data'): und_data_neutral_vol['meta_data']['current_market_regime_v2_5'] = "neutral"
    else: und_data_neutral_vol['meta_data'] = {"current_market_regime_v2_5": "neutral"}
    und_data_neutral_vol['vri_2_0_und_aggregate'] = 0.0 

    bundle_neutral_vol = EOTSDataBundleV2_5(options_df_raw=options_df_for_adaptive_tests, 
                                            und_data_api_raw=und_data_neutral_vol, 
                                            current_time_dt=current_dt, symbol="TESTADAGNEUTRALVOL")
    processed_bundle_neutral_vol = metrics_calculator_v2_5_instance.calculate_all_metrics(bundle_neutral_vol)
    a_dag_neutral_vol = processed_bundle_neutral_vol.und_data_enriched.get(adapted_dag_metric_name)
    assert a_dag_neutral_vol is not None, f"{adapted_dag_metric_name} (neutral_vol) not found."

    high_vri_value = 0.8 
    und_data_high_vol = sample_underlying_data_api_raw.copy()
    und_data_high_vol['current_market_regime_v2_5'] = "neutral" 
    if und_data_high_vol.get('meta_data'): und_data_high_vol['meta_data']['current_market_regime_v2_5'] = "neutral"
    else: und_data_high_vol['meta_data'] = {"current_market_regime_v2_5": "neutral"}
    und_data_high_vol['vri_2_0_und_aggregate'] = high_vri_value 

    bundle_high_vol = EOTSDataBundleV2_5(options_df_raw=options_df_for_adaptive_tests, 
                                         und_data_api_raw=und_data_high_vol, 
                                         current_time_dt=current_dt, symbol="TESTADAGHIGHVOL")
    processed_bundle_high_vol = metrics_calculator_v2_5_instance.calculate_all_metrics(bundle_high_vol)
    a_dag_high_vol = processed_bundle_high_vol.und_data_enriched.get(adapted_dag_metric_name)
    assert a_dag_high_vol is not None, f"{adapted_dag_metric_name} (high_vol) not found."

    cfg_adapt_adag = mock_config_manager_v2_5.get_setting("adaptive_metric_params", "a_dag_params")
    base_dag_col = cfg_adapt_adag.get("base_dag_col_name", "dag_raw")
    min_oi_threshold = cfg_adapt_adag.get("min_oi_threshold_for_adaptation", 0)
    cfg_strat = mock_config_manager_v2_5.get_setting("strategy_settings")
    oi_col = cfg_strat['open_interest_col_name']
    dte_col_cfg = cfg_strat['dte_col_name']

    df_calc = options_df_for_adaptive_tests.copy()
    regime_mult = cfg_adapt_adag['regime_alpha_multipliers'].get("neutral", 1.0) 
    
    vol_params = cfg_adapt_adag['vol_context_multiplier_thresholds']
    vol_mult = vol_params.get("default_factor", 1.0) 
    if high_vri_value >= vol_params['high_vol_entry']:
        vol_mult = vol_params['high_vol_factor']
    elif high_vri_value <= vol_params['low_vol_entry']:
        vol_mult = vol_params['low_vol_factor']
    assert vol_mult == vol_params['high_vol_factor'], "High vol factor not picked up as expected."

    dte_params = cfg_adapt_adag['dte_sensitivity_factors']
    def get_dte_factor(dte_val):
        if dte_val <= dte_params['short_term_dte_threshold']: return dte_params.get('short_term_factor',1.0)
        elif dte_val >= dte_params['long_term_dte_threshold']: return dte_params.get('long_term_factor',1.0)
        return dte_params.get('medium_term_factor',1.0)
    df_calc['dte_factor'] = df_calc[dte_col_cfg].apply(get_dte_factor)
    
    df_calc['adaptation_multiplier'] = 1.0
    eligible_mask = df_calc[oi_col] >= min_oi_threshold
    df_calc.loc[eligible_mask, 'adaptation_multiplier'] = regime_mult * vol_mult * df_calc.loc[eligible_mask, 'dte_factor']
    
    df_calc['adapted_dag_value_for_sum'] = df_calc[base_dag_col] * df_calc['adaptation_multiplier']
    expected_a_dag_high_vol = df_calc['adapted_dag_value_for_sum'].sum()

    assert np.isclose(a_dag_high_vol, expected_a_dag_high_vol, atol=1e-5), \
        f"Expected high_vol {adapted_dag_metric_name} {expected_a_dag_high_vol}, got {a_dag_high_vol}"
    
    if options_df_for_adaptive_tests[options_df_for_adaptive_tests[oi_col] >= min_oi_threshold].shape[0] > 0 and vol_mult != 1.0:
        assert not np.isclose(a_dag_high_vol, a_dag_neutral_vol), \
            f"A-DAG did not change for high_vol context. Neutral_vol: {a_dag_neutral_vol}, High_vol: {a_dag_high_vol}"

# Test E-SDAG Baseline Calculation (Moneyness)
def test_calculate_e_sdag_baseline(metrics_calculator_v2_5_instance, options_df_for_adaptive_tests, sample_underlying_data_api_raw, mock_config_manager_v2_5):
    if EOTSDataBundleV2_5 is None: pytest.skip("Schemas not loaded")

    current_dt = datetime.now()
    und_data = sample_underlying_data_api_raw.copy()
    und_data['current_market_regime_v2_5'] = "neutral" 
    if und_data.get('meta_data'): und_data['meta_data']['current_market_regime_v2_5'] = "neutral"
    else: und_data['meta_data'] = {"current_market_regime_v2_5": "neutral"}
    und_data['vri_2_0_und_aggregate'] = 0.0 

    cfg_strat = mock_config_manager_v2_5.get_setting("strategy_settings")
    cfg_adapt_adag = mock_config_manager_v2_5.get_setting("adaptive_metric_params", "a_dag_params") 
    cfg_adapt_esdag = mock_config_manager_v2_5.get_setting("adaptive_metric_params", "e_sdag_params")
    
    base_dag_col = cfg_adapt_adag.get("base_dag_col_name", "dag_raw") 
    strike_col = cfg_strat['strike_col_name']
    delta_col = cfg_strat['delta_col_name']
    oi_col = cfg_strat['open_interest_col_name']
    
    adapted_sdag_metric_name = "E_SDAG_Strike" 

    bundle = EOTSDataBundleV2_5(options_df_raw=options_df_for_adaptive_tests, 
                                und_data_api_raw=und_data, 
                                current_time_dt=current_dt, symbol="TESTESDAGBASE")
    
    processed_bundle = metrics_calculator_v2_5_instance.calculate_all_metrics(bundle)
    
    assert not processed_bundle.df_strike_level_metrics.empty, "df_strike_level_metrics is empty."
    assert adapted_sdag_metric_name in processed_bundle.df_strike_level_metrics.columns, \
        f"{adapted_sdag_metric_name} not found in strike level metrics."

    df_calc = options_df_for_adaptive_tests.copy()
    
    moneyness_cfg = cfg_adapt_esdag['moneyness_factors']
    
    def get_moneyness_factor(delta_val):
        abs_delta = abs(delta_val)
        if abs_delta >= moneyness_cfg['itm_deep']['delta_threshold']: return moneyness_cfg['itm_deep']['factor']
        if abs_delta >= moneyness_cfg['itm']['delta_threshold']: return moneyness_cfg['itm']['factor']
        if abs_delta >= moneyness_cfg['atm']['delta_threshold']: return moneyness_cfg['atm']['factor']
        if abs_delta >= moneyness_cfg['otm']['delta_threshold']: return moneyness_cfg['otm']['factor']
        return moneyness_cfg['otm_deep']['factor']

    df_calc['moneyness_factor'] = df_calc[delta_col].apply(get_moneyness_factor)
    
    regime_mult = cfg_adapt_esdag['regime_alpha_multipliers'].get("neutral", 1.0)
    df_calc['adapted_contract_dag'] = df_calc[base_dag_col] * df_calc['moneyness_factor'] * regime_mult
    
    expected_e_sdag_series = df_calc.groupby(strike_col)['adapted_contract_dag'].sum().rename("expected_e_sdag")
    
    min_strike_oi = cfg_adapt_esdag['min_total_strike_oi_for_adaptation']
    strike_oi_totals = df_calc.groupby(strike_col)[oi_col].sum()
    
    for strike_val_loop, total_oi in strike_oi_totals.items(): 
        if total_oi < min_strike_oi:
            raw_sdag_at_strike = df_calc[df_calc[strike_col] == strike_val_loop][base_dag_col].sum()
            expected_e_sdag_series.loc[strike_val_loop] = raw_sdag_at_strike

    output_df = processed_bundle.df_strike_level_metrics.set_index(strike_col)
    comparison_df = pd.DataFrame(expected_e_sdag_series).join(output_df[[adapted_sdag_metric_name]], how='left')
    
    assert not comparison_df[adapted_sdag_metric_name].isnull().any(), "NaN values found in calculated E-SDAG results."

    pd.testing.assert_series_equal(
        comparison_df[adapted_sdag_metric_name].sort_index(),
        comparison_df['expected_e_sdag'].sort_index(),
        check_dtype=False, rtol=1e-4, 
        check_names=False 
    )

# Test E-SDAG Adaptation to Market Regime
def test_calculate_e_sdag_adaptive_regime(metrics_calculator_v2_5_instance, options_df_for_adaptive_tests, sample_underlying_data_api_raw, mock_config_manager_v2_5):
    if EOTSDataBundleV2_5 is None: pytest.skip("Schemas not loaded")

    current_dt = datetime.now()
    adapted_sdag_metric_name = "E_SDAG_Strike"
    cfg_strat = mock_config_manager_v2_5.get_setting("strategy_settings")
    strike_col = cfg_strat['strike_col_name']

    und_data_neutral = sample_underlying_data_api_raw.copy()
    und_data_neutral['current_market_regime_v2_5'] = "neutral"
    if und_data_neutral.get('meta_data'): und_data_neutral['meta_data']['current_market_regime_v2_5'] = "neutral"
    else: und_data_neutral['meta_data'] = {"current_market_regime_v2_5": "neutral"}
    und_data_neutral['vri_2_0_und_aggregate'] = 0.0

    bundle_neutral = EOTSDataBundleV2_5(options_df_raw=options_df_for_adaptive_tests, 
                                        und_data_api_raw=und_data_neutral, 
                                        current_time_dt=current_dt, symbol="TESTESDAGNEUTRAL")
    processed_bundle_neutral = metrics_calculator_v2_5_instance.calculate_all_metrics(bundle_neutral)
    e_sdag_neutral_series = processed_bundle_neutral.df_strike_level_metrics.set_index(strike_col)[adapted_sdag_metric_name]
    assert not e_sdag_neutral_series.empty, f"{adapted_sdag_metric_name} (neutral) not found or empty."

    target_regime = "bullish_strong"
    und_data_bullish = sample_underlying_data_api_raw.copy()
    und_data_bullish['current_market_regime_v2_5'] = target_regime
    if und_data_bullish.get('meta_data'): und_data_bullish['meta_data']['current_market_regime_v2_5'] = target_regime
    else: und_data_bullish['meta_data'] = {"current_market_regime_v2_5": target_regime}
    und_data_bullish['vri_2_0_und_aggregate'] = 0.0

    bundle_bullish = EOTSDataBundleV2_5(options_df_raw=options_df_for_adaptive_tests, 
                                        und_data_api_raw=und_data_bullish, 
                                        current_time_dt=current_dt, symbol="TESTESDAGBULL")
    processed_bundle_bullish = metrics_calculator_v2_5_instance.calculate_all_metrics(bundle_bullish)
    assert not processed_bundle_bullish.df_strike_level_metrics.empty, "df_strike_level_metrics is empty for bullish regime."
    e_sdag_bullish_series = processed_bundle_bullish.df_strike_level_metrics.set_index(strike_col)[adapted_sdag_metric_name]
    assert not e_sdag_bullish_series.empty, f"{adapted_sdag_metric_name} (bullish) not found or empty."

    cfg_adapt_adag = mock_config_manager_v2_5.get_setting("adaptive_metric_params", "a_dag_params") 
    cfg_adapt_esdag = mock_config_manager_v2_5.get_setting("adaptive_metric_params", "e_sdag_params")
    base_dag_col = cfg_adapt_adag.get("base_dag_col_name", "dag_raw")
    delta_col = cfg_strat['delta_col_name']
    oi_col = cfg_strat['open_interest_col_name']

    df_calc = options_df_for_adaptive_tests.copy()
    moneyness_cfg = cfg_adapt_esdag['moneyness_factors']
    def get_moneyness_factor(delta_val): 
        abs_delta = abs(delta_val)
        if abs_delta >= moneyness_cfg['itm_deep']['delta_threshold']: return moneyness_cfg['itm_deep']['factor']
        if abs_delta >= moneyness_cfg['itm']['delta_threshold']: return moneyness_cfg['itm']['factor']
        if abs_delta >= moneyness_cfg['atm']['delta_threshold']: return moneyness_cfg['atm']['factor']
        if abs_delta >= moneyness_cfg['otm']['delta_threshold']: return moneyness_cfg['otm']['factor']
        return moneyness_cfg['otm_deep']['factor']
    df_calc['moneyness_factor'] = df_calc[delta_col].apply(get_moneyness_factor)
    
    regime_mult_bullish = cfg_adapt_esdag['regime_alpha_multipliers'].get(target_regime, 1.0)
    assert regime_mult_bullish != 1.0, "Bullish strong multiplier for E-SDAG should not be 1.0 for a meaningful test."
    
    df_calc['adapted_contract_dag'] = df_calc[base_dag_col] * df_calc['moneyness_factor'] * regime_mult_bullish
    
    expected_e_sdag_bullish_series = df_calc.groupby(strike_col)['adapted_contract_dag'].sum().rename("expected_e_sdag_bullish")
    
    min_strike_oi = cfg_adapt_esdag['min_total_strike_oi_for_adaptation']
    strike_oi_totals = df_calc.groupby(strike_col)[oi_col].sum()
    
    raw_sdag_sum_by_strike = df_calc.groupby(strike_col)[base_dag_col].sum()
    for strike_val_loop, total_oi in strike_oi_totals.items():
        if total_oi < min_strike_oi:
            expected_e_sdag_bullish_series.loc[strike_val_loop] = raw_sdag_sum_by_strike.get(strike_val_loop, 0)
            if e_sdag_neutral_series.index.isin([strike_val_loop]).any() and strike_oi_totals.get(strike_val_loop,0) < min_strike_oi :
                 e_sdag_neutral_series.loc[strike_val_loop] = raw_sdag_sum_by_strike.get(strike_val_loop, 0)


    comparison_df = pd.DataFrame(expected_e_sdag_bullish_series).join(pd.DataFrame(e_sdag_bullish_series), how='left')
    pd.testing.assert_series_equal(
        comparison_df[adapted_sdag_metric_name].sort_index(),
        comparison_df['expected_e_sdag_bullish'].sort_index(),
        check_dtype=False, rtol=1e-4, check_names=False
    )
    
    changed_strikes = strike_oi_totals[strike_oi_totals >= min_strike_oi].index
    if not changed_strikes.empty:
        assert not np.allclose(e_sdag_bullish_series.loc[changed_strikes].fillna(0), e_sdag_neutral_series.loc[changed_strikes].fillna(0)), \
            f"E-SDAG did not change for {target_regime} regime on eligible strikes."

# --- D-TDPI Tests ---
def test_calculate_d_tdpi_baseline(metrics_calculator_v2_5_instance, sample_options_df_raw, sample_underlying_data_api_raw, mock_config_manager_v2_5):
    if EOTSDataBundleV2_5 is None: pytest.skip("Schemas not loaded")

    current_dt = datetime.now()
    und_data = sample_underlying_data_api_raw.copy() 
    und_data['current_market_regime_v2_5'] = "neutral"
    if und_data.get('meta_data'): und_data['meta_data']['current_market_regime_v2_5'] = "neutral"
    else: und_data['meta_data'] = {"current_market_regime_v2_5": "neutral"}
    und_data['st_momentum_score'] = "neutral" 
    und_data['lt_sentiment_score'] = "neutral"

    cfg_d_tdpi = mock_config_manager_v2_5.get_setting("adaptive_metric_params", "d_tdpi_params")
    base_tdpi_col = cfg_d_tdpi.get("base_tdpi_col_name") 
    
    assert 'tdpi_data' in und_data and isinstance(und_data['tdpi_data'], dict) and base_tdpi_col in und_data['tdpi_data'], \
        f"Base TDPI key '{base_tdpi_col}' not found in und_data['tdpi_data']."
    base_tdpi_value = und_data['tdpi_data'][base_tdpi_col]
    
    adapted_d_tdpi_metric_name = "D_TDPI_Und"

    bundle = EOTSDataBundleV2_5(options_df_raw=sample_options_df_raw, 
                                und_data_api_raw=und_data, 
                                current_time_dt=current_dt, symbol="TESTDTDPIBASE")
    
    processed_bundle = metrics_calculator_v2_5_instance.calculate_all_metrics(bundle)
    d_tdpi_output = processed_bundle.und_data_enriched.get(adapted_d_tdpi_metric_name)
    
    assert d_tdpi_output is not None, f"{adapted_d_tdpi_metric_name} not found in und_data_enriched."

    regime_factor = cfg_d_tdpi['regime_adjustment_factors'].get("neutral", 1.0)
    momentum_factor = cfg_d_tdpi['momentum_context_factor_map'].get("neutral", 1.0)
    sentiment_factor = cfg_d_tdpi['sentiment_context_factor_map'].get("neutral", 1.0)
    
    expected_d_tdpi = base_tdpi_value * regime_factor * momentum_factor * sentiment_factor
    
    assert np.isclose(d_tdpi_output, expected_d_tdpi, atol=1e-5), \
        f"Expected baseline {adapted_d_tdpi_metric_name} {expected_d_tdpi}, got {d_tdpi_output}"

def test_calculate_d_tdpi_adaptive_regime(metrics_calculator_v2_5_instance, sample_options_df_raw, sample_underlying_data_api_raw, mock_config_manager_v2_5):
    if EOTSDataBundleV2_5 is None: pytest.skip("Schemas not loaded")
    current_dt = datetime.now()
    d_tdpi_metric_name = "D_TDPI_Und"
    cfg_d_tdpi = mock_config_manager_v2_5.get_setting("adaptive_metric_params", "d_tdpi_params")
    base_tdpi_col = cfg_d_tdpi.get("base_tdpi_col_name")
    
    und_data_neutral = sample_underlying_data_api_raw.copy()
    base_tdpi_value = und_data_neutral['tdpi_data'][base_tdpi_col] 

    bundle_neutral = EOTSDataBundleV2_5(options_df_raw=sample_options_df_raw, und_data_api_raw=und_data_neutral, current_time_dt=current_dt, symbol="TESTDTDPINEUTRAL")
    d_tdpi_neutral = metrics_calculator_v2_5_instance.calculate_all_metrics(bundle_neutral).und_data_enriched.get(d_tdpi_metric_name)
    assert d_tdpi_neutral is not None

    target_regime = "bullish_strong"
    und_data_bullish = sample_underlying_data_api_raw.copy()
    und_data_bullish['current_market_regime_v2_5'] = target_regime
    if und_data_bullish.get('meta_data'): und_data_bullish['meta_data']['current_market_regime_v2_5'] = target_regime
    und_data_bullish['st_momentum_score'] = "neutral" 
    und_data_bullish['lt_sentiment_score'] = "neutral"

    bundle_bullish = EOTSDataBundleV2_5(options_df_raw=sample_options_df_raw, und_data_api_raw=und_data_bullish, current_time_dt=current_dt, symbol="TESTDTDPIRREG")
    d_tdpi_bullish = metrics_calculator_v2_5_instance.calculate_all_metrics(bundle_bullish).und_data_enriched.get(d_tdpi_metric_name)
    assert d_tdpi_bullish is not None

    regime_factor_bullish = cfg_d_tdpi['regime_adjustment_factors'].get(target_regime, 1.0)
    momentum_factor = cfg_d_tdpi['momentum_context_factor_map'].get("neutral", 1.0)
    sentiment_factor = cfg_d_tdpi['sentiment_context_factor_map'].get("neutral", 1.0)
    expected_d_tdpi_bullish = base_tdpi_value * regime_factor_bullish * momentum_factor * sentiment_factor
    
    assert np.isclose(d_tdpi_bullish, expected_d_tdpi_bullish, atol=1e-5)
    if regime_factor_bullish != 1.0 :
         assert not np.isclose(d_tdpi_bullish, d_tdpi_neutral)

def test_calculate_d_tdpi_adaptive_momentum(metrics_calculator_v2_5_instance, sample_options_df_raw, sample_underlying_data_api_raw, mock_config_manager_v2_5):
    if EOTSDataBundleV2_5 is None: pytest.skip("Schemas not loaded")
    current_dt = datetime.now()
    d_tdpi_metric_name = "D_TDPI_Und"
    cfg_d_tdpi = mock_config_manager_v2_5.get_setting("adaptive_metric_params", "d_tdpi_params")
    base_tdpi_col = cfg_d_tdpi.get("base_tdpi_col_name")
    
    und_data_neutral = sample_underlying_data_api_raw.copy()
    base_tdpi_value = und_data_neutral['tdpi_data'][base_tdpi_col]
    bundle_neutral = EOTSDataBundleV2_5(options_df_raw=sample_options_df_raw, und_data_api_raw=und_data_neutral, current_time_dt=current_dt, symbol="TESTDTDPINEUTRALMOM")
    d_tdpi_neutral = metrics_calculator_v2_5_instance.calculate_all_metrics(bundle_neutral).und_data_enriched.get(d_tdpi_metric_name)

    target_momentum = "strong_up"
    und_data_momentum = sample_underlying_data_api_raw.copy()
    und_data_momentum['st_momentum_score'] = target_momentum

    bundle_momentum = EOTSDataBundleV2_5(options_df_raw=sample_options_df_raw, und_data_api_raw=und_data_momentum, current_time_dt=current_dt, symbol="TESTDTDPIMOM")
    d_tdpi_momentum = metrics_calculator_v2_5_instance.calculate_all_metrics(bundle_momentum).und_data_enriched.get(d_tdpi_metric_name)
    assert d_tdpi_momentum is not None

    regime_factor = cfg_d_tdpi['regime_adjustment_factors'].get("neutral", 1.0)
    momentum_factor_strong_up = cfg_d_tdpi['momentum_context_factor_map'].get(target_momentum, 1.0)
    sentiment_factor = cfg_d_tdpi['sentiment_context_factor_map'].get("neutral", 1.0)
    expected_d_tdpi_momentum = base_tdpi_value * regime_factor * momentum_factor_strong_up * sentiment_factor

    assert np.isclose(d_tdpi_momentum, expected_d_tdpi_momentum, atol=1e-5)
    if momentum_factor_strong_up != 1.0:
        assert not np.isclose(d_tdpi_momentum, d_tdpi_neutral)

def test_calculate_d_tdpi_adaptive_sentiment(metrics_calculator_v2_5_instance, sample_options_df_raw, sample_underlying_data_api_raw, mock_config_manager_v2_5):
    if EOTSDataBundleV2_5 is None: pytest.skip("Schemas not loaded")
    current_dt = datetime.now()
    d_tdpi_metric_name = "D_TDPI_Und"
    cfg_d_tdpi = mock_config_manager_v2_5.get_setting("adaptive_metric_params", "d_tdpi_params")
    base_tdpi_col = cfg_d_tdpi.get("base_tdpi_col_name")
    
    und_data_neutral = sample_underlying_data_api_raw.copy()
    base_tdpi_value = und_data_neutral['tdpi_data'][base_tdpi_col]
    bundle_neutral = EOTSDataBundleV2_5(options_df_raw=sample_options_df_raw, und_data_api_raw=und_data_neutral, current_time_dt=current_dt, symbol="TESTDTDPINEUTRALSENT")
    d_tdpi_neutral = metrics_calculator_v2_5_instance.calculate_all_metrics(bundle_neutral).und_data_enriched.get(d_tdpi_metric_name)

    target_sentiment = "very_bullish"
    und_data_sentiment = sample_underlying_data_api_raw.copy()
    und_data_sentiment['lt_sentiment_score'] = target_sentiment

    bundle_sentiment = EOTSDataBundleV2_5(options_df_raw=sample_options_df_raw, und_data_api_raw=und_data_sentiment, current_time_dt=current_dt, symbol="TESTDTDPISENT")
    d_tdpi_sentiment = metrics_calculator_v2_5_instance.calculate_all_metrics(bundle_sentiment).und_data_enriched.get(d_tdpi_metric_name)
    assert d_tdpi_sentiment is not None

    regime_factor = cfg_d_tdpi['regime_adjustment_factors'].get("neutral", 1.0)
    momentum_factor = cfg_d_tdpi['momentum_context_factor_map'].get("neutral", 1.0)
    sentiment_factor_very_bullish = cfg_d_tdpi['sentiment_context_factor_map'].get(target_sentiment, 1.0)
    expected_d_tdpi_sentiment = base_tdpi_value * regime_factor * momentum_factor * sentiment_factor_very_bullish
    
    assert np.isclose(d_tdpi_sentiment, expected_d_tdpi_sentiment, atol=1e-5)
    if sentiment_factor_very_bullish != 1.0:
        assert not np.isclose(d_tdpi_sentiment, d_tdpi_neutral)

# --- VRI 2.0 Tests ---
def test_calculate_vri_2_0_baseline(metrics_calculator_v2_5_instance, sample_options_df_raw, sample_underlying_data_api_raw, mock_config_manager_v2_5):
    if EOTSDataBundleV2_5 is None: pytest.skip("Schemas not loaded")
    current_dt = datetime.now()
    vri_2_0_metric_name = "VRI_2_0_Und" 
    
    cfg_vri_params = mock_config_manager_v2_5.get_setting("adaptive_metric_params", "vri_2_0_params")
    call_vxoi_col = cfg_vri_params.get("base_vxoi_call_col_name", "aggregate_call_vxoi")
    put_vxoi_col = cfg_vri_params.get("base_vxoi_put_col_name", "aggregate_put_vxoi")

    call_vxoi = sample_underlying_data_api_raw.get(call_vxoi_col, 0)
    put_vxoi = sample_underlying_data_api_raw.get(put_vxoi_col, 0)

    und_data = sample_underlying_data_api_raw.copy() 
    und_data['current_market_regime_v2_5'] = "neutral" 
    if und_data.get('meta_data'): und_data['meta_data']['current_market_regime_v2_5'] = "neutral"
    
    bundle = EOTSDataBundleV2_5(options_df_raw=sample_options_df_raw, 
                                und_data_api_raw=und_data, 
                                current_time_dt=current_dt, symbol="TESTVRI2BASE")
    
    processed_bundle = metrics_calculator_v2_5_instance.calculate_all_metrics(bundle)
    vri_2_0_output = processed_bundle.und_data_enriched.get(vri_2_0_metric_name)
    assert vri_2_0_output is not None, f"{vri_2_0_metric_name} not found in und_data_enriched."

    epsilon = mock_config_manager_v2_5.get_setting("metrics_settings", "common_epsilon", default=1e-9)
    expected_vri_2_0 = 0.0
    denominator = call_vxoi + put_vxoi
    if abs(denominator) > epsilon:
        expected_vri_2_0 = (call_vxoi - put_vxoi) / denominator
    elif abs(call_vxoi - put_vxoi) > epsilon : 
        expected_vri_2_0 = np.inf * np.sign(call_vxoi - put_vxoi)
    
    if np.isinf(expected_vri_2_0) and np.isinf(vri_2_0_output) and (np.sign(expected_vri_2_0) == np.sign(vri_2_0_output)):
        assert True 
    elif np.isinf(expected_vri_2_0) or np.isinf(vri_2_0_output):
        assert False, f"Infinity mismatch for VRI 2.0: Expected {expected_vri_2_0}, got {vri_2_0_output}"
    else:
        assert np.isclose(vri_2_0_output, expected_vri_2_0, atol=1e-5), \
            f"Expected baseline {vri_2_0_metric_name} {expected_vri_2_0}, got {vri_2_0_output}"

def test_calculate_vri_2_0_adaptive_regime(metrics_calculator_v2_5_instance, sample_options_df_raw, sample_underlying_data_api_raw, mock_config_manager_v2_5):
    if EOTSDataBundleV2_5 is None: pytest.skip("Schemas not loaded")
    current_dt = datetime.now()
    
    base_vri_metric_name = "VRI_2_0_Und" 
    adapted_vri_metric_name = "VRI_2_0_Regime_Adapted_Und" 
    
    cfg_vri_params = mock_config_manager_v2_5.get_setting("adaptive_metric_params", "vri_2_0_params")
    call_vxoi_col = cfg_vri_params.get("base_vxoi_call_col_name", "aggregate_call_vxoi")
    put_vxoi_col = cfg_vri_params.get("base_vxoi_put_col_name", "aggregate_put_vxoi")
    call_vxoi = sample_underlying_data_api_raw.get(call_vxoi_col, 0)
    put_vxoi = sample_underlying_data_api_raw.get(put_vxoi_col, 0)
    epsilon = mock_config_manager_v2_5.get_setting("metrics_settings", "common_epsilon", default=1e-9)
    
    base_vri_2_0_neutral = 0.0
    denominator_neutral = call_vxoi + put_vxoi
    if abs(denominator_neutral) > epsilon: base_vri_2_0_neutral = (call_vxoi - put_vxoi) / denominator_neutral
    elif abs(call_vxoi-put_vxoi) > epsilon : base_vri_2_0_neutral = np.inf * np.sign(call_vxoi-put_vxoi)

    target_regime = "high_vol" 
    und_data_regime = sample_underlying_data_api_raw.copy()
    und_data_regime['current_market_regime_v2_5'] = target_regime
    if und_data_regime.get('meta_data'): und_data_regime['meta_data']['current_market_regime_v2_5'] = target_regime
    
    bundle_regime = EOTSDataBundleV2_5(options_df_raw=sample_options_df_raw, 
                                       und_data_api_raw=und_data_regime, 
                                       current_time_dt=current_dt, symbol="TESTVRI2REG")
    
    processed_bundle_regime = metrics_calculator_v2_5_instance.calculate_all_metrics(bundle_regime)
    vri_2_0_adapted_output = processed_bundle_regime.und_data_enriched.get(adapted_vri_metric_name)
    
    if vri_2_0_adapted_output is None:
        pytest.skip(f"{adapted_vri_metric_name} not found, VRI 2.0 might not be directly adapted by regime in this calculator version.") 
        return

    regime_impact_factor = cfg_vri_params['regime_impact_factors'].get(target_regime, 1.0)
    assert regime_impact_factor != 1.0, f"Regime impact factor for '{target_regime}' should not be 1.0 for a meaningful test."
    
    expected_vri_2_0_adapted = base_vri_2_0_neutral * regime_impact_factor

    if np.isinf(expected_vri_2_0_adapted) and np.isinf(vri_2_0_adapted_output) and (np.sign(expected_vri_2_0_adapted) == np.sign(vri_2_0_adapted_output)):
        assert True
    elif np.isinf(expected_vri_2_0_adapted) or np.isinf(vri_2_0_adapted_output):
        assert False, f"Infinity mismatch for adapted VRI 2.0: Expected {expected_vri_2_0_adapted}, got {vri_2_0_adapted_output}"
    else:
        assert np.isclose(vri_2_0_adapted_output, expected_vri_2_0_adapted, atol=1e-5), \
            f"Expected adapted {adapted_vri_metric_name} ({target_regime}) {expected_vri_2_0_adapted}, got {vri_2_0_adapted_output}"


# --- Tier 3: Enhanced Rolling Flow Metrics (Spot Calculation Tests) ---

def test_calculate_vapi_fa_spot(metrics_calculator_v2_5_instance, sample_options_df_raw, sample_underlying_data_api_raw, mock_config_manager_v2_5):
    if EOTSDataBundleV2_5 is None: pytest.skip("Schemas not loaded")
    
    current_dt = datetime.now()
    cfg = mock_config_manager_v2_5.get_setting("strategy_settings")
    cfg_vapi = mock_config_manager_v2_5.get_setting("enhanced_flow_metric_settings", "vapi_fa")
    epsilon = mock_config_manager_v2_5.get_setting("metrics_settings", "common_epsilon")

    vapi_fa_metric_name = "VAPI_FA_Und"
    options_df = sample_options_df_raw 

    bundle = EOTSDataBundleV2_5(options_df_raw=options_df, 
                                und_data_api_raw=sample_underlying_data_api_raw, 
                                current_time_dt=current_dt, symbol="TESTVAPISPOT")
    
    processed_bundle = metrics_calculator_v2_5_instance.calculate_all_metrics(bundle)
    vapi_fa_output = processed_bundle.und_data_enriched.get(vapi_fa_metric_name)
    
    assert vapi_fa_output is not None, f"{vapi_fa_metric_name} not found in und_data_enriched."

    df_calc = options_df.copy()
    
    df_calc['pvr'] = 0.0
    valid_vol_mask = df_calc[cfg['volume_col_name']] > epsilon
    und_price_col_name_in_df = cfg['underlying_price_col_name'] 
    
    df_calc.loc[valid_vol_mask, 'pvr'] = \
        (df_calc.loc[valid_vol_mask, cfg['net_premium_col_name']] / df_calc.loc[valid_vol_mask, cfg['volume_col_name']]) / \
         df_calc.loc[valid_vol_mask, und_price_col_name_in_df]
    
    df_calc.loc[df_calc['pvr'] < cfg_vapi['pvr_threshold'], 'pvr'] = 0.0

    df_calc['current_svoi'] = df_calc[cfg['signed_volume_col_name']] * df_calc[cfg['open_interest_col_name']]
    prev_svoi_col = cfg_vapi.get("fa_prev_svoi_col_name", "prev_svoi") 
    
    df_calc['fa_multiplier'] = cfg_vapi.get("fa_default_multiplier", 1.0) 
    
    if prev_svoi_col in df_calc.columns:
        has_prev_svoi = df_calc[prev_svoi_col].notna()
        avg_abs_svoi = (df_calc['current_svoi'].abs() + df_calc[prev_svoi_col].abs()) / 2.0 + epsilon
        fa = (df_calc['current_svoi'] - df_calc[prev_svoi_col]) / avg_abs_svoi
        calculated_fa_mult = 1 + fa
        min_fa_mult = cfg_vapi.get("fa_min_multiplier", 0.5) 
        max_fa_mult = cfg_vapi.get("fa_max_multiplier", 3.0)
        df_calc.loc[has_prev_svoi, 'fa_multiplier'] = np.clip(calculated_fa_mult[has_prev_svoi], min_fa_mult, max_fa_mult)

    df_calc['vapi_fa_contract'] = df_calc['pvr'] * df_calc[cfg['open_interest_col_name']] * df_calc['fa_multiplier']
    expected_vapi_fa_und = df_calc['vapi_fa_contract'].sum()
    
    assert np.isclose(vapi_fa_output, expected_vapi_fa_und, atol=1e-5), \
        f"Expected VAPI_FA_Und {expected_vapi_fa_und}, got {vapi_fa_output}"

def test_calculate_dwfd_spot_components(metrics_calculator_v2_5_instance, sample_options_df_raw, sample_underlying_data_api_raw, mock_config_manager_v2_5):
    if EOTSDataBundleV2_5 is None: pytest.skip("Schemas not loaded")

    current_dt = datetime.now()
    cfg_strat = mock_config_manager_v2_5.get_setting("strategy_settings")
    cfg_dwfd = mock_config_manager_v2_5.get_setting("enhanced_flow_metric_settings", "dwfd")
    epsilon = mock_config_manager_v2_5.get_setting("metrics_settings", "common_epsilon")

    pddf_col_name = "pddf_contract" 
    fvvd_col_name = "fvvd_contract" 

    options_df = sample_options_df_raw 

    bundle = EOTSDataBundleV2_5(options_df_raw=options_df, 
                                und_data_api_raw=sample_underlying_data_api_raw, 
                                current_time_dt=current_dt, symbol="TESTDWFDSPOT")
    
    processed_bundle = metrics_calculator_v2_5_instance.calculate_all_metrics(bundle)
    results_df = processed_bundle.df_chain_with_metrics
    
    assert pddf_col_name in results_df.columns, f"{pddf_col_name} not found in df_chain_with_metrics."
    assert fvvd_col_name in results_df.columns, f"{fvvd_col_name} not found in df_chain_with_metrics."

    expected_pddf = options_df[cfg_strat['signed_volume_col_name']] * \
                    options_df[cfg_strat['delta_col_name']] * \
                    options_df[cfg_strat['contract_multiplier_col_name']]
    pd.testing.assert_series_equal(results_df[pddf_col_name].fillna(0), expected_pddf.fillna(0), rtol=1e-5, check_names=False)
    
    df_calc = options_df.copy()
    fvvd_num = df_calc[cfg_strat['net_premium_col_name']]
    fvvd_den = df_calc[cfg_strat['volume_col_name']] * \
               df_calc[cfg_strat['underlying_price_col_name']] * \
               df_calc[cfg_strat['contract_multiplier_col_name']]
               
    current_ratio = pd.Series(np.zeros(len(df_calc)), index=df_calc.index)
    valid_den_mask = fvvd_den.abs() > epsilon
    current_ratio[valid_den_mask] = fvvd_num[valid_den_mask] / fvvd_den[valid_den_mask]
    
    hist_avg_col = cfg_dwfd.get("fvvd_historical_avg_col_name", "fvvd_hist_avg")
    assert hist_avg_col in df_calc.columns, f"'{hist_avg_col}' column missing in test DataFrame for FVVD hist avg."
    
    expected_fvvd = current_ratio - df_calc[hist_avg_col]
    expected_fvvd.loc[~valid_den_mask] = 0 - df_calc.loc[~valid_den_mask, hist_avg_col]


    pd.testing.assert_series_equal(results_df[fvvd_col_name].fillna(0), expected_fvvd.fillna(0), rtol=1e-5, check_names=False)

def test_calculate_laf_component(metrics_calculator_v2_5_instance, sample_options_df_raw, sample_underlying_data_api_raw, mock_config_manager_v2_5):
    if EOTSDataBundleV2_5 is None: pytest.skip("Schemas not loaded")

    current_dt = datetime.now()
    cfg_strat = mock_config_manager_v2_5.get_setting("strategy_settings")
    epsilon = mock_config_manager_v2_5.get_setting("metrics_settings", "common_epsilon")
    
    laf_col_name = "laf_contract" 

    options_df = sample_options_df_raw

    bundle = EOTSDataBundleV2_5(options_df_raw=options_df, 
                                und_data_api_raw=sample_underlying_data_api_raw, 
                                current_time_dt=current_dt, symbol="TESTLAFSPOT")
    
    processed_bundle = metrics_calculator_v2_5_instance.calculate_all_metrics(bundle)
    results_df = processed_bundle.df_chain_with_metrics

    assert laf_col_name in results_df.columns, f"{laf_col_name} not found in df_chain_with_metrics."

    df_calc = options_df.copy()
    bid_ask_spread = df_calc[cfg_strat['ask_col_name']] - df_calc[cfg_strat['bid_col_name']]
    
    liquidity_adj_factor = pd.Series(np.zeros(len(df_calc)), index=df_calc.index)
    valid_strike_mask = df_calc[cfg_strat['strike_col_name']].abs() > epsilon
    liquidity_adj_factor[valid_strike_mask] = bid_ask_spread[valid_strike_mask] / df_calc[cfg_strat['strike_col_name']][valid_strike_mask]
    
    expected_laf = df_calc[cfg_strat['signed_volume_col_name']] * liquidity_adj_factor
    
    pd.testing.assert_series_equal(results_df[laf_col_name].fillna(0), expected_laf.fillna(0), rtol=1e-5, check_names=False)


# --- Heatmap Data Component Tests ---

def test_calculate_sgdhp_scores(metrics_calculator_v2_5_instance, sample_options_df_raw, sample_underlying_data_api_raw, mock_config_manager_v2_5):
    if EOTSDataBundleV2_5 is None: pytest.skip("Schemas not loaded")

    current_dt = datetime.now()
    cfg_strat = mock_config_manager_v2_5.get_setting("strategy_settings")
    cfg_heatmap_sgdhp = mock_config_manager_v2_5.get_setting("heatmap_generation_settings", "sgdhp_params")
    epsilon = mock_config_manager_v2_5.get_setting("metrics_settings", "common_epsilon")
    
    sgdhp_score_col_name = cfg_heatmap_sgdhp.get("output_col_name", "SGDHP_Score")
    dxoi_strike_input_col = cfg_heatmap_sgdhp.get("dxoi_col_name", "dxoi_strike") 
    price_prox_input_col = cfg_heatmap_sgdhp.get("price_prox_factor_col_name", "price_prox_factor")
    
    options_df = sample_options_df_raw.copy()
    options_df.loc[options_df[cfg_strat['dte_col_name']] == 0, cfg_strat['dte_col_name']] = 0.001 

    und_px = sample_underlying_data_api_raw['market_data']['price']
    atm_iv = sample_underlying_data_api_raw['vol_surface_data']['atm_iv']

    bundle = EOTSDataBundleV2_5(options_df_raw=options_df, 
                                und_data_api_raw=sample_underlying_data_api_raw, 
                                current_time_dt=current_dt, symbol="TESTSGDHP")
    
    processed_bundle = metrics_calculator_v2_5_instance.calculate_all_metrics(bundle)
    results_strike_df = processed_bundle.df_strike_level_metrics.copy() 
    
    assert sgdhp_score_col_name in results_strike_df.columns, f"{sgdhp_score_col_name} not found in df_strike_level_metrics."
    assert dxoi_strike_input_col in results_strike_df.columns, f"Prerequisite '{dxoi_strike_input_col}' not found in strike df."
    assert price_prox_input_col in results_strike_df.columns, f"Prerequisite '{price_prox_input_col}' not found in strike df."
    
    df_for_manual_calc = results_strike_df.copy()
    
    dxoi_strike_series = df_for_manual_calc[dxoi_strike_input_col]
    norm_dxoi_strike = dxoi_strike_series / (dxoi_strike_series.abs().max() + epsilon)
    
    flow_conf_factor_series = pd.Series(1.0, index=results_strike_df.index) 
    if cfg_heatmap_sgdhp['recent_flow_confirmation_col_name'] in results_strike_df.columns:
        flow_conf_factor_series = results_strike_df[cfg_heatmap_sgdhp['recent_flow_confirmation_col_name']]
        
    expected_sgdhp = results_strike_df[price_prox_input_col] * norm_dxoi_strike * flow_conf_factor_series
    
    pd.testing.assert_series_equal(
        results_strike_df[sgdhp_score_col_name].fillna(0), 
        expected_sgdhp.fillna(0), 
        rtol=1e-4, 
        check_names=False,
        check_index_type=False 
    )

def test_calculate_ugch_scores(metrics_calculator_v2_5_instance, sample_options_df_raw, sample_underlying_data_api_raw, mock_config_manager_v2_5):
    if EOTSDataBundleV2_5 is None: pytest.skip("Schemas not loaded")

    current_dt = datetime.now()
    cfg_heatmap_ugch = mock_config_manager_v2_5.get_setting("heatmap_generation_settings", "ugch_greek_weights")
    epsilon = mock_config_manager_v2_5.get_setting("metrics_settings", "common_epsilon")
    ugch_score_col_name = cfg_heatmap_ugch.get("output_col_name", "UGCH_Score")

    bundle = EOTSDataBundleV2_5(options_df_raw=sample_options_df_raw, 
                                und_data_api_raw=sample_underlying_data_api_raw, 
                                current_time_dt=current_dt, symbol="TESTUGCH")
    
    processed_bundle = metrics_calculator_v2_5_instance.calculate_all_metrics(bundle)
    results_strike_df = processed_bundle.df_strike_level_metrics.copy()

    assert ugch_score_col_name in results_strike_df.columns, f"{ugch_score_col_name} not found in df_strike_level_metrics."

    # Manual Calculation
    # These are the expected names of the strike-level aggregated $GreekOI columns
    gxoi_col = cfg_heatmap_ugch['gxoi_col_name'] 
    dxoi_col = cfg_heatmap_ugch['dxoi_col_name']
    vxoi_col = cfg_heatmap_ugch['vxoi_col_name']
    txoi_col = cfg_heatmap_ugch['txoi_col_name']

    for col in [gxoi_col, dxoi_col, vxoi_col, txoi_col]:
        assert col in results_strike_df.columns, f"Prerequisite Greek column '{col}' for UGCH not in strike df."

    # Snapshot normalization (value / max_abs_value)
    norm_gxoi = results_strike_df[gxoi_col] / (results_strike_df[gxoi_col].abs().max() + epsilon)
    norm_dxoi = results_strike_df[dxoi_col] / (results_strike_df[dxoi_col].abs().max() + epsilon)
    norm_vxoi = results_strike_df[vxoi_col] / (results_strike_df[vxoi_col].abs().max() + epsilon)
    norm_txoi = results_strike_df[txoi_col] / (results_strike_df[txoi_col].abs().max() + epsilon)
    
    weights = cfg_heatmap_ugch
    
    expected_ugch_score = (norm_gxoi * weights['gamma_weight'] +
                           norm_dxoi * weights['delta_weight'] +
                           norm_vxoi * weights['vega_weight'] +
                           norm_txoi * weights['theta_weight'])
                           
    pd.testing.assert_series_equal(
        results_strike_df[ugch_score_col_name].fillna(0),
        expected_ugch_score.fillna(0),
        rtol=1e-4,
        check_names=False
    )

def test_calculate_ivsdh_surface(metrics_calculator_v2_5_instance, sample_options_df_raw, sample_underlying_data_api_raw, mock_config_manager_v2_5):
    if EOTSDataBundleV2_5 is None: pytest.skip("Schemas not loaded")

    current_dt = datetime.now()
    cfg_strat = mock_config_manager_v2_5.get_setting("strategy_settings")
    cfg_heatmap_ivsdh = mock_config_manager_v2_5.get_setting("heatmap_generation_settings", "ivsdh_params")
    epsilon = mock_config_manager_v2_5.get_setting("metrics_settings", "common_epsilon")
    
    output_pivot_name = cfg_heatmap_ivsdh.get("output_pivot_table_name", "IVSDH_Surface")

    # Ensure options_df has varied DTEs for a meaningful pivot table
    options_df = sample_options_df_raw.copy() 

    bundle = EOTSDataBundleV2_5(options_df_raw=options_df, 
                                und_data_api_raw=sample_underlying_data_api_raw, 
                                current_time_dt=current_dt, symbol="TESTIVSDH")
    
    processed_bundle = metrics_calculator_v2_5_instance.calculate_all_metrics(bundle)
    
    assert output_pivot_name in processed_bundle.heatmap_data, f"'{output_pivot_name}' not found in heatmap_data."
    ivsdh_surface_actual = processed_bundle.heatmap_data[output_pivot_name]
    assert isinstance(ivsdh_surface_actual, pd.DataFrame), "IVSDH Surface is not a DataFrame."
    assert not ivsdh_surface_actual.empty, "IVSDH Surface is empty."

    # Manual Calculation for a few cells
    # Formula: Norm(VannaOI) + Norm(VommaOI) + Norm(VegaOI) + Norm(CharmOI) + DTE_Sens_Factor
    # These are contract-level $OI greeks
    vannaxoi_col = cfg_heatmap_ivsdh['vanna_oi_col'] # 'vannaxoi'
    vommaxoi_col = cfg_heatmap_ivsdh['vomma_oi_col'] # 'vommaxoi'
    vxoi_col = cfg_heatmap_ivsdh['vega_oi_col']       # 'vxoi'
    charmxoi_col = cfg_heatmap_ivsdh['charm_oi_col'] # 'charmxoi'
    dte_col = cfg_strat['dte_col_name']
    strike_col = cfg_strat['strike_col_name']

    df_calc = options_df.copy()

    # Snapshot Normalization (per contract for these greeks)
    for greek_col in [vannaxoi_col, vommaxoi_col, vxoi_col, charmxoi_col]:
        max_abs_val = df_calc[greek_col].abs().max()
        df_calc[f"norm_{greek_col}"] = df_calc[greek_col] / (max_abs_val + epsilon)

    # DTE Sensitivity Factor (using default from config for simplicity in test)
    # A more complex DTE factor could be a map {dte_bucket: factor}
    dte_sens_factor_val = cfg_heatmap_ivsdh.get("default_dte_sens_factor", 0.5)
    df_calc['dte_sens_factor'] = dte_sens_factor_val # Applying a constant factor for all for simplicity

    df_calc['ivsdh_value_contract'] = df_calc[f"norm_{vannaxoi_col}"] + \
                                     df_calc[f"norm_{vommaxoi_col}"] + \
                                     df_calc[f"norm_{vxoi_col}"] + \
                                     df_calc[f"norm_{charmxoi_col}"] + \
                                     df_calc['dte_sens_factor']
    
    # The IVSDH surface is typically Strike vs DTE. The values are per contract.
    # If multiple contracts share same strike/DTE, the calculator needs to define aggregation (e.g., mean, sum).
    # Assuming here it might take the value from the first available contract if not aggregated.
    # Let's check a specific cell based on one contract from sample_options_df_raw.
    # Example: First contract: strike 95, DTE 0 (from sample_options_df_raw)
    
    if not df_calc.empty:
        first_contract_expected_val = df_calc['ivsdh_value_contract'].iloc[0]
        first_contract_strike = df_calc[strike_col].iloc[0]
        first_contract_dte = df_calc[dte_col].iloc[0]

        # Check if the DTE and Strike from the first contract exist as index/columns in the pivot
        if first_contract_dte in ivsdh_surface_actual.columns and first_contract_strike in ivsdh_surface_actual.index:
            actual_val_cell = ivsdh_surface_actual.loc[first_contract_strike, first_contract_dte]
            assert np.isclose(actual_val_cell, first_contract_expected_val, atol=1e-4), \
                f"Mismatch for IVSDH cell Strike={first_contract_strike}, DTE={first_contract_dte}. Expected {first_contract_expected_val}, got {actual_val_cell}"
        else:
            pytest.skip(f"Strike {first_contract_strike} or DTE {first_contract_dte} not in IVSDH surface, cannot check cell.")
    else:
        assert ivsdh_surface_actual.empty # If no input contracts, surface should be empty.


# --- Original Tier 1 tests below ---
# (test_calculate_hp_eod and subsequent tests remain unchanged from previous state)

def test_calculate_hp_eod(metrics_calculator_v2_5_instance, sample_options_df_raw, sample_underlying_data_api_raw, mock_config_manager_v2_5):
    if EOTSDataBundleV2_5 is None: pytest.skip("Schemas not loaded")

    eod_trigger_str = mock_config_manager_v2_5.get_setting("market_regime_engine_settings", "time_of_day_definitions", "eod_trigger_time")
    eod_trigger_time_obj = datetime.strptime(eod_trigger_str, "%H:%M:%S").time()
    
    today_date = date.today()
    eod_trigger_datetime = datetime.combine(today_date, eod_trigger_time_obj)

    current_price = sample_underlying_data_api_raw['market_data']['price']
    eod_ref_price = sample_underlying_data_api_raw.get('eod_ref_price') 
    assert eod_ref_price is not None, "eod_ref_price not found in sample_underlying_data_api_raw for HP_EOD test"

    time_before_eod = eod_trigger_datetime - timedelta(minutes=5)
    bundle_before_eod = EOTSDataBundleV2_5(options_df_raw=sample_options_df_raw, 
                                           und_data_api_raw=sample_underlying_data_api_raw, 
                                           current_time_dt=time_before_eod, symbol="TESTHPEOD1")
    
    processed_bundle_before = metrics_calculator_v2_5_instance.calculate_all_metrics(bundle_before_eod)
    hp_eod_before = processed_bundle_before.und_data_enriched.get('HP_EOD_Und')
    if 'HP_EOD_Und' in processed_bundle_before.und_data_enriched: 
        assert np.isnan(hp_eod_before), f"Expected HP_EOD_Und to be NaN before EOD trigger, got {hp_eod_before}"

    bundle_at_eod = EOTSDataBundleV2_5(options_df_raw=sample_options_df_raw, 
                                       und_data_api_raw=sample_underlying_data_api_raw, 
                                       current_time_dt=eod_trigger_datetime, symbol="TESTHPEOD2")
    
    processed_bundle_at = metrics_calculator_v2_5_instance.calculate_all_metrics(bundle_at_eod)
    hp_eod_at = processed_bundle_at.und_data_enriched.get('HP_EOD_Und')
    assert hp_eod_at is not None, "HP_EOD_Und not found at EOD trigger"
    
    expected_hp_eod = 0
    if eod_ref_price != 0:
        expected_hp_eod = (current_price - eod_ref_price) / eod_ref_price
    elif current_price == eod_ref_price: 
        expected_hp_eod = 0 
    else: 
        expected_hp_eod = np.inf * np.sign(current_price)

    if np.isinf(expected_hp_eod) and np.isinf(hp_eod_at) and (np.sign(expected_hp_eod) == np.sign(hp_eod_at)):
        assert True
    elif np.isinf(expected_hp_eod) or np.isinf(hp_eod_at):
        assert False, f"Infinity mismatch for HP_EOD: Expected {expected_hp_eod}, got {hp_eod_at}"
    else:
        assert np.isclose(hp_eod_at, expected_hp_eod), f"Expected HP_EOD_Und {expected_hp_eod}, got {hp_eod_at}"

    time_after_eod = eod_trigger_datetime + timedelta(minutes=5)
    und_data_after = sample_underlying_data_api_raw.copy()
    new_current_price = current_price + 0.2 
    und_data_after['market_data']['price'] = new_current_price
    
    bundle_after_eod = EOTSDataBundleV2_5(options_df_raw=sample_options_df_raw, 
                                          und_data_api_raw=und_data_after, 
                                          current_time_dt=time_after_eod, symbol="TESTHPEOD3")
    
    processed_bundle_after = metrics_calculator_v2_5_instance.calculate_all_metrics(bundle_after_eod)
    hp_eod_after = processed_bundle_after.und_data_enriched.get('HP_EOD_Und')
    assert hp_eod_after is not None, "HP_EOD_Und not found after EOD trigger"
    
    expected_hp_eod_after = 0
    if eod_ref_price != 0: 
        expected_hp_eod_after = (new_current_price - eod_ref_price) / eod_ref_price
    elif new_current_price == eod_ref_price:
        expected_hp_eod_after = 0
    else:
        expected_hp_eod_after = np.inf * np.sign(new_current_price)

    if np.isinf(expected_hp_eod_after) and np.isinf(hp_eod_after) and (np.sign(expected_hp_eod_after) == np.sign(hp_eod_after)):
        assert True
    elif np.isinf(expected_hp_eod_after) or np.isinf(hp_eod_after):
        assert False, f"Infinity mismatch for HP_EOD (after): Expected {expected_hp_eod_after}, got {hp_eod_after}"
    else:
        assert np.isclose(hp_eod_after, expected_hp_eod_after), f"Expected HP_EOD_Und (after) {expected_hp_eod_after}, got {hp_eod_after}"


# --- Test for ARFI ---

def test_calculate_arfi(metrics_calculator_v2_5_instance, sample_options_df_raw, sample_underlying_data_api_raw, mock_config_manager_v2_5):
    if EOTSDataBundleV2_5 is None: pytest.skip("Schemas not loaded")

    current_dt = datetime.now()
    data_bundle = EOTSDataBundleV2_5(options_df_raw=sample_options_df_raw, 
                                     und_data_api_raw=sample_underlying_data_api_raw, 
                                     current_time_dt=current_dt, symbol="TESTARFI")
    
    processed_bundle = metrics_calculator_v2_5_instance.calculate_all_metrics(data_bundle)
    arfi = processed_bundle.und_data_enriched.get('ARFI_Und')
    assert arfi is not None, "ARFI_Und not found"

    cfg = mock_config_manager_v2_5.get_setting("strategy_settings")
    arfi_epsilon = mock_config_manager_v2_5.get_setting("metrics_settings", "arfi_epsilon", default=1e-7) 

    df_with_svoi = sample_options_df_raw.copy()
    signed_vol_col = cfg['signed_volume_col_name']
    oi_col = cfg['open_interest_col_name']
    opt_type_col = cfg['option_type_col_name']
    df_with_svoi['sv_x_oi'] = df_with_svoi[signed_vol_col] * df_with_svoi[oi_col]
    calls_df = df_with_svoi[df_with_svoi[opt_type_col] == 'call']
    puts_df = df_with_svoi[df_with_svoi[opt_type_col] == 'put']
    sum_call_sv_x_oi = calls_df['sv_x_oi'].sum()
    sum_put_sv_x_oi = puts_df['sv_x_oi'].sum()
    
    expected_arfi = 0.0
    if abs(sum_put_sv_x_oi) > arfi_epsilon:
        expected_arfi = sum_call_sv_x_oi / sum_put_sv_x_oi
    elif abs(sum_call_sv_x_oi) > arfi_epsilon: 
        expected_arfi = np.inf * np.sign(sum_call_sv_x_oi)

    if np.isinf(expected_arfi) and np.isinf(arfi) and (np.sign(expected_arfi) == np.sign(arfi)):
        assert True
    elif np.isinf(expected_arfi) or np.isinf(arfi):
        assert False, f"Infinity mismatch for ARFI: Expected {expected_arfi}, got {arfi}"
    else:
        assert np.isclose(arfi, expected_arfi, atol=1e-5), f"Expected ARFI_Und {expected_arfi}, got {arfi}"


# --- Test for Net Signed Volume (Premium Weighted) ---
def test_calculate_net_signed_volume_premium_weighted(metrics_calculator_v2_5_instance, sample_options_df_raw, sample_underlying_data_api_raw, mock_config_manager_v2_5):
    if EOTSDataBundleV2_5 is None: pytest.skip("Schemas not loaded")

    current_dt = datetime.now()
    data_bundle = EOTSDataBundleV2_5(options_df_raw=sample_options_df_raw, 
                                     und_data_api_raw=sample_underlying_data_api_raw, 
                                     current_time_dt=current_dt, symbol="TESTNSVPW")
    
    processed_bundle = metrics_calculator_v2_5_instance.calculate_all_metrics(data_bundle)
    nsvp_und = processed_bundle.und_data_enriched.get('NetSignedVolPrem_Und') 
    assert nsvp_und is not None, "NetSignedVolPrem_Und not found"

    cfg = mock_config_manager_v2_5.get_setting("strategy_settings")
    df_copy = sample_options_df_raw.copy()
    signed_vol_col = cfg['signed_volume_col_name']
    mid_price_col = cfg['mid_price_col_name'] 
    multiplier_col = cfg.get("contract_multiplier_col_name", "multiplier") 
    assert multiplier_col in df_copy.columns, f"Multiplier column '{multiplier_col}' not in dataframe for NSVPW test."
    df_copy['nsvp_contract'] = df_copy[signed_vol_col] * df_copy[mid_price_col] * df_copy[multiplier_col]
    expected_nsvp_und = df_copy['nsvp_contract'].sum()
    assert np.isclose(nsvp_und, expected_nsvp_und, atol=1e-5), f"Expected NetSignedVolPrem_Und {expected_nsvp_und}, got {nsvp_und}"

# --- End of Tier 1 Tests / Start of Tier 2 Placeholders ---
# (Comments from previous summary removed for brevity here, will be in final report)

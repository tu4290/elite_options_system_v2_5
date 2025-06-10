from pydantic import BaseModel, Field, FilePath
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import pandas as pd # Will be used where DataFrames are unavoidable

# Placeholder for pandas DataFrame if we decide to type hint it (not true Pydantic validation for content)
PandasDataFrame = Any # Or pd.DataFrame, but then Pydantic won't validate its contents


# --- Canonical Parameter Lists from ConvexValue ---
# For reference and ensuring Raw models are comprehensive.
UNDERLYING_REQUIRED_PARAMS_CV = [
    "price", "volatility", "day_volume", "call_gxoi", "put_gxoi",
    "gammas_call_buy", "gammas_call_sell", "gammas_put_buy", "gammas_put_sell",
    "deltas_call_buy", "deltas_call_sell", "deltas_put_buy", "deltas_put_sell",
    "vegas_call_buy", "vegas_call_sell", "vegas_put_buy", "vegas_put_sell",
    "thetas_call_buy", "thetas_call_sell", "thetas_put_buy", "thetas_put_sell",
    "call_vxoi", "put_vxoi", "value_bs", "volm_bs", "deltas_buy", "deltas_sell",
    "vegas_buy", "vegas_sell", "thetas_buy", "thetas_sell", "volm_call_buy",
    "volm_put_buy", "volm_call_sell", "volm_put_sell", "value_call_buy",
    "value_put_buy", "value_call_sell", "value_put_sell", "vflowratio",
    "dxoi", "gxoi", "vxoi", "txoi", "call_dxoi", "put_dxoi"
]

OPTIONS_CHAIN_REQUIRED_PARAMS_CV = [
    "price", "volatility", "multiplier", "oi", "delta", "gamma", "theta", "vega",
    "vanna", "vomma", "charm", "dxoi", "gxoi", "vxoi", "txoi", "vannaxoi",
    "vommaxoi", "charmxoi", "dxvolm", "gxvolm", "vxvolm", "txvolm", "vannaxvolm",
    "vommaxvolm", "charmxvolm", "value_bs", "volm_bs", "deltas_buy", "deltas_sell",
    "gammas_buy", "gammas_sell", "vegas_buy", "vegas_sell", "thetas_buy", "thetas_sell",
    "valuebs_5m", "volmbs_5m", "valuebs_15m", "volmbs_15m",
    "valuebs_30m", "volmbs_30m", "valuebs_60m", "volmbs_60m",
    "volm", "volm_buy", "volm_sell", "value_buy", "value_sell"
]
# End Canonical Parameter Lists


class RawOptionsContractV2_5(BaseModel):
    contract_symbol: str
    strike: float
    opt_kind: str # "call" or "put"
    dte_calc: float # Calculated DTE

    # Fields corresponding to OPTIONS_CHAIN_REQUIRED_PARAMS_CV
    # Existing fields (comments denote mapping if name differs)
    open_interest: Optional[float] = Field(None, description="Open interest for the contract (maps to CV 'oi')")
    iv: Optional[float] = Field(None, description="Implied Volatility for the contract (maps to CV 'volatility')")
    raw_price: Optional[float] = Field(None, description="Raw price of the option contract from CV (maps to CV 'price')") # Explicit for CV 'price'
    delta_contract: Optional[float] = Field(None, description="Delta per contract (maps to CV 'delta')")
    gamma_contract: Optional[float] = Field(None, description="Gamma per contract (maps to CV 'gamma')")
    theta_contract: Optional[float] = Field(None, description="Theta per contract (maps to CV 'theta')")
    vega_contract: Optional[float] = Field(None, description="Vega per contract (maps to CV 'vega')")
    rho_contract: Optional[float] = Field(None, description="Rho per contract") # Rho is standard, not explicitly in CV list but good to have
    vanna_contract: Optional[float] = Field(None, description="Vanna per contract (maps to CV 'vanna')")
    vomma_contract: Optional[float] = Field(None, description="Vomma per contract (maps to CV 'vomma')")
    charm_contract: Optional[float] = Field(None, description="Charm per contract (maps to CV 'charm')")

    # Greeks OI (Open Interest based Greeks, if provided directly)
    dxoi: Optional[float] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (dxoi)")
    gxoi: Optional[float] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (gxoi)")
    vxoi: Optional[float] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (vxoi)")
    txoi: Optional[float] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (txoi)")
    vannaxoi: Optional[float] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (vannaxoi)")
    vommaxoi: Optional[float] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (vommaxoi)")
    charmxoi: Optional[float] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (charmxoi)")

    # Greek-Volume Proxies (some may be redundant if direct signed Greek flows are available)
    dxvolm: Optional[Any] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (dxvolm)")
    gxvolm: Optional[Any] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (gxvolm)")
    vxvolm: Optional[Any] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (vxvolm)")
    txvolm: Optional[Any] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (txvolm)")
    vannaxvolm: Optional[Any] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (vannaxvolm)")
    vommaxvolm: Optional[Any] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (vommaxvolm)")
    charmxvolm: Optional[Any] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (charmxvolm)")

    # Transaction Data
    value_bs: Optional[float] = Field(None, description="Day Sum of Buy Value minus Sell Value Traded (maps to CV 'value_bs')")
    volm_bs: Optional[float] = Field(None, description="Volume of Buys minus Sells (maps to CV 'volm_bs')")
    volm: Optional[float] = Field(None, description="Total volume for the contract (maps to CV 'volm')")

    # Rolling Flows
    valuebs_5m: Optional[float] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (valuebs_5m)")
    volmbs_5m: Optional[float] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (volmbs_5m)")
    valuebs_15m: Optional[float] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (valuebs_15m)")
    volmbs_15m: Optional[float] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (volmbs_15m)")
    valuebs_30m: Optional[float] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (valuebs_30m)")
    volmbs_30m: Optional[float] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (volmbs_30m)")
    valuebs_60m: Optional[float] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (valuebs_60m)")
    volmbs_60m: Optional[float] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (volmbs_60m)")

    # Bid/Ask for liquidity calculations
    bid_price: Optional[float] = Field(None, description="Bid price of the option")
    ask_price: Optional[float] = Field(None, description="Ask price of the option")
    mid_price: Optional[float] = Field(None, description="Midpoint price of the option")

    # New fields from OPTIONS_CHAIN_REQUIRED_PARAMS_CV
    multiplier: Optional[float] = Field(None, description="Option contract multiplier (e.g., 100) (maps to CV 'multiplier')")
    deltas_buy: Optional[Any] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (deltas_buy)")
    deltas_sell: Optional[Any] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (deltas_sell)")
    gammas_buy: Optional[Any] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (gammas_buy)")
    gammas_sell: Optional[Any] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (gammas_sell)")
    vegas_buy: Optional[Any] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (vegas_buy)")
    vegas_sell: Optional[Any] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (vegas_sell)")
    thetas_buy: Optional[Any] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (thetas_buy)")
    thetas_sell: Optional[Any] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (thetas_sell)")
    volm_buy: Optional[Any] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (volm_buy)")
    volm_sell: Optional[Any] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (volm_sell)")
    value_buy: Optional[Any] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (value_buy)")
    value_sell: Optional[Any] = Field(None, description="Field from ConvexValue OPTIONS_CHAIN_REQUIRED_PARAMS_CV (value_sell)")

    class Config:
        extra = 'allow'

class RawUnderlyingDataV2_5(BaseModel):
    symbol: str
    timestamp: datetime
    price: Optional[float] = Field(None, description="Current underlying price (maps to CV 'price')")
    price_change_abs_und: Optional[float] = Field(None, description="Absolute price change of the underlying")
    price_change_pct_und: Optional[float] = Field(None, description="Percentage price change of the underlying")

    # Tradier specific daily OHLC (can also come from other sources if CV doesn't provide all)
    day_open_price_und: Optional[float] = Field(None, description="Daily open price from primary source (e.g. Tradier)")
    day_high_price_und: Optional[float] = Field(None, description="Daily high price from primary source")
    day_low_price_und: Optional[float] = Field(None, description="Daily low price from primary source")
    prev_day_close_price_und: Optional[float] = Field(None, description="Previous day's closing price from primary source")

    # Fields from UNDERLYING_REQUIRED_PARAMS_CV
    u_volatility: Optional[float] = Field(None, description="General IV for the underlying (maps to CV 'volatility')")
    day_volume: Optional[Any] = Field(None, description="Field from ConvexValue UNDERLYING_REQUIRED_PARAMS_CV (day_volume)")
    call_gxoi: Optional[Any] = Field(None, description="Field from ConvexValue UNDERLYING_REQUIRED_PARAMS_CV (call_gxoi)")
    put_gxoi: Optional[Any] = Field(None, description="Field from ConvexValue UNDERLYING_REQUIRED_PARAMS_CV (put_gxoi)")
    gammas_call_buy: Optional[Any] = Field(None, description="Field from ConvexValue UNDERLYING_REQUIRED_PARAMS_CV (gammas_call_buy)")
    gammas_call_sell: Optional[Any] = Field(None, description="Field from ConvexValue UNDERLYING_REQUIRED_PARAMS_CV (gammas_call_sell)")
    gammas_put_buy: Optional[Any] = Field(None, description="Field from ConvexValue UNDERLYING_REQUIRED_PARAMS_CV (gammas_put_buy)")
    gammas_put_sell: Optional[Any] = Field(None, description="Field from ConvexValue UNDERLYING_REQUIRED_PARAMS_CV (gammas_put_sell)")
    deltas_call_buy: Optional[Any] = Field(None, description="Field from ConvexValue UNDERLYING_REQUIRED_PARAMS_CV (deltas_call_buy)")
    deltas_call_sell: Optional[Any] = Field(None, description="Field from ConvexValue UNDERLYING_REQUIRED_PARAMS_CV (deltas_call_sell)")
    deltas_put_buy: Optional[Any] = Field(None, description="Field from ConvexValue UNDERLYING_REQUIRED_PARAMS_CV (deltas_put_buy)")
    deltas_put_sell: Optional[Any] = Field(None, description="Field from ConvexValue UNDERLYING_REQUIRED_PARAMS_CV (deltas_put_sell)")
    vegas_call_buy: Optional[Any] = Field(None, description="Field from ConvexValue UNDERLYING_REQUIRED_PARAMS_CV (vegas_call_buy)")
    vegas_call_sell: Optional[Any] = Field(None, description="Field from ConvexValue UNDERLYING_REQUIRED_PARAMS_CV (vegas_call_sell)")
    vegas_put_buy: Optional[Any] = Field(None, description="Field from ConvexValue UNDERLYING_REQUIRED_PARAMS_CV (vegas_put_buy)")
    vegas_put_sell: Optional[Any] = Field(None, description="Field from ConvexValue UNDERLYING_REQUIRED_PARAMS_CV (vegas_put_sell)")
    thetas_call_buy: Optional[Any] = Field(None, description="Field from ConvexValue UNDERLYING_REQUIRED_PARAMS_CV (thetas_call_buy)")
    thetas_call_sell: Optional[Any] = Field(None, description="Field from ConvexValue UNDERLYING_REQUIRED_PARAMS_CV (thetas_call_sell)")
    thetas_put_buy: Optional[Any] = Field(None, description="Field from ConvexValue UNDERLYING_REQUIRED_PARAMS_CV (thetas_put_buy)")
    thetas_put_sell: Optional[Any] = Field(None, description="Field from ConvexValue UNDERLYING_REQUIRED_PARAMS_CV (thetas_put_sell)")
    call_vxoi: Optional[Any] = Field(None, description="Field from ConvexValue UNDERLYING_REQUIRED_PARAMS_CV (call_vxoi)")
    put_vxoi: Optional[Any] = Field(None, description="Field from ConvexValue UNDERLYING_REQUIRED_PARAMS_CV (put_vxoi)")
    value_bs: Optional[Any] = Field(None, description="Overall value buy/sell for underlying (maps to CV 'value_bs')")
    volm_bs: Optional[Any] = Field(None, description="Overall volume buy/sell for underlying (maps to CV 'volm_bs')")
    deltas_buy: Optional[Any] = Field(None, description="Overall delta buy for underlying (maps to CV 'deltas_buy')")
    deltas_sell: Optional[Any] = Field(None, description="Overall delta sell for underlying (maps to CV 'deltas_sell')")
    vegas_buy: Optional[Any] = Field(None, description="Overall vega buy for underlying (maps to CV 'vegas_buy')")
    vegas_sell: Optional[Any] = Field(None, description="Overall vega sell for underlying (maps to CV 'vegas_sell')")
    thetas_buy: Optional[Any] = Field(None, description="Overall theta buy for underlying (maps to CV 'thetas_buy')")
    thetas_sell: Optional[Any] = Field(None, description="Overall theta sell for underlying (maps to CV 'thetas_sell')")
    volm_call_buy: Optional[Any] = Field(None, description="Field from ConvexValue UNDERLYING_REQUIRED_PARAMS_CV (volm_call_buy)")
    volm_put_buy: Optional[Any] = Field(None, description="Field from ConvexValue UNDERLYING_REQUIRED_PARAMS_CV (volm_put_buy)")
    volm_call_sell: Optional[Any] = Field(None, description="Field from ConvexValue UNDERLYING_REQUIRED_PARAMS_CV (volm_call_sell)")
    volm_put_sell: Optional[Any] = Field(None, description="Field from ConvexValue UNDERLYING_REQUIRED_PARAMS_CV (volm_put_sell)")
    value_call_buy: Optional[Any] = Field(None, description="Field from ConvexValue UNDERLYING_REQUIRED_PARAMS_CV (value_call_buy)")
    value_put_buy: Optional[Any] = Field(None, description="Field from ConvexValue UNDERLYING_REQUIRED_PARAMS_CV (value_put_buy)")
    value_call_sell: Optional[Any] = Field(None, description="Field from ConvexValue UNDERLYING_REQUIRED_PARAMS_CV (value_call_sell)")
    value_put_sell: Optional[Any] = Field(None, description="Field from ConvexValue UNDERLYING_REQUIRED_PARAMS_CV (value_put_sell)")
    vflowratio: Optional[Any] = Field(None, description="Field from ConvexValue UNDERLYING_REQUIRED_PARAMS_CV (vflowratio)")
    dxoi: Optional[Any] = Field(None, description="Overall dxoi for underlying (maps to CV 'dxoi')")
    gxoi: Optional[Any] = Field(None, description="Overall gxoi for underlying (maps to CV 'gxoi')")
    vxoi: Optional[Any] = Field(None, description="Overall vxoi for underlying (maps to CV 'vxoi')")
    txoi: Optional[Any] = Field(None, description="Overall txoi for underlying (maps to CV 'txoi')")
    call_dxoi: Optional[Any] = Field(None, description="Field from ConvexValue UNDERLYING_REQUIRED_PARAMS_CV (call_dxoi)")
    put_dxoi: Optional[Any] = Field(None, description="Field from ConvexValue UNDERLYING_REQUIRED_PARAMS_CV (put_dxoi)")

    # Other existing fields
    tradier_iv5_approx_smv_avg: Optional[float] = Field(None, description="Tradier IV5 approximation")
    total_call_oi_und: Optional[float] = Field(None, description="Total call OI for the underlying")
    total_put_oi_und: Optional[float] = Field(None, description="Total put OI for the underlying")
    total_call_vol_und: Optional[float] = Field(None, description="Total call volume for the underlying")
    total_put_vol_und: Optional[float] = Field(None, description="Total put volume for the underlying")

    class Config:
        extra = 'allow'

class RawUnderlyingDataCombinedV2_5(RawUnderlyingDataV2_5):
    tradier_open: Optional[float] = Field(None, description="Tradier daily open price")
    tradier_high: Optional[float] = Field(None, description="Tradier daily high price")
    tradier_low: Optional[float] = Field(None, description="Tradier daily low price")
    tradier_close: Optional[float] = Field(None, description="Tradier daily close price")
    tradier_volume: Optional[float] = Field(None, description="Tradier daily volume")
    tradier_vwap: Optional[float] = Field(None, description="Tradier daily VWAP")

    class Config:
        extra = 'allow'

class UnprocessedDataBundleV2_5(BaseModel):
    options_contracts: List[RawOptionsContractV2_5] = Field(default_factory=list)
    underlying_data: RawUnderlyingDataCombinedV2_5
    fetch_timestamp: datetime
    errors: List[str] = Field(default_factory=list)

class ProcessedContractMetricsV2_5(RawOptionsContractV2_5):
    vri_0dte_contract: Optional[float] = None
    vfi_0dte_contract: Optional[float] = None
    vvr_0dte_contract: Optional[float] = None

class ProcessedStrikeLevelMetricsV2_5(BaseModel):
    strike: float
    total_dxoi_at_strike: Optional[float] = None
    total_gxoi_at_strike: Optional[float] = None
    total_vxoi_at_strike: Optional[float] = None
    total_txoi_at_strike: Optional[float] = None
    total_charmxoi_at_strike: Optional[float] = None
    total_vannaxoi_at_strike: Optional[float] = None
    total_vommaxoi_at_strike: Optional[float] = None
    net_cust_delta_flow_at_strike: Optional[float] = None
    net_cust_gamma_flow_at_strike: Optional[float] = None
    net_cust_vega_flow_at_strike: Optional[float] = None
    net_cust_theta_flow_at_strike: Optional[float] = None
    net_cust_charm_flow_proxy_at_strike: Optional[float] = None
    net_cust_vanna_flow_proxy_at_strike: Optional[float] = None
    nvp_at_strike: Optional[float] = None
    nvp_vol_at_strike: Optional[float] = None
    a_dag_strike: Optional[float] = None
    e_sdag_mult_strike: Optional[float] = None
    e_sdag_dir_strike: Optional[float] = None
    e_sdag_w_strike: Optional[float] = None
    e_sdag_vf_strike: Optional[float] = None
    d_tdpi_strike: Optional[float] = None
    e_ctr_strike: Optional[float] = None
    e_tdfi_strike: Optional[float] = None
    vri_2_0_strike: Optional[float] = None
    e_vvr_sens_strike: Optional[float] = None
    e_vfi_sens_strike: Optional[float] = None
    arfi_strike: Optional[float] = None
    sgdhp_score_strike: Optional[float] = None
    ugch_score_strike: Optional[float] = None
    class Config:
        extra = 'allow'

class ProcessedUnderlyingAggregatesV2_5(RawUnderlyingDataCombinedV2_5):
    gib_oi_based_und: Optional[float] = None
    td_gib_und: Optional[float] = None
    hp_eod_und: Optional[float] = None
    net_cust_delta_flow_und: Optional[float] = None
    net_cust_gamma_flow_und: Optional[float] = None
    net_cust_vega_flow_und: Optional[float] = None
    net_cust_theta_flow_und: Optional[float] = None
    net_value_flow_5m_und: Optional[float] = None
    net_vol_flow_5m_und: Optional[float] = None
    net_value_flow_15m_und: Optional[float] = None
    net_vol_flow_15m_und: Optional[float] = None
    net_value_flow_30m_und: Optional[float] = None
    net_vol_flow_30m_und: Optional[float] = None
    net_value_flow_60m_und: Optional[float] = None
    net_vol_flow_60m_und: Optional[float] = None
    vri_0dte_und_sum: Optional[float] = None
    vfi_0dte_und_sum: Optional[float] = None
    vvr_0dte_und_avg: Optional[float] = None
    vci_0dte_agg: Optional[float] = None
    arfi_overall_und_avg: Optional[float] = None
    a_mspi_und_summary_score: Optional[float] = None
    a_sai_und_avg: Optional[float] = None
    a_ssi_und_avg: Optional[float] = None
    vri_2_0_und_aggregate: Optional[float] = None
    vapi_fa_z_score_und: Optional[float] = None
    dwfd_z_score_und: Optional[float] = None
    tw_laf_z_score_und: Optional[float] = None
    ivsdh_surface_data: Optional[PandasDataFrame] = None
    current_market_regime_v2_5: Optional[str] = None
    ticker_context_dict_v2_5: Optional['TickerContextDictV2_5'] = None
    atr_und: Optional[float] = None
    class Config:
        arbitrary_types_allowed = True
        extra = 'allow'

class ProcessedDataBundleV2_5(BaseModel):
    options_data_with_metrics: List[ProcessedContractMetricsV2_5] = Field(default_factory=list)
    strike_level_data_with_metrics: List[ProcessedStrikeLevelMetricsV2_5] = Field(default_factory=list)
    underlying_data_enriched: ProcessedUnderlyingAggregatesV2_5
    processing_timestamp: datetime
    errors: List[str] = Field(default_factory=list)

class TickerContextDictV2_5(BaseModel):
    is_0dte: Optional[bool] = None
    is_1dte: Optional[bool] = None
    is_spx_mwf_expiry_type: Optional[bool] = None
    is_spy_eom_expiry: Optional[bool] = None
    is_quad_witching_week_flag: Optional[bool] = None
    days_to_nearest_0dte: Optional[int] = None
    days_to_monthly_opex: Optional[int] = None
    is_fomc_meeting_day: Optional[bool] = None
    is_fomc_announcement_imminent: Optional[bool] = None
    post_fomc_drift_period_active: Optional[bool] = None
    vix_spy_price_divergence_strong_negative: Optional[bool] = None
    active_intraday_session: Optional[str] = None
    is_near_auction_period: Optional[bool] = None
    ticker_liquidity_profile_flag: Optional[str] = None
    ticker_volatility_state_flag: Optional[str] = None
    earnings_approaching_flag: Optional[bool] = None
    days_to_earnings: Optional[int] = None
    class Config:
        extra = 'allow'

class SignalPayloadV2_5(BaseModel):
    signal_id: str
    signal_name: str
    symbol: str
    timestamp: datetime
    signal_type: str
    direction: Optional[str] = None
    strength_score: float
    strike_impacted: Optional[float] = None
    regime_at_signal_generation: Optional[str] = None
    supporting_metrics: Dict[str, Any] = Field(default_factory=dict)

class KeyLevelV2_5(BaseModel):
    level_price: float
    level_type: str
    conviction_score: float
    contributing_metrics: List[str] = Field(default_factory=list)
    source_identifier: Optional[str] = None

class KeyLevelsDataV2_5(BaseModel):
    supports: List[KeyLevelV2_5] = Field(default_factory=list)
    resistances: List[KeyLevelV2_5] = Field(default_factory=list)
    pin_zones: List[KeyLevelV2_5] = Field(default_factory=list)
    vol_triggers: List[KeyLevelV2_5] = Field(default_factory=list)
    major_walls: List[KeyLevelV2_5] = Field(default_factory=list)
    timestamp: datetime

class ATIFSituationalAssessmentProfileV2_5(BaseModel):
    bullish_assessment_score: float = 0.0
    bearish_assessment_score: float = 0.0
    vol_expansion_score: float = 0.0
    vol_contraction_score: float = 0.0
    mean_reversion_likelihood: float = 0.0
    timestamp: datetime

class ATIFStrategyDirectivePayloadV2_5(BaseModel):
    selected_strategy_type: str
    target_dte_min: int
    target_dte_max: int
    target_delta_long_leg_min: Optional[float] = None
    target_delta_long_leg_max: Optional[float] = None
    target_delta_short_leg_min: Optional[float] = None
    target_delta_short_leg_max: Optional[float] = None
    underlying_price_at_decision: float
    final_conviction_score_from_atif: float
    supportive_rationale_components: Dict[str, Any] = Field(default_factory=dict)
    assessment_profile: ATIFSituationalAssessmentProfileV2_5

class ATIFManagementDirectiveV2_5(BaseModel):
    recommendation_id: str
    action: str
    reason: str
    new_stop_loss: Optional[float] = None
    new_target_1: Optional[float] = None
    new_target_2: Optional[float] = None
    exit_price_type: Optional[str] = None
    percentage_to_manage: Optional[float] = None

class TradeParametersV2_5(BaseModel):
    option_symbol: str
    option_type: str
    strike: float
    expiration_str: str
    entry_price_suggested: float
    stop_loss_price: float
    target_1_price: float
    target_2_price: Optional[float] = None
    target_3_price: Optional[float] = None
    target_rationale: str

class ActiveRecommendationPayloadV2_5(BaseModel):
    recommendation_id: str
    symbol: str
    timestamp_issued: datetime
    strategy_type: str
    selected_option_details: List[Dict[str, Any]]
    trade_bias: str
    entry_price_initial: float
    stop_loss_initial: float
    target_1_initial: float
    target_2_initial: Optional[float] = None
    target_3_initial: Optional[float] = None
    entry_price_actual: Optional[float] = None
    stop_loss_current: float
    target_1_current: float
    target_2_current: Optional[float] = None
    target_3_current: Optional[float] = None
    target_rationale: str
    status: str
    status_update_reason: Optional[str] = None
    atif_conviction_score_at_issuance: float
    triggering_signals_summary: Optional[str] = None
    regime_at_issuance: str
    exit_timestamp: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl_percentage: Optional[float] = None
    pnl_absolute: Optional[float] = None
    exit_reason: Optional[str] = None
    class Config:
        extra = 'allow'

class FinalAnalysisBundleV2_5(BaseModel):
    processed_data_bundle: ProcessedDataBundleV2_5
    scored_signals_v2_5: Dict[str, List[SignalPayloadV2_5]] = Field(default_factory=dict)
    key_levels_data_v2_5: KeyLevelsDataV2_5
    active_recommendations_v2_5: List[ActiveRecommendationPayloadV2_5] = Field(default_factory=list)
    bundle_timestamp: datetime
    target_symbol: str
    system_status_messages: List[str] = Field(default_factory=list)
    class Config:
        arbitrary_types_allowed = True

if __name__ == '__main__':
    raw_contract = RawOptionsContractV2_5(
        contract_symbol="SPY231215C00450000",
        strike=450,
        opt_kind="call",
        dte_calc=10.5,
        open_interest=1000,
        rho_contract=0.05,
        # Example of a newly added field
        multiplier=100.0
    )
    print(raw_contract.model_dump_json(indent=2))

    underlying_info = RawUnderlyingDataCombinedV2_5(
        symbol="SPY",
        timestamp=datetime.now(),
        price=450.50,
        price_change_abs_und=1.5,
        price_change_pct_und=0.0033,
        tradier_open=450.0,
        tradier_high=451.0,
        tradier_low=449.5,
        tradier_close=450.75,
        tradier_volume=1000000,
        tradier_vwap=450.5,
        # Example of a newly added field
        day_volume=120000000
    )
    print(underlying_info.model_dump_json(indent=2))
    # ... (rest of the __main__ block unchanged) ...

# --- EOTS Configuration Models V2.5 ---
# ... (rest of the config models unchanged) ...
# Ensure EOTSConfigV2_5 and its dependencies are defined below this line
# For brevity, assuming the config models section from the original file is appended here without changes.
# This is just a placeholder comment. The actual content from the original file for config models should be used.

# --- EOTS Configuration Models V2.5 ---
# Based on elite_options_system_v2_5/config/config.schema.v2_5.json

# Remove AnyHttpUrl if not used after final check
# from pydantic import BaseModel, Field, FilePath # AnyHttpUrl removed # Already imported at top
# from typing import List, Dict, Any, Optional, Union # Already imported at top
# datetime, pd are already imported from the top of the file

# From definitions in the JSON schema
class DagAlphaCoeffs(BaseModel):
    aligned: float = Field(1.35, description="Multiplier when flow aligns with OI structure.")
    opposed: float = Field(0.65, description="Multiplier when flow opposes OI structure.")
    neutral: float = Field(1.0, description="Multiplier when there is no clear alignment or opposition.")

class SystemSettings(BaseModel):
    project_root_override: Optional[str] = Field(None, description="Absolute path to override the auto-detected project root. Use null for auto-detection.")
    logging_level: str = Field("INFO", description="The minimum level of logs to record.")
    log_to_file: bool = Field(True, description="If true, logs will be written to the file specified in log_file_path.")
    log_file_path: str = Field("logs/eots_v2_5.log", description="Relative path from project root for the log file.", pattern="\\.log$")
    max_log_file_size_bytes: int = Field(10485760, description="Maximum size of a single log file in bytes before rotation.", ge=1024)
    backup_log_count: int = Field(5, description="Number of old log files to keep after rotation.", ge=0)
    metrics_for_dynamic_threshold_distribution_tracking: List[str] = Field(
        default=["GIB_OI_based_Und", "VAPI_FA_Z_Score_Und", "DWFD_Z_Score_Und", "TW_LAF_Z_Score_Und"],
        description="List of underlying aggregate metric names to track historically for dynamic threshold calculations."
    )
    signal_activation: Dict[str, Any] = Field(default_factory=lambda: {"EnableAllSignals": True}, description="Toggles for enabling or disabling specific signal generation routines.")

class DataFetcherSettings(BaseModel):
    convexvalue_api_key: str = Field(..., description="API Key for ConvexValue.")
    tradier_api_key: str = Field(..., description="API Key for Tradier.")
    tradier_account_id: str = Field(..., description="Account ID for Tradier.")
    max_retries: int = Field(3, description="Maximum number of retry attempts for a failing API call.", ge=0)
    retry_delay_seconds: float = Field(5, description="Base delay in seconds between API call retries.", ge=0)

class DataManagementSettings(BaseModel):
    data_cache_dir: str = Field("data_cache_v2_5", description="Directory for caching data.")
    historical_data_store_dir: str = Field("data_cache_v2_5/historical_data_store", description="Directory for historical data storage.")
    performance_data_store_dir: str = Field("data_cache_v2_5/performance_data_store", description="Directory for performance data storage.")

class CoefficientsSettings(BaseModel):
    dag_alpha: DagAlphaCoeffs = Field(default_factory=DagAlphaCoeffs)
    tdpi_beta: DagAlphaCoeffs = Field(default_factory=DagAlphaCoeffs)
    vri_gamma: DagAlphaCoeffs = Field(default_factory=DagAlphaCoeffs)

class DataProcessorSettings(BaseModel):
    factors: Dict[str, Any] = Field(default_factory=dict)
    coefficients: CoefficientsSettings = Field(default_factory=CoefficientsSettings)
    iv_context_parameters: Dict[str, Any] = Field(default_factory=dict)

class StrategySettings(BaseModel):
    class Config:
        extra = 'allow'

class AdaptiveMetricParameters(BaseModel):
    a_dag_settings: Dict[str, Any] = Field(default_factory=dict)
    e_sdag_settings: Dict[str, Any] = Field(default_factory=dict)
    d_tdpi_settings: Dict[str, Any] = Field(default_factory=dict)
    vri_2_0_settings: Dict[str, Any] = Field(default_factory=dict)

class EnhancedFlowMetricSettings(BaseModel):
    class Config:
        extra = 'allow'

class LearningParams(BaseModel):
    performance_tracker_query_lookback: int = Field(90, description="Number of days of historical performance data to consider for learning.", ge=1)
    learning_rate_for_signal_weights: float = Field(0.05, description="How aggressively ATIF adjusts signal weights based on new performance data (0-1 scale).", ge=0, le=1)

class AdaptiveTradeIdeaFrameworkSettings(BaseModel):
    min_conviction_to_initiate_trade: float = Field(2.5, description="The minimum ATIF conviction score (0-5 scale) required to generate a new trade recommendation.", ge=0, le=5)
    learning_params: LearningParams = Field(default_factory=LearningParams)

class TickerContextAnalyzerSettings(BaseModel):
    class Config:
        extra = 'allow'

class KeyLevelIdentifierSettings(BaseModel):
    class Config:
        extra = 'allow'

class HeatmapGenerationSettings(BaseModel):
    class Config:
        extra = 'allow'

class MarketRegimeEngineSettings(BaseModel):
    default_regime: str = Field("REGIME_UNCLEAR_OR_TRANSITIONING")
    regime_evaluation_order: List[Any] = Field(default_factory=list)
    regime_rules: Dict[str, Any] = Field(default_factory=dict)

class VisualizationSettings(BaseModel):
    class Config:
        extra = 'allow'

class SymbolDefaultOverridesStrategySettingsTargets(BaseModel):
    target_atr_stop_loss_multiplier: float = Field(1.5)

class SymbolDefaultOverridesStrategySettings(BaseModel):
    targets: SymbolDefaultOverridesStrategySettingsTargets = Field(default_factory=SymbolDefaultOverridesStrategySettingsTargets)

class SymbolDefaultOverrides(BaseModel):
    strategy_settings: Optional[SymbolDefaultOverridesStrategySettings] = Field(default_factory=SymbolDefaultOverridesStrategySettings)
    class Config:
        extra = 'allow'

class SymbolSpecificOverrides(BaseModel):
    DEFAULT: Optional[SymbolDefaultOverrides] = Field(default_factory=SymbolDefaultOverrides)
    class Config:
        extra = 'allow'

class EOTSConfigV2_5(BaseModel):
    system_settings: SystemSettings = Field(default_factory=SystemSettings)
    data_fetcher_settings: DataFetcherSettings
    data_management_settings: DataManagementSettings = Field(default_factory=DataManagementSettings)
    data_processor_settings: DataProcessorSettings = Field(default_factory=DataProcessorSettings)
    strategy_settings: StrategySettings = Field(default_factory=StrategySettings)
    adaptive_metric_parameters: AdaptiveMetricParameters = Field(default_factory=AdaptiveMetricParameters)
    enhanced_flow_metric_settings: EnhancedFlowMetricSettings = Field(default_factory=EnhancedFlowMetricSettings)
    adaptive_trade_idea_framework_settings: AdaptiveTradeIdeaFrameworkSettings
    ticker_context_analyzer_settings: TickerContextAnalyzerSettings = Field(default_factory=TickerContextAnalyzerSettings)
    key_level_identifier_settings: KeyLevelIdentifierSettings = Field(default_factory=KeyLevelIdentifierSettings)
    heatmap_generation_settings: HeatmapGenerationSettings = Field(default_factory=HeatmapGenerationSettings)
    market_regime_engine_settings: MarketRegimeEngineSettings = Field(default_factory=MarketRegimeEngineSettings)
    visualization_settings: VisualizationSettings = Field(default_factory=VisualizationSettings)
    symbol_specific_overrides: SymbolSpecificOverrides = Field(default_factory=SymbolSpecificOverrides)

    class Config:
        json_schema_extra = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "EOTS_V2_5_Config_Schema",
            "description": "Canonical schema for EOTS v2.5 configuration (config_v2_5.json). Defines all valid parameters, types, defaults, and descriptions for system operation."
        }
        extra = 'forbid'

if __name__ == '__main__':
    # Example of creating a default config and validating
    try:
        # Note: DataFetcherSettings requires API keys, so direct instantiation of EOTSConfigV2_5
        # without providing them will fail if we don't mock them or load a real config.
        # For a simple test here, we can check individual components or a mocked main config.

        print("\n--- Testing DagAlphaCoeffs ---")
        dag_coeffs = DagAlphaCoeffs()
        print(dag_coeffs.model_dump_json(indent=2))

        print("\n--- Testing SystemSettings (Default) ---")
        system_settings = SystemSettings()
        print(system_settings.model_dump_json(indent=2))

        print("\n--- Testing DataManagementSettings (Default) ---")
        data_mgmt_settings = DataManagementSettings()
        print(data_mgmt_settings.model_dump_json(indent=2))

        print("\n--- Testing CoefficientsSettings (Default) ---")
        coeffs_settings = CoefficientsSettings()
        print(coeffs_settings.model_dump_json(indent=2))

        print("\n--- Testing LearningParams (Default) ---")
        learning_params = LearningParams()
        print(learning_params.model_dump_json(indent=2))

        print("\n--- Testing SymbolDefaultOverrides (Default) ---")
        symbol_default_overrides = SymbolDefaultOverrides()
        print(symbol_default_overrides.model_dump_json(indent=2))

        print("\n--- Testing SymbolSpecificOverrides (Default) ---")
        symbol_specific_overrides = SymbolSpecificOverrides()
        print(symbol_specific_overrides.model_dump_json(indent=2))

        symbol_specific_overrides_custom = SymbolSpecificOverrides(
            DEFAULT=SymbolDefaultOverrides(),
            SPY={"some_specific_setting_for_spy": True}
        )
        print(symbol_specific_overrides_custom.model_dump_json(indent=2))


        print("\n--- Minimal EOTSConfigV2_5 (requires DataFetcherSettings) ---")

        min_config_data = {
            "data_fetcher_settings": {
                "convexvalue_api_key": "YOUR_CV_KEY",
                "tradier_api_key": "YOUR_TRADIER_KEY",
                "tradier_account_id": "YOUR_TRADIER_ACCOUNT_ID"
            },
            "adaptive_trade_idea_framework_settings": { # Added as it's required
                 "min_conviction_to_initiate_trade": 2.0
            }
        }
        config_instance = EOTSConfigV2_5(**min_config_data)
        print("Minimal EOTSConfigV2_5 instance created successfully.")

        print("\nAll new config Pydantic models defined and basic tests passed.")

    except Exception as e:
        print(f"An error occurred during __main__ tests for config models: {e}")

    # Restore original print for example usage
    print("\n--- Original Model Examples ---")
    raw_contract = RawOptionsContractV2_5(
        contract_symbol="SPY231215C00450000",
        strike=450,
        opt_kind="call",
        dte_calc=10.5,
        open_interest=1000,
        rho_contract=0.05,
        multiplier=100.0
    )
    print(raw_contract.model_dump_json(indent=2))

    underlying_info = RawUnderlyingDataCombinedV2_5(
        symbol="SPY",
        timestamp=datetime.now(),
        price=450.50,
        price_change_abs_und=1.5,
        price_change_pct_und=0.0033,
        tradier_open=450.0,
        tradier_high=451.0,
        tradier_low=449.5,
        tradier_close=450.75,
        tradier_volume=1000000,
        tradier_vwap=450.5,
        day_volume=120000000
    )
    print(underlying_info.model_dump_json(indent=2))

    raw_bundle = UnprocessedDataBundleV2_5(options_contracts=[raw_contract], underlying_data=underlying_info, fetch_timestamp=datetime.now())

    processed_underlying = ProcessedUnderlyingAggregatesV2_5(
        symbol="SPY", timestamp=datetime.now(), price=450.50,
        gib_oi_based_und=-500e6, vapi_fa_z_score_und=2.1,
        current_market_regime_v2_5="REGIME_BULLISH_TREND_VAPI_CONFIRMED"
    )
    print(processed_underlying.model_dump_json(indent=2))

    final_bundle_example = FinalAnalysisBundleV2_5(
        processed_data_bundle=ProcessedDataBundleV2_5(
            underlying_data_enriched=processed_underlying,
            processing_timestamp=datetime.now()
        ),
        key_levels_data_v2_5=KeyLevelsDataV2_5(timestamp=datetime.now()),
        bundle_timestamp=datetime.now(),
        target_symbol="SPY"
    )

    print("Pydantic models defined.")

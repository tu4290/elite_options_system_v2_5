from pydantic import BaseModel, Field, FilePath
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import pandas as pd # Will be used where DataFrames are unavoidable

# Placeholder for pandas DataFrame if we decide to type hint it (not true Pydantic validation for content)
PandasDataFrame = Any # Or pd.DataFrame, but then Pydantic won't validate its contents


# Config model is defined at the end of this file


class RawOptionsContractDataV2_5(BaseModel):
    # Fields directly from get_chain for a single contract
    # Example fields - need to be comprehensive based on actual get_chain output used by v2.4/v2.5
    contract_symbol: str
    strike: float
    opt_kind: str # "call" or "put"
    dte_calc: float # Calculated DTE
    # Greeks OI
    dxoi: Optional[float] = None
    gxoi: Optional[float] = None
    vxoi: Optional[float] = None
    txoi: Optional[float] = None
    charmxoi: Optional[float] = None
    vannaxoi: Optional[float] = None
    vommaxoi: Optional[float] = None
    # Greeks per contract (if available and used)
    delta_contract: Optional[float] = None
    gamma_contract: Optional[float] = None
    vega_contract: Optional[float] = None
    theta_contract: Optional[float] = None
    charm_contract: Optional[float] = None
    vanna_contract: Optional[float] = None
    vomma_contract: Optional[float] = None
    # Flows per contract (rolling, from get_chain)
    valuebs_5m: Optional[float] = None
    volmbs_5m: Optional[float] = None
    valuebs_15m: Optional[float] = None
    volmbs_15m: Optional[float] = None
    valuebs_30m: Optional[float] = None
    volmbs_30m: Optional[float] = None
    valuebs_60m: Optional[float] = None
    volmbs_60m: Optional[float] = None
    # Other transaction data per contract
    value_bs: Optional[float] = None # Day Sum of Buy Value minus Sell Value Traded
    volm_bs: Optional[float] = None  # Volume of Buys minus Sells
    volm: Optional[float] = None # Total volume for the contract
    # Bid/Ask for liquidity calculations
    bid_price: Optional[float] = None
    ask_price: Optional[float] = None
    mid_price: Optional[float] = None
    iv: Optional[float] = None # Implied Volatility for the contract
    # Fields for proxies if direct signed Greek flows per contract are not available
    # (e.g., for VRI 0DTE calculation)
    vannaxvolm: Optional[float] = None # Vanna * Volume
    vommaxvolm: Optional[float] = None # Vomma * Volume
    gxvolm: Optional[float] = None # Gamma * Volume (for potential A-DAG refinement)
    charmxvolm: Optional[float] = None # Charm * Volume (for D-TDPI flow proxy)

    class Config:
        extra = 'allow' # Allow extra fields initially if v2.4 has more

class RawUnderlyingDataV2_5(BaseModel):
    # Fields from get_und and Tradier snapshot
    symbol: str
    timestamp: datetime
    price: Optional[float] = None # Current underlying price
    day_open_price_und: Optional[float] = None
    day_high_price_und: Optional[float] = None
    day_low_price_und: Optional[float] = None
    prev_day_close_price_und: Optional[float] = None
    # Aggregated Greek OI from get_und (if still used, v2.5 guide implies summing from get_chain)
    # Example: call_dxoi_sum_und, put_dxoi_sum_und for GIB calculation if not from chain
    # Aggregated Customer Flows from get_und (if still used)
    # Example: gammas_call_buy_und, gammas_put_sell_und for NetCustGammaFlow if not from chain
    # Implied Volatility from Tradier/other source
    tradier_iv5_approx_smv_avg: Optional[float] = None
    u_volatility: Optional[float] = None # General IV for the underlying (e.g. from CV get_und)
    # Other relevant fields from get_und
    # ... many potential fields based on v2.4 usage and v2.5 needs ...
    # Fields for total call/put OI/Volume if needed for ratios
    total_call_oi_und: Optional[float] = None
    total_put_oi_und: Optional[float] = None
    total_call_vol_und: Optional[float] = None
    total_put_vol_und: Optional[float] = None

    # Per-strike signed greek flows from get_chain (these are aggregated to underlying in metrics_calculator)
    # but the raw per-strike data might be part of a more detailed bundle before full aggregation
    # For now, assuming underlying aggregates are calculated and stored in ProcessedUnderlyingAggregatesV2_5
    # Example of where such strike-level data might come from:
    # deltas_call_buy_strike: Dict[float, float] = Field(default_factory=dict) # strike -> value
    # This model is for the *underlying specific data*, not the full chain.

    class Config:
        extra = 'allow'

class UnprocessedDataBundleV2_5(BaseModel):
    # This bundle is what fetchers provide to initial_processor (formerly RawDataBundleV2_5)
    # It contains the list of all contracts from get_chain and the combined underlying data
    options_contracts: List[RawOptionsContractDataV2_5] = Field(default_factory=list)
    underlying_data: RawUnderlyingDataCombinedV2_5
    fetch_timestamp: datetime
    errors: List[str] = Field(default_factory=list)

# --- Processed Data Models ---

class ProcessedContractMetricsV2_5(RawOptionsContractDataV2_5): # Inherits all raw fields
    # Adds calculated metrics specific to each contract
    # Adaptive Metrics - per contract where applicable
    vri_0dte_contract: Optional[float] = None
    vfi_0dte_contract: Optional[float] = None
    vvr_0dte_contract: Optional[float] = None
    # Other per-contract calculations if any
    # ...

class ProcessedStrikeLevelMetricsV2_5(BaseModel):
    strike: float
    # Aggregated OI at strike
    total_dxoi_at_strike: Optional[float] = None
    total_gxoi_at_strike: Optional[float] = None
    total_vxoi_at_strike: Optional[float] = None
    total_txoi_at_strike: Optional[float] = None
    total_charmxoi_at_strike: Optional[float] = None
    total_vannaxoi_at_strike: Optional[float] = None
    total_vommaxoi_at_strike: Optional[float] = None
    # Aggregated Net Customer Flows at strike (from get_chain call/put components)
    net_cust_delta_flow_at_strike: Optional[float] = None
    net_cust_gamma_flow_at_strike: Optional[float] = None # True signed gamma flow
    net_cust_vega_flow_at_strike: Optional[float] = None  # True signed vega flow
    net_cust_theta_flow_at_strike: Optional[float] = None # True signed theta flow
    net_cust_charm_flow_proxy_at_strike: Optional[float] = None # From charmxvolm sum
    net_cust_vanna_flow_proxy_at_strike: Optional[float] = None # From vannaxvolm sum
    # NVP/NVP_Vol
    nvp_at_strike: Optional[float] = None
    nvp_vol_at_strike: Optional[float] = None
    # Adaptive Metrics - at strike level
    a_dag_strike: Optional[float] = None
    e_sdag_mult_strike: Optional[float] = None
    e_sdag_dir_strike: Optional[float] = None
    e_sdag_w_strike: Optional[float] = None
    e_sdag_vf_strike: Optional[float] = None
    d_tdpi_strike: Optional[float] = None
    e_ctr_strike: Optional[float] = None # Enhanced Charm Decay Rate
    e_tdfi_strike: Optional[float] = None # Enhanced Time Decay Flow Imbalance
    vri_2_0_strike: Optional[float] = None
    e_vvr_sens_strike: Optional[float] = None # Enhanced Vanna-Vomma Ratio (from VRI 2.0 components)
    e_vfi_sens_strike: Optional[float] = None # Enhanced Volatility Flow Indicator (from VRI 2.0 components)
    # ARFI
    arfi_strike: Optional[float] = None
    # Data for Heatmaps - calculated per strike
    sgdhp_score_strike: Optional[float] = None
    ugch_score_strike: Optional[float] = None
    # IVSDH is a surface (strike vs DTE), might be stored differently, see ProcessedUnderlyingAggregatesV2_5

    class Config:
        extra = 'allow'


class ProcessedUnderlyingAggregatesV2_5(RawUnderlyingDataCombinedV2_5): # Inherits raw underlying fields
    # Tier 1 Foundational Metrics (Underlying Level)
    gib_oi_based_und: Optional[float] = None # Gamma Imbalance from OI
    td_gib_und: Optional[float] = None # Traded Dealer Gamma Imbalance
    hp_eod_und: Optional[float] = None # End-of-Day Hedging Pressure
    net_cust_delta_flow_und: Optional[float] = None # Daily total
    net_cust_gamma_flow_und: Optional[float] = None # Daily total
    net_cust_vega_flow_und: Optional[float] = None  # Daily total
    net_cust_theta_flow_und: Optional[float] = None # Daily total
    # Standard Rolling Net Signed Flows (Underlying Level)
    net_value_flow_5m_und: Optional[float] = None
    net_vol_flow_5m_und: Optional[float] = None
    net_value_flow_15m_und: Optional[float] = None
    net_vol_flow_15m_und: Optional[float] = None
    net_value_flow_30m_und: Optional[float] = None
    net_vol_flow_30m_und: Optional[float] = None
    net_value_flow_60m_und: Optional[float] = None
    net_vol_flow_60m_und: Optional[float] = None
    # 0DTE Suite (Aggregated to Underlying)
    vri_0dte_und_sum: Optional[float] = None
    vfi_0dte_und_sum: Optional[float] = None
    vvr_0dte_und_avg: Optional[float] = None # Or sum, depending on interpretation
    vci_0dte_agg: Optional[float] = None # Vanna Concentration Index (HHI-style)
    # ARFI (Aggregated)
    arfi_overall_und_avg: Optional[float] = None
    # Tier 2 Adaptive Metrics (Aggregated where applicable, or profile data)
    # A-MSPI and its components (A-SAI, A-SSI) are often profiles or summary scores
    a_mspi_und_summary_score: Optional[float] = None # Example
    a_sai_und_avg: Optional[float] = None
    a_ssi_und_avg: Optional[float] = None
    # VRI 2.0 aggregate for overall vol context
    vri_2_0_und_aggregate: Optional[float] = None
    # Tier 3 Enhanced Rolling Flow Metrics (Underlying Level)
    vapi_fa_z_score_und: Optional[float] = None
    dwfd_z_score_und: Optional[float] = None
    tw_laf_z_score_und: Optional[float] = None
    # Data for IVSDH Heatmap (Strike vs DTE) - can be a serialized DataFrame or List of Lists
    ivsdh_surface_data: Optional[PandasDataFrame] = None # Or List[List[float]] with separate strike/DTE axes
    # Current Market Regime & Ticker Context (added by orchestrator after MRE/TCA steps)
    current_market_regime_v2_5: Optional[str] = None
    ticker_context_dict_v2_5: Optional['TickerContextDictV2_5'] = None # Changed type hint
    # ATR for TPO
    atr_und: Optional[float] = None

    class Config:
        arbitrary_types_allowed = True # For PandasDataFrame if used
        extra = 'allow'

class ProcessedDataBundleV2_5(BaseModel):
    # This is what initial_processor + metrics_calculator produce for the orchestrator
    options_data_with_metrics: List[ProcessedContractMetricsV2_5] = Field(default_factory=list) # Full chain with per-contract metrics
    strike_level_data_with_metrics: List[ProcessedStrikeLevelMetricsV2_5] = Field(default_factory=list) # Strike-aggregated metrics
    underlying_data_enriched: ProcessedUnderlyingAggregatesV2_5 # All underlying aggregates and context
    processing_timestamp: datetime
    errors: List[str] = Field(default_factory=list)

# --- Context, Signals, Levels ---
class TickerContextDictV2_5(BaseModel): # As per Section VI of the guide
    # SPY/SPX Specific
    is_0dte: Optional[bool] = None
    is_1dte: Optional[bool] = None
    is_spx_mwf_expiry_type: Optional[bool] = None
    is_spy_eom_expiry: Optional[bool] = None
    is_quad_witching_week_flag: Optional[bool] = None
    days_to_nearest_0dte: Optional[int] = None
    days_to_monthly_opex: Optional[int] = None
    # Behavioral Patterns (Examples)
    is_fomc_meeting_day: Optional[bool] = None
    is_fomc_announcement_imminent: Optional[bool] = None
    post_fomc_drift_period_active: Optional[bool] = None
    vix_spy_price_divergence_strong_negative: Optional[bool] = None
    # Intraday Patterns
    active_intraday_session: Optional[str] = None # e.g., "OPENING_VOLATILITY", "LUNCH_LULL"
    is_near_auction_period: Optional[bool] = None
    # General Tickers
    ticker_liquidity_profile_flag: Optional[str] = None # e.g., "High", "Medium", "Low"
    ticker_volatility_state_flag: Optional[str] = None # e.g., "IV_HIGH_RV_LOW"
    earnings_approaching_flag: Optional[bool] = None
    days_to_earnings: Optional[int] = None

    class Config:
        extra = 'allow' # For additional dynamic flags

class SignalPayloadV2_5(BaseModel): # As per Section VIII
    signal_id: str # Unique ID for this signal instance
    signal_name: str # e.g., "VAPI_FA_Bullish_Surge", "Adaptive_Directional_Bullish_A_DAG_Support"
    symbol: str
    timestamp: datetime
    signal_type: str # e.g., "Directional", "Volatility", "Flow_Momentum", "Pin_Risk"
    direction: Optional[str] = None # "Bullish", "Bearish", "Neutral"
    strength_score: float # Continuous score, e.g., 0.0 to 1.0, or -1.0 to +1.0
    strike_impacted: Optional[float] = None
    # Other relevant details
    regime_at_signal_generation: Optional[str] = None
    supporting_metrics: Dict[str, Any] = Field(default_factory=dict)
    # Deprecating initial_stars for strength_score

class KeyLevelV2_5(BaseModel): # As per Section VII
    level_price: float
    level_type: str # "Support", "Resistance", "PinZone", "VolTrigger", "MajorWall"
    conviction_score: float # e.g., 0.0 to 1.0
    contributing_metrics: List[str] = Field(default_factory=list)
    source_identifier: Optional[str] = None # e.g. "A-MSPI", "NVP_Peak", "SGDHP"

class KeyLevelsDataV2_5(BaseModel):
    supports: List[KeyLevelV2_5] = Field(default_factory=list)
    resistances: List[KeyLevelV2_5] = Field(default_factory=list)
    pin_zones: List[KeyLevelV2_5] = Field(default_factory=list)
    vol_triggers: List[KeyLevelV2_5] = Field(default_factory=list)
    major_walls: List[KeyLevelV2_5] = Field(default_factory=list)
    timestamp: datetime

# --- ATIF Models (Section IX) ---
class ATIFSituationalAssessmentProfileV2_5(BaseModel):
    # Net assessed strength for various potential biases
    bullish_assessment_score: float = 0.0
    bearish_assessment_score: float = 0.0
    vol_expansion_score: float = 0.0
    vol_contraction_score: float = 0.0
    mean_reversion_likelihood: float = 0.0
    # ... other relevant assessment dimensions
    timestamp: datetime

class ATIFStrategyDirectivePayloadV2_5(BaseModel):
    # Output of ATIF Component 3
    selected_strategy_type: str # e.g., "BullCallSpread", "LongPut", "ShortIronCondor"
    target_dte_min: int
    target_dte_max: int
    target_delta_long_leg_min: Optional[float] = None
    target_delta_long_leg_max: Optional[float] = None
    target_delta_short_leg_min: Optional[float] = None
    target_delta_short_leg_max: Optional[float] = None
    # Potentially target IV preference, underlying price at decision etc.
    underlying_price_at_decision: float
    final_conviction_score_from_atif: float # The numeric score (0-5)
    supportive_rationale_components: Dict[str, Any] = Field(default_factory=dict)
    # Include the situational assessment that led to this directive
    assessment_profile: ATIFSituationalAssessmentProfileV2_5

class ATIFManagementDirectiveV2_5(BaseModel):
    # Output of ATIF Component 4
    recommendation_id: str # ID of the active recommendation being managed
    action: str # "EXIT", "ADJUST_STOPLOSS", "ADJUST_TARGET", "PARTIAL_PROFIT_TAKE", "HOLD"
    reason: str
    # Specific parameters for the action
    new_stop_loss: Optional[float] = None
    new_target_1: Optional[float] = None
    new_target_2: Optional[float] = None
    exit_price_type: Optional[str] = None # e.g., "Market", "Mid"
    percentage_to_manage: Optional[float] = None # For partial takes/closes

class TradeParametersV2_5(BaseModel): # Output of TPO (Section X)
    # Details for a single option leg
    option_symbol: str
    option_type: str # Call/Put
    strike: float
    expiration_str: str # YYYY-MM-DD
    # Parameters for the overall trade
    entry_price_suggested: float # For the option or spread
    stop_loss_price: float       # For the option or spread
    target_1_price: float        # For the option or spread
    target_2_price: Optional[float] = None
    target_3_price: Optional[float] = None
    target_rationale: str
    # If multi-leg, this might be a list of single leg details + overall spread params
    # For now, keeping it simple, assuming TPO provides final actionable prices

class ActiveRecommendationPayloadV2_5(BaseModel): # Stored by Orchestrator
    recommendation_id: str
    symbol: str
    timestamp_issued: datetime
    strategy_type: str # From ATIF
    # Selected option details - could be a list of dicts for multi-leg
    # For simplicity, let's assume TPO flattens this if possible, or we use a nested model
    selected_option_details: List[Dict[str, Any]] # [{symbol, strike, type, entry_price_leg}, ...]
    trade_bias: str # Bullish/Bearish/Neutral
    
    # Initial parameters from TPO
    entry_price_initial: float
    stop_loss_initial: float
    target_1_initial: float
    target_2_initial: Optional[float] = None
    target_3_initial: Optional[float] = None
    
    # Current / Adjusted parameters
    entry_price_actual: Optional[float] = None # If filled
    stop_loss_current: float
    target_1_current: float
    target_2_current: Optional[float] = None
    target_3_current: Optional[float] = None
    
    target_rationale: str
    status: str # e.g., "ACTIVE_NEW_NO_TSL", "ACTIVE_TSL_ADJUSTED", "EXITED_T1_HIT", "EXITED_SL_HIT"
    status_update_reason: Optional[str] = None
    
    # Context at issuance
    atif_conviction_score_at_issuance: float # The 0-5 score
    triggering_signals_summary: Optional[str] = None # Brief summary
    regime_at_issuance: str
    
    # Performance tracking fields (populated on close)
    exit_timestamp: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl_percentage: Optional[float] = None
    pnl_absolute: Optional[float] = None
    exit_reason: Optional[str] = None

    class Config:
        extra = 'allow'

# --- Main Bundle for Dashboard ---
class FinalAnalysisBundleV2_5(BaseModel):
    # This is the comprehensive bundle passed to the dashboard after a full analysis cycle
    # It contains all necessary data for all dashboard modes
    # Replaces previous "analysis_bundle"

    # Core data from processing
    processed_data_bundle: ProcessedDataBundleV2_5 # Contains all metric-enriched data

    # Outputs from analytical components
    # current_market_regime_v2_5: str # Already in processed_data_bundle.underlying_data_enriched
    # ticker_context_dict_v2_5: TickerContextDictV2_5 # Already in processed_data_bundle.underlying_data_enriched
    
    scored_signals_v2_5: Dict[str, List[SignalPayloadV2_5]] = Field(default_factory=dict) # category -> list of signals
    key_levels_data_v2_5: KeyLevelsDataV2_5
    
    # Active recommendations from Orchestrator
    active_recommendations_v2_5: List[ActiveRecommendationPayloadV2_5] = Field(default_factory=list)
    
    # Other global info
    bundle_timestamp: datetime
    target_symbol: str
    system_status_messages: List[str] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True # For ProcessedDataBundleV2_5 containing DataFrames if any


# Example Usage (for testing model definitions, not part of the actual file)
if __name__ == '__main__':
    raw_contract = RawOptionsContractDataV2_5(contract_symbol="SPY231215C00450000", strike=450, opt_kind="call", dte_calc=10.5)
    print(raw_contract.json(indent=2))

    underlying_info = RawUnderlyingDataCombinedV2_5(symbol="SPY", timestamp=datetime.now(), price=450.50)
    print(underlying_info.json(indent=2))

    raw_bundle = UnprocessedDataBundleV2_5(options_contracts=[raw_contract], underlying_data=underlying_info, fetch_timestamp=datetime.now())
    # print(raw_bundle.json(indent=2)) # Can be very long

    processed_underlying = ProcessedUnderlyingAggregatesV2_5(
        symbol="SPY", timestamp=datetime.now(), price=450.50,
        gib_oi_based_und=-500e6, vapi_fa_z_score_und=2.1,
        current_market_regime_v2_5="REGIME_BULLISH_TREND_VAPI_CONFIRMED"
    )
    print(processed_underlying.json(indent=2))

    final_bundle_example = FinalAnalysisBundleV2_5(
        processed_data_bundle=ProcessedDataBundleV2_5(
            underlying_data_enriched=processed_underlying,
            processing_timestamp=datetime.now()
        ),
        key_levels_data_v2_5=KeyLevelsDataV2_5(timestamp=datetime.now()),
        bundle_timestamp=datetime.now(),
        target_symbol="SPY"
    )
    # print(final_bundle_example.json(indent=2)) # Can be very long

    print("Pydantic models defined.")


# --- EOTS Configuration Models V2.5 ---
# Based on elite_options_system_v2_5/config/config.schema.v2_5.json

# Remove AnyHttpUrl if not used after final check
from pydantic import BaseModel, Field, FilePath # AnyHttpUrl removed
from typing import List, Dict, Any, Optional, Union
# datetime, pd are already imported from the top of the file

# From definitions in the JSON schema
class DagAlphaCoeffs(BaseModel):
    aligned: float = Field(1.35, description="Multiplier when flow aligns with OI structure.")
    opposed: float = Field(0.65, description="Multiplier when flow opposes OI structure.")
    neutral: float = Field(1.0, description="Multiplier when there is no clear alignment or opposition.")

class SystemSettings(BaseModel):
    project_root_override: Optional[str] = Field(None, description="Absolute path to override the auto-detected project root. Use null for auto-detection.")
    logging_level: str = Field("INFO", description="The minimum level of logs to record.") # enum handled by Pydantic's validation if a Literal type is used, or by validator. For now, simple str.
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

class StrategySettings(BaseModel): # Empty in schema, structure can be added if known
    class Config:
        extra = 'allow' # Allow any strategy specific keys

class AdaptiveMetricParameters(BaseModel):
    a_dag_settings: Dict[str, Any] = Field(default_factory=dict)
    e_sdag_settings: Dict[str, Any] = Field(default_factory=dict)
    d_tdpi_settings: Dict[str, Any] = Field(default_factory=dict)
    vri_2_0_settings: Dict[str, Any] = Field(default_factory=dict)

class EnhancedFlowMetricSettings(BaseModel): # Empty in schema
    class Config:
        extra = 'allow'

class LearningParams(BaseModel):
    performance_tracker_query_lookback: int = Field(90, description="Number of days of historical performance data to consider for learning.", ge=1)
    learning_rate_for_signal_weights: float = Field(0.05, description="How aggressively ATIF adjusts signal weights based on new performance data (0-1 scale).", ge=0, le=1)

class AdaptiveTradeIdeaFrameworkSettings(BaseModel):
    min_conviction_to_initiate_trade: float = Field(2.5, description="The minimum ATIF conviction score (0-5 scale) required to generate a new trade recommendation.", ge=0, le=5)
    learning_params: LearningParams = Field(default_factory=LearningParams)

class TickerContextAnalyzerSettings(BaseModel): # Empty in schema
    class Config:
        extra = 'allow'

class KeyLevelIdentifierSettings(BaseModel): # Empty in schema
    class Config:
        extra = 'allow'

class HeatmapGenerationSettings(BaseModel): # Empty in schema
    class Config:
        extra = 'allow'

class MarketRegimeEngineSettings(BaseModel):
    default_regime: str = Field("REGIME_UNCLEAR_OR_TRANSITIONING")
    regime_evaluation_order: List[Any] = Field(default_factory=list)
    regime_rules: Dict[str, Any] = Field(default_factory=dict)

class VisualizationSettings(BaseModel): # Empty in schema
    class Config:
        extra = 'allow'

class SymbolDefaultOverridesStrategySettingsTargets(BaseModel):
    target_atr_stop_loss_multiplier: float = Field(1.5)

class SymbolDefaultOverridesStrategySettings(BaseModel):
    targets: SymbolDefaultOverridesStrategySettingsTargets = Field(default_factory=SymbolDefaultOverridesStrategySettingsTargets)

class SymbolDefaultOverrides(BaseModel):
    strategy_settings: Optional[SymbolDefaultOverridesStrategySettings] = Field(default_factory=SymbolDefaultOverridesStrategySettings)
    # Add other overridable sections here if schema for DEFAULT evolves (e.g. adaptive_metric_parameters)
    class Config:
        extra = 'allow' # Allow other keys within DEFAULT if not explicitly defined yet

class SymbolSpecificOverrides(BaseModel):
    DEFAULT: Optional[SymbolDefaultOverrides] = Field(default_factory=SymbolDefaultOverrides)
    # Pydantic allows additional fields not defined in the model if extra='allow'
    # The schema's "additionalProperties": {"type": "object"} means other keys (e.g. "SPY")
    # should map to objects. We can define a generic object model or use Dict[str, Any].
    # For now, this structure with extra='allow' will accept any additional symbol keys
    # with any structure, but validation won't be specific for them unless models are defined.
    class Config:
        extra = 'allow' # Allows for keys like "SPY", "QQQ" etc. to be present

# Main Configuration Model
class EOTSConfigV2_5(BaseModel):
    system_settings: SystemSettings = Field(default_factory=SystemSettings)
    data_fetcher_settings: DataFetcherSettings # Required as per schema
    data_management_settings: DataManagementSettings = Field(default_factory=DataManagementSettings)
    data_processor_settings: DataProcessorSettings = Field(default_factory=DataProcessorSettings)
    strategy_settings: StrategySettings = Field(default_factory=StrategySettings)
    adaptive_metric_parameters: AdaptiveMetricParameters = Field(default_factory=AdaptiveMetricParameters)
    enhanced_flow_metric_settings: EnhancedFlowMetricSettings = Field(default_factory=EnhancedFlowMetricSettings)
    adaptive_trade_idea_framework_settings: AdaptiveTradeIdeaFrameworkSettings # Required as per schema
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
        extra = 'forbid' # Corresponds to "additionalProperties": false in the root of the JSON schema

    # Optional: Method to load from config_v2_5.json (example)
    # @classmethod
    # def load_from_json(cls, path: Union[str, FilePath] = "config_v2_5.json") -> 'EOTSConfigV2_5':
    #     import json
    #     with open(path, 'r') as f:
    #         data = json.load(f)
    #     return cls(**data)

if __name__ == '__main__':
    # Example of creating a default config and validating
    try:
        # Note: DataFetcherSettings requires API keys, so direct instantiation of EOTSConfigV2_5
        # without providing them will fail if we don't mock them or load a real config.
        # For a simple test here, we can check individual components or a mocked main config.

        print("\n--- Testing DagAlphaCoeffs ---")
        dag_coeffs = DagAlphaCoeffs()
        print(dag_coeffs.json(indent=2))

        print("\n--- Testing SystemSettings (Default) ---")
        system_settings = SystemSettings()
        print(system_settings.json(indent=2))
        # Test enum for logging_level (Pydantic v2 would use Literal for schema enum)
        # SystemSettings(logging_level="INVALID_LEVEL") # This would raise ValidationError

        print("\n--- Testing DataManagementSettings (Default) ---")
        data_mgmt_settings = DataManagementSettings()
        print(data_mgmt_settings.json(indent=2))

        print("\n--- Testing CoefficientsSettings (Default) ---")
        coeffs_settings = CoefficientsSettings()
        print(coeffs_settings.json(indent=2))

        print("\n--- Testing LearningParams (Default) ---")
        learning_params = LearningParams()
        print(learning_params.json(indent=2))

        print("\n--- Testing SymbolDefaultOverrides (Default) ---")
        symbol_default_overrides = SymbolDefaultOverrides()
        print(symbol_default_overrides.json(indent=2))

        print("\n--- Testing SymbolSpecificOverrides (Default) ---")
        symbol_specific_overrides = SymbolSpecificOverrides()
        print(symbol_specific_overrides.json(indent=2))
        # Example with additional symbol
        symbol_specific_overrides_custom = SymbolSpecificOverrides(
            DEFAULT=SymbolDefaultOverrides(),
            SPY={"some_specific_setting_for_spy": True}
        )
        print(symbol_specific_overrides_custom.json(indent=2))


        print("\n--- Minimal EOTSConfigV2_5 (requires DataFetcherSettings) ---")
        # To fully test EOTSConfigV2_5, we need to provide required fields.
        # The schema lists system_settings, data_fetcher_settings, adaptive_trade_idea_framework_settings as required at the root.
        # However, system_settings and adaptive_trade_idea_framework_settings have default_factory.
        # So only data_fetcher_settings is truly required for instantiation if others default.

        min_config_data = {
            "data_fetcher_settings": {
                "convexvalue_api_key": "YOUR_CV_KEY",
                "tradier_api_key": "YOUR_TRADIER_KEY",
                "tradier_account_id": "YOUR_TRADIER_ACCOUNT_ID"
            }
            # adaptive_trade_idea_framework_settings will use default_factory
            # system_settings will use default_factory
        }
        config_instance = EOTSConfigV2_5(**min_config_data)
        print("Minimal EOTSConfigV2_5 instance created successfully.")
        # print(config_instance.json(indent=2, exclude_none=True)) # Can be very long

        # Test loading from a hypothetical full JSON (if one were available and matched)
        # config_from_file = EOTSConfigV2_5.load_from_json("path_to_config_v2_5.json")
        # print(config_from_file.json(indent=2))

        print("\nAll new config Pydantic models defined and basic tests passed.")

    except Exception as e:
        print(f"An error occurred during __main__ tests for config models: {e}")

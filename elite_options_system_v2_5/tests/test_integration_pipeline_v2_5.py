# tests/test_integration_pipeline_v2_5.py
# EOTS v2.5 - SENTRY-APPROVED CANONICAL SCRIPT

import pytest
import pandas as pd
import logging
from typing import Dict, Any
from datetime import datetime

# --- EOTS V2.5 Component Imports ---
from utils.config_manager_v2_5 import ConfigManagerV2_5
from core_analytics_engine.metrics_calculator_v2_5 import MetricsCalculatorV2_5
from core_analytics_engine.market_regime_engine_v2_5 import MarketRegimeEngineV2_5
from core_analytics_engine.signal_generator_v2_5 import SignalGeneratorV2_5
from data_management.performance_tracker_v2_5 import PerformanceTrackerV2_5
from core_analytics_engine.adaptive_trade_idea_framework_v2_5 import AdaptiveTradeIdeaFrameworkV2_5

# --- EOTS V2.5 Schema Imports ---
from data_models.eots_schemas_v2_5 import (
    ProcessedDataBundleV2_5,
    UnderlyingDataEnrichedV2_5,
    ATIFStrategyDirectivePayloadV2_5
)

# --- Test Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Test Fixtures ---
@pytest.fixture(scope="module")
def config_manager() -> ConfigManagerV2_5:
    """Provides a ConfigManager instance for the test suite."""
    return ConfigManagerV2_5(config_path='config/config_v2_5.json')

@pytest.fixture(scope="module")
def core_components(config_manager: ConfigManagerV2_5) -> Dict[str, Any]:
    """Initializes and provides all core v2.5 analytics components."""
    performance_tracker = PerformanceTrackerV2_5(config_manager)
    return {
        "metrics_calculator": MetricsCalculatorV2_5(config_manager),
        "market_regime_engine": MarketRegimeEngineV2_5(config_manager),
        "signal_generator": SignalGeneratorV2_5(config_manager),
        "performance_tracker": performance_tracker,
        "atif": AdaptiveTradeIdeaFrameworkV2_5(config_manager, performance_tracker, "TEST_TICKER"),
    }

def create_mock_data_bundle() -> ProcessedDataBundleV2_5:
    """
    Creates a mock ProcessedDataBundleV2_5 designed to trigger a specific
    regime (BULLISH_TREND_LOW_VOL) and a specific directional signal.
    """
    # 1. Craft Underlying Data to trigger "BULLISH_TREND_LOW_VOL" regime
    #    - GIB > dynamic threshold
    #    - IVSDH < dynamic threshold
    underlying_data = UnderlyingDataEnrichedV2_5(
        price=500.0,
        gib_oi_based_und=3.5e9,  # High positive GIB
        ivsdh_und_avg=-0.5,       # Low IVSDH
        vapi_fa_z_score_und=1.0,
        dwfd_z_score_und=1.5,     # Positive confirming flow
        tw_laf_z_score_und=1.2,     # Positive confirming trend
        arfi_overall_und_avg=0.6,
        arfi_overall_und_avg_mean=0.1,
        arfi_overall_und_avg_std=0.2,
        current_market_regime_v2_5="UNINITIALIZED", # This will be determined by the engine
        ticker_context_dict_v2_5={"iv_percentile_rank": 20} # Low IV context
    )

    # 2. Craft Strike-Level Data to trigger a directional signal
    #    - High SAI and high MSPI at a specific strike
    strike_data = {
        'strike': [490, 500, 510],
        'sai': [0.5, 0.95, 0.4],  # High SAI at the 500 strike
        'mspi': [-0.2, 0.85, 0.3], # High positive MSPI at the 500 strike
    }
    strike_df = pd.DataFrame(strike_data)

    return ProcessedDataBundleV2_5(
        underlying_symbol="TEST_TICKER",
        timestamp=datetime.now(),
        underlying_data_enriched=underlying_data,
        strike_level_data_with_metrics=strike_df
    )


class TestIntegrationPipelineV2_5:
    """
    End-to-end integration test for the full EOTS v2.5 analysis pipeline.
    This test ensures that all core components interface correctly and produce
    a valid, logical output from a known input.
    """

    def test_full_pipeline_execution(self, core_components: Dict[str, Any]):
        """
        Simulates a full analysis cycle from data bundle to ATIF directive.
        """
        logger.info("--- Starting EOTS v2.5 Full Pipeline Integration Test ---")

        # --- 1. SETUP: Create mock data and get components ---
        mock_bundle = create_mock_data_bundle()
        regime_engine = core_components["market_regime_engine"]
        signal_generator = core_components["signal_generator"]
        atif = core_components["atif"]

        # Define dynamic thresholds the orchestrator would normally provide
        dynamic_thresholds = {
            "gib_strong_pos_thresh": 2.0e9,
            "ivsdh_subdued_thresh": 0.0,
        }

        # --- 2. EXECUTION: Market Regime Engine ---
        logger.info("Step 2: Determining Market Regime...")
        determined_regime = regime_engine.determine_market_regime(mock_bundle, dynamic_thresholds)
        
        # --- ASSERTION: Regime ---
        expected_regime = "BULLISH_TREND_LOW_VOL"
        assert determined_regime == expected_regime
        logger.info(f"SUCCESS: Correctly determined regime: {determined_regime}")

        # Update the bundle with the correct regime, as the orchestrator would
        mock_bundle.underlying_data_enriched.current_market_regime_v2_5 = determined_regime

        # --- 3. EXECUTION: Signal Generator ---
        logger.info("Step 3: Generating Signals...")
        generated_signals = signal_generator.generate_all_signals(mock_bundle)
        
        # --- ASSERTION: Signals ---
        assert "directional" in generated_signals
        assert len(generated_signals["directional"]) > 0
        directional_signal = generated_signals["directional"][0]
        assert directional_signal.direction == "Bullish"
        assert directional_signal.strike_impacted == 500.0
        logger.info(f"SUCCESS: Generated expected directional signal: {directional_signal.signal_name}")

        # --- 4. EXECUTION: Adaptive Trade Idea Framework (ATIF) ---
        logger.info("Step 4: Generating Trade Idea from ATIF...")
        final_directive = atif.generate_trade_idea(mock_bundle, generated_signals)

        # --- ASSERTION: Final ATIF Directive ---
        assert final_directive is not None, "ATIF failed to produce a directive."
        assert isinstance(final_directive, ATIFStrategyDirectivePayloadV2_5)
        
        # Based on BULLISH_TREND_LOW_VOL and high conviction, config should select a debit spread or long call
        expected_strategy = "BullCallSpread" 
        assert final_directive.selected_strategy_type == expected_strategy
        assert final_directive.final_conviction_score_from_atif > 1.0 # Should have decent conviction
        
        logger.info(f"SUCCESS: ATIF produced a valid and expected directive: {final_directive.selected_strategy_type}")
        logger.info("--- EOTS v2.5 Full Pipeline Integration Test PASSED ---")
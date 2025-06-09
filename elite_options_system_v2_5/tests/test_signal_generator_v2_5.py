# tests/test_signal_generator_v2_5.py
# EOTS v2.5 - SENTRY-APPROVED CANONICAL SCRIPT
#
# This file contains unit tests for the refactored SignalGeneratorV2_5,
# verifying Pydantic integration and regime-aware logic.

import pytest
from datetime import datetime
import pandas as pd

# EOTS V2.5 Imports
from data_models.eots_schemas_v2_5 import (
    ProcessedDataBundleV2_5,
    ProcessedStrikeLevelMetricsV2_5,
    ProcessedUnderlyingAggregatesV2_5,
    SignalPayloadV2_5
)
from core_analytics_engine.signal_generator_v2_5 import SignalGeneratorV2_5

# Mock Objects and Test Fixtures
class MockConfigManager:
    """A mock ConfigManager for testing."""
    def __init__(self, config_dict):
        self._config = config_dict
    def get_setting(self, *keys, default=None):
        val = self._config
        for key in keys:
            val = val.get(key)
            if val is None: return default
        return val

@pytest.fixture
def base_config():
    """Provides a base configuration for the signal generator."""
    return {
        "system_settings": {"signal_activation": {"directional_signals": True, "v2_5_enhanced_signals": True}},
        "strategy_settings": {"thresholds": {"sai_high_conviction": 0.7, "vapi_fa_strong_z_score": 1.8}}
    }

@pytest.fixture
def signal_generator_instance(base_config):
    """Fixture to create an instance of the SignalGeneratorV2_5."""
    mock_cm = MockConfigManager(base_config)
    return SignalGeneratorV2_5(config_manager=mock_cm)

def create_test_bundle(regime: str, mspi: float, sai: float, vapi_z: float) -> ProcessedDataBundleV2_5:
    """Helper function to create a test data bundle with specific values."""
    strike_metrics = [
        ProcessedStrikeLevelMetricsV2_5(strike=100.0, a_mspi_und_summary_score=mspi, a_sai_und_avg=sai)
    ]
    underlying_data = ProcessedUnderlyingAggregatesV2_5(
        symbol="TEST", timestamp=datetime.now(), price=100.5,
        current_market_regime_v2_5=regime,
        vapi_fa_z_score_und=vapi_z,
        dwfd_z_score_und=0.0
    )
    return ProcessedDataBundleV2_5(
        strike_level_data_with_metrics=strike_metrics,
        underlying_data_enriched=underlying_data,
        processing_timestamp=datetime.now()
    )

# --- Unit Tests ---

def test_pydantic_output_contract(signal_generator_instance):
    """Test that the main method returns the correct Pydantic model structure."""
    bundle = create_test_bundle("REGIME_BULLISH_TREND", 0.8, 0.8, 0.0)
    signals = signal_generator_instance.generate_all_signals(bundle)
    
    assert isinstance(signals, dict)
    assert 'directional' in signals
    assert isinstance(signals['directional'], list)
    if signals['directional']:
        assert isinstance(signals['directional'][0], SignalPayloadV2_5)

def test_regime_modulates_directional_signal(signal_generator_instance):
    """Test that a bullish regime allows a bullish directional signal."""
    bundle = create_test_bundle("REGIME_BULLISH_TREND", 0.8, 0.8, 0.0)
    signals = signal_generator_instance.generate_all_signals(bundle)
    
    assert len(signals['directional']) > 0
    assert signals['directional'][0].signal_name == "MSPI_SAI_Bullish"
    assert signals['directional'][0].direction == "Bullish"

def test_no_signal_if_metrics_below_threshold(signal_generator_instance):
    """Test that no signal is generated if metrics do not meet thresholds."""
    bundle = create_test_bundle("REGIME_BULLISH_TREND", 0.4, 0.5, 0.0) # MSPI and SAI below thresholds
    signals = signal_generator_instance.generate_all_signals(bundle)
    
    assert len(signals['directional']) == 0

def test_tier3_vapi_signal_generation(signal_generator_instance):
    """Test the generation of a VAPI-FA signal based on its Z-score."""
    bundle = create_test_bundle("REGIME_ANY", 0.0, 0.0, 2.5) # VAPI Z-score is high
    signals = signal_generator_instance.generate_all_signals(bundle)
    
    assert len(signals['v2_5_enhanced']) > 0
    vapi_signal = signals['v2_5_enhanced'][0]
    assert "VAPI-FA" in vapi_signal.signal_name
    assert vapi_signal.direction == "Bullish"
    assert vapi_signal.strength_score == 2.5

def test_context_aware_signal_logic_placeholder():
    """
    Placeholder test to demonstrate how context-aware logic would be tested.
    This would be expanded to check, for example, that an 'EOD Hedging Pressure'
    signal is only generated when the TickerContext indicates the EOD session.
    """
    # 1. Create a bundle with context indicating "EOD_SESSION"
    # 2. Assert that the specific EOD signal IS generated.
    # 3. Create another bundle with context "LUNCH_LULL"
    # 4. Assert that the specific EOD signal IS NOT generated.
    assert True # Placeholder assertion
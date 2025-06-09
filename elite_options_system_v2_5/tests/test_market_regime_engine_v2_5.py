# tests/test_market_regime_engine_v2_5.py
# EOTS v2.5 - SENTRY-APPROVED CANONICAL SCRIPT
#
# This file contains comprehensive unit tests for the refactored
# MarketRegimeEngineV2_5, validating rule parsing, condition evaluation,
# and end-to-end regime classification.

import pytest
from datetime import datetime, time
import pandas as pd

# Mock EOTS Components for testing
from data_models.eots_schemas_v2_5 import ProcessedDataBundleV2_5, ProcessedStrikeLevelMetricsV2_5, ProcessedUnderlyingAggregatesV2_5
from core_analytics_engine.market_regime_engine_v2_5 import MarketRegimeEngineV2_5, ParsedRule

class MockConfigManager:
    """A mock ConfigManager that returns a predefined dictionary."""
    def __init__(self, config_dict):
        self._config = config_dict
    def get_setting(self, *keys, default=None):
        val = self._config
        for key in keys:
            val = val.get(key)
            if val is None: return default
        return val

@pytest.fixture
def sample_data_bundle():
    """Provides a sample ProcessedDataBundleV2_5 for testing."""
    strike_metrics = [
        ProcessedStrikeLevelMetricsV2_5(strike=95.0, mspi=-0.8, sdag_multiplicative=-0.2),
        ProcessedStrikeLevelMetricsV2_5(strike=100.0, mspi=0.1, sdag_multiplicative=0.05),
        ProcessedStrikeLevelMetricsV2_5(strike=105.0, mspi=0.9, sdag_multiplicative=0.3),
    ]
    underlying_data = ProcessedUnderlyingAggregatesV2_5(
        symbol="TEST", timestamp=datetime.now(), price=100.5,
        gib_oi_based_und=-60e9,
        vapi_fa_z_score_und=2.5
    )
    return ProcessedDataBundleV2_5(
        strike_level_data_with_metrics=strike_metrics,
        underlying_data_enriched=underlying_data,
        processing_timestamp=datetime(2025, 6, 10, 15, 30, 0) # Afternoon
    )

def test_rule_parsing_success():
    """Verify correct parsing of valid rule keys."""
    engine = MarketRegimeEngineV2_5(MockConfigManager({}))
    rule = engine._parse_rule_key("GIB_OI_based_Und_lt", "-50e9")
    assert isinstance(rule, ParsedRule)
    assert rule.metric_key == "GIB_OI_based_Und"
    assert rule.operator == "_lt"
    assert rule.target_value_str == "-50e9"
    assert rule.selector is None

    rule_atm = engine._parse_rule_key("mspi@ATM_gt", "0.5")
    assert rule_atm.metric_key == "mspi"
    assert rule_atm.selector == "@ATM"
    assert rule_atm.operator == "_gt"

    rule_agg = engine._parse_rule_key("sdag_multiplicative[AGG=mean]_lt", "-0.1")
    assert rule_agg.metric_key == "sdag_multiplicative"
    assert rule_agg.aggregator == "mean"
    assert rule_agg.operator == "_lt"
    
    rule_time = engine._parse_rule_key("time_is_final_hour_eq", "true")
    assert rule_time.is_special_condition is True
    assert rule_time.metric_key == "time_is_final_hour_eq"

def test_rule_parsing_failure():
    """Test that malformed rule keys are handled gracefully."""
    engine = MarketRegimeEngineV2_5(MockConfigManager({}))
    rule = engine._parse_rule_key("GibOiBasedUndlt", "-50e9") # Missing operator underscores
    assert rule is None
    rule2 = engine._parse_rule_key("mspi@_gt", "0.5") # Invalid selector
    assert rule2 is None

def test_evaluator_resolve_metric(sample_data_bundle):
    """Test the _RuleConditionEvaluator's ability to resolve metric values."""
    engine = MarketRegimeEngineV2_5(MockConfigManager({}))
    evaluator = engine._RuleConditionEvaluator(sample_data_bundle, engine)
    
    # Test simple underlying lookup
    rule_gib = ParsedRule("", "gib_oi_based_und", "_lt", "")
    assert evaluator._resolve_metric_value(rule_gib) == -60e9
    
    # Test aggregator
    rule_sdag_mean = ParsedRule("", "sdag_multiplicative", "_lt", "", aggregator="mean")
    mean_val = evaluator._resolve_metric_value(rule_sdag_mean)
    assert np.isclose(mean_val, (-0.2 + 0.05 + 0.3) / 3)

    # Test selector
    rule_mspi_atm = ParsedRule("", "mspi", "_gt", "", selector="@ATM")
    atm_val = evaluator._resolve_metric_value(rule_mspi_atm)
    assert np.isclose(atm_val, 0.1) # Strike 100 is closest to price 100.5

def test_regime_classification_scenario_1(sample_data_bundle):
    """Test scenario: GIB is very low, triggering a specific regime."""
    mock_config = {
        "market_regime_engine_settings": {
            "default_regime": "REGIME_DEFAULT",
            "regime_evaluation_order": ["REGIME_EXTREME_NEG_GIB"],
            "regime_rules": {
                "REGIME_EXTREME_NEG_GIB": {"GIB_OI_based_Und_lt": "-55e9"}
            }
        }
    }
    engine = MarketRegimeEngineV2_5(MockConfigManager(mock_config))
    regime = engine.determine_market_regime(sample_data_bundle)
    assert regime == "REGIME_EXTREME_NEG_GIB"
    
def test_regime_classification_scenario_2(sample_data_bundle):
    """Test scenario: High VAPI and time condition trigger a different regime."""
    mock_config = {
        "market_regime_engine_settings": {
            "default_regime": "REGIME_DEFAULT",
            "regime_evaluation_order": ["REGIME_EXTREME_NEG_GIB", "REGIME_VAPI_SURGE_AFTERNOON"],
            "regime_rules": {
                "REGIME_EXTREME_NEG_GIB": {"GIB_OI_based_Und_lt": "-70e9"}, # This rule will fail
                "REGIME_VAPI_SURGE_AFTERNOON": {
                    "vapi_fa_z_score_und_gt": "2.0",
                    "time_is_final_hour_eq": "true"
                }
            },
            "time_of_day_definitions": {"final_hour_start_time": "15:00:00"}
        }
    }
    engine = MarketRegimeEngineV2_5(MockConfigManager(mock_config))
    # data_bundle processing_timestamp is 15:30, so time condition should pass
    regime = engine.determine_market_regime(sample_data_bundle)
    assert regime == "REGIME_VAPI_SURGE_AFTERNOON"

def test_regime_classification_fallback_to_default(sample_data_bundle):
    """Test scenario: No rules match, falls back to the default regime."""
    mock_config = {
        "market_regime_engine_settings": {
            "default_regime": "REGIME_FALLBACK_SUCCESS",
            "regime_evaluation_order": ["REGIME_NON_MATCHING"],
            "regime_rules": {
                "REGIME_NON_MATCHING": {"gib_oi_based_und_gt": "0"} # Will fail
            }
        }
    }
    engine = MarketRegimeEngineV2_5(MockConfigManager(mock_config))
    regime = engine.determine_market_regime(sample_data_bundle)
    assert regime == "REGIME_FALLBACK_SUCCESS"
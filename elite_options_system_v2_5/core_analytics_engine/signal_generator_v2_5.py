# core_analytics_engine/signal_generator_v2_5.py
# EOTS v2.5 - S-GRADE PRODUCTION HARDENED ARTIFACT

import logging
import uuid
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

from data_models.eots_schemas_v2_5 import ProcessedDataBundleV2_5, SignalPayloadV2_5, ProcessedUnderlyingAggregatesV2_5
from utils.config_manager_v2_5 import ConfigManagerV2_5

logger = logging.getLogger(__name__)
EPSILON = 1e-9

class SignalGeneratorV2_5:
    """
    Generates discrete, scored trading signals based on a fully processed data bundle.
    This hardened version ensures production-grade resilience through robust error
    handling and employs vectorized logic where possible for maximum performance.
    """

    def __init__(self, config_manager: ConfigManagerV2_5):
        self.logger = logger.getChild(self.__class__.__name__)
        self.config_manager = config_manager
        
        # Fetch signal generation settings, with a default to avoid crashes
        self.settings = self.config_manager.get_setting("signal_generator_settings_v2_5", default={})
        self.activation = self.settings.get("signal_activation", {"EnableAllSignals": True})
        self.logger.info("SignalGeneratorV2_5 initialized.")

    def generate_all_signals(self, bundle: ProcessedDataBundleV2_5) -> Dict[str, List[SignalPayloadV2_5]]:
        """
        Orchestrates the generation of all signal categories, passing the full data
        bundle to the specialized methods.
        """
        if not isinstance(bundle, ProcessedDataBundleV2_5):
            self.logger.error("generate_all_signals received an invalid data bundle type.")
            return {}

        regime = getattr(bundle.underlying_data_enriched, 'current_market_regime_v2_5', 'default')
        signals: Dict[str, List[SignalPayloadV2_5]] = {
            'directional': [], 'volatility': [], 'time_decay': [],
            'complex': [], 'v2_5_enhanced_flow': []
        }
        
        enable_all = self.activation.get("EnableAllSignals", False)

        # Create DataFrames once for performance
        df_strike = pd.DataFrame([s.model_dump() for s in bundle.strike_level_data_with_metrics])
        if not df_strike.empty:
            df_strike.set_index('strike', inplace=True, drop=False)

        if enable_all or self.activation.get("v2_5_enhanced_flow_signals", False):
            signals['v2_5_enhanced_flow'] = self._generate_v2_5_enhanced_flow_signals(bundle.underlying_data_enriched, regime)
        
        # Add other signal categories here as their logic is developed
        # if enable_all or self.activation.get("directional_signals", False):
        #     signals['directional'] = self._generate_directional_signals(bundle, regime, df_strike)

        self.logger.info(f"Signal generation complete. Found {sum(len(v) for v in signals.values())} total signals.")
        return signals

    def _generate_v2_5_enhanced_flow_signals(self, und_data: ProcessedUnderlyingAggregatesV2_5, regime: str) -> List[SignalPayloadV2_5]:
        """Generates underlying-level signals from Tier 3 flow metrics."""
        signals = []
        try:
            params = self.settings.get("v2_5_enhanced_flow_signals", {})
            regime_params = params.get("regime_thresholds", {}).get(regime, params.get("default_thresholds", {}))

            # VAPI-FA Signal
            vapi_z = und_data.vapi_fa_z_score_und
            vapi_thresh = regime_params.get("vapi_fa_z_thresh", 2.0)
            if vapi_z is not None and abs(vapi_z) > vapi_thresh:
                signals.append(self._create_signal_payload(
                    und_data, "VAPI-FA_Momentum_Surge", vapi_z, regime,
                    details={"vapi_z_score": vapi_z, "threshold": vapi_thresh}
                ))
            
            # DWFD Signal
            dwfd_z = und_data.dwfd_z_score_und
            dwfd_thresh = regime_params.get("dwfd_z_thresh", 2.0)
            if dwfd_z is not None and abs(dwfd_z) > dwfd_thresh:
                signals.append(self._create_signal_payload(
                    und_data, "DWFD_Smart_Money_Flow", dwfd_z, regime,
                    details={"dwfd_z_score": dwfd_z, "threshold": dwfd_thresh}
                ))
            
            # TW-LAF Signal
            tw_laf_z = und_data.tw_laf_z_score_und
            tw_laf_thresh = regime_params.get("tw_laf_z_thresh", 1.5)
            if tw_laf_z is not None and abs(tw_laf_z) > tw_laf_thresh:
                signals.append(self._create_signal_payload(
                    und_data, "TW-LAF_Sustained_Trend", tw_laf_z, regime,
                    details={"tw_laf_z_score": tw_laf_z, "threshold": tw_laf_thresh}
                ))

            return signals
        except (KeyError, AttributeError, TypeError) as e:
            self.logger.error(f"Failed to generate v2.5 enhanced flow signals due to data/config issue: {e}", exc_info=True)
            return []

    def _create_signal_payload(self, und_data: ProcessedUnderlyingAggregatesV2_5, name: str, strength: float, regime: str, strike: Optional[float] = None, details: Optional[Dict] = None) -> SignalPayloadV2_5:
        """Helper to construct the SignalPayloadV2_5 Pydantic model."""
        direction = "Neutral"
        if strength > EPSILON:
            direction = "Bullish"
        elif strength < -EPSILON:
            direction = "Bearish"

        # Determine signal type based on name
        signal_type = "Complex"
        if any(k in name.upper() for k in ["VAPI", "DWFD", "TW-LAF", "FLOW"]): 
            signal_type = "Flow_Momentum"
        elif any(k in name.upper() for k in ["MSPI", "SDAG", "DIRECTIONAL"]): 
            signal_type = "Directional"
        elif any(k in name.upper() for k in ["VOLATILITY", "VRI"]): 
            signal_type = "Volatility"
        elif any(k in name.upper() for k in ["DECAY", "PIN", "TDPI"]): 
            signal_type = "Time_Decay"

        return SignalPayloadV2_5(
            signal_id=f"sig_{uuid.uuid4().hex[:8]}",
            signal_name=name,
            symbol=und_data.symbol,
            timestamp=datetime.now(),
            signal_type=signal_type,
            direction=direction,
            strength_score=np.clip(strength, -5.0, 5.0), # Clip score to a reasonable range
            strike_impacted=strike,
            regime_at_signal_generation=regime,
            supporting_metrics=details or {}
        )
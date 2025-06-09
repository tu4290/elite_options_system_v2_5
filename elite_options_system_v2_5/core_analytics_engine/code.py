# core_analytics_engine/trade_parameter_optimizer_v2_5.py
# EOTS v2.5 - S-GRADE PRODUCTION HARDENED ARTIFACT

import logging
import uuid
from typing import Optional, Any, Tuple, List, Dict
from datetime import datetime
import pandas as pd
import numpy as np

from utils.config_manager_v2_5 import ConfigManagerV2_5
from data_models.eots_schemas_v2_5 import (
    ATIFStrategyDirectivePayloadV2_5, ActiveRecommendationPayloadV2_5,
    KeyLevelsDataV2_5, ProcessedDataBundleV2_5
)

logger = logging.getLogger(__name__)
EPSILON = 1e-9

class TradeParameterOptimizerV2_5:
    """
    Translates a strategic directive from the ATIF into precise, executable
    trade parameters by selecting the optimal option contract and calculating
    adaptive entry, stop-loss, and profit target levels.
    """

    def __init__(self, config_manager: ConfigManagerV2_5):
        self.logger = logger.getChild(self.__class__.__name__)
        self.config_manager = config_manager
        self.settings = self.config_manager.get_setting("strategy_settings", "targets", default={})
        if not self.settings:
             raise ValueError("TPO settings ('strategy_settings.targets') not found in configuration.")
        self.logger.info("TradeParameterOptimizerV2_5 Initialized.")


    def optimize_parameters_for_directive(
        self,
        directive: ATIFStrategyDirectivePayloadV2_5,
        processed_data: ProcessedDataBundleV2_5,
        key_levels: KeyLevelsDataV2_5
    ) -> Optional[ActiveRecommendationPayloadV2_5]:
        """Main orchestration method to generate a fully parameterized trade recommendation."""
        if not all(isinstance(arg, (ATIFStrategyDirectivePayloadV2_5, ProcessedDataBundleV2_5, KeyLevelsDataV2_5)) for arg in [directive, processed_data, key_levels]):
            self.logger.error("Invalid input types to optimize_parameters_for_directive. Aborting.")
            return None

        try:
            options_chain_df = pd.DataFrame(processed_data.options_data_with_metrics)
            if options_chain_df.empty:
                self.logger.warning("Options chain DataFrame is empty. Cannot select contract.")
                return None

            # 1. Select the optimal option contract(s)
            selected_contracts = self._select_optimal_contracts(directive, options_chain_df)
            if not selected_contracts:
                self.logger.warning(f"Could not find a suitable contract for directive: {directive.selected_strategy_type}")
                return None

            # 2. Calculate adaptive stop-loss and profit targets for the underlying
            trade_bias = "Bullish" if directive.final_conviction_score_from_atif > 0 else "Bearish"
            atr = processed_data.underlying_data_enriched.atr_und or (processed_data.underlying_data_enriched.price * 0.01) # Fallback ATR
            
            und_sl, und_t1, und_t2 = self._calculate_sl_and_targets(
                trade_bias=trade_bias,
                entry_price_und=processed_data.underlying_data_enriched.price,
                key_levels=key_levels,
                atr=atr
            )

            # 3. Construct the final recommendation payload
            return self._construct_recommendation_payload(
                directive, processed_data, selected_contracts, und_sl, und_t1, und_t2, trade_bias
            )
            
        except Exception as e:
            self.logger.critical(f"Unhandled exception during parameter optimization: {e}", exc_info=True)
            return None

    def _select_optimal_contracts(self, directive: ATIFStrategyDirectivePayloadV2_5, chain_df: pd.DataFrame) -> List[Dict]:
        """Selects the best option contracts from the chain based on ATIF directives."""
        # This is a simplified version. A real implementation would be more complex for spreads.
        if directive.selected_strategy_type in ["LongCall", "LongPut"]:
            is_call = directive.selected_strategy_type == "LongCall"
            
            candidates = chain_df[
                (chain_df['opt_kind'] == ('call' if is_call else 'put')) &
                (chain_df['dte_calc'] >= directive.target_dte_min) &
                (chain_df['dte_calc'] <= directive.target_dte_max) &
                (chain_df['delta'].abs().between(directive.target_delta_long_leg_min, directive.target_delta_long_leg_max))
            ].copy()

            if candidates.empty: return []

            # Add liquidity scoring
            candidates['liquidity_score'] = candidates['oi'] + candidates['volm'] * 5 # Simple score
            best_contract = candidates.loc[candidates['liquidity_score'].idxmax()]
            return [best_contract.to_dict()]
        
        # Placeholder for spread logic
        self.logger.warning(f"Contract selection for strategy '{directive.selected_strategy_type}' is not yet implemented.")
        return []

    def _calculate_sl_and_targets(self, trade_bias: str, entry_price_und: float, key_levels: KeyLevelsDataV2_5, atr: float) -> Tuple[float, float, Optional[float]]:
        """Calculates SL and multiple TP levels for the underlying based on ATR and key levels."""
        sl_mult = self.settings.get("target_atr_stop_loss_multiplier", 1.5)
        t1_mult = self.settings.get("t1_mult_no_sr", 1.0)
        t2_mult = self.settings.get("t2_mult_no_sr", 2.0)

        if trade_bias == "Bullish":
            stop_loss = entry_price_und - (atr * sl_mult)
            # Refine SL with support levels
            supports = sorted([lvl.level_price for lvl in key_levels.supports if lvl.level_price < entry_price_und], reverse=True)
            if supports:
                stop_loss = min(stop_loss, supports[0] - (atr * 0.1)) # Place SL just below highest support

            target_1 = entry_price_und + (atr * t1_mult)
            resistances = sorted([lvl.level_price for lvl in key_levels.resistances if lvl.level_price > entry_price_und])
            if resistances:
                target_1 = min(target_1, resistances[0]) # T1 is the first key resistance
            target_2 = target_1 + (atr * (t2_mult - t1_mult)) if resistances else None

        else: # Bearish
            stop_loss = entry_price_und + (atr * sl_mult)
            resistances = sorted([lvl.level_price for lvl in key_levels.resistances if lvl.level_price > entry_price_und])
            if resistances:
                stop_loss = max(stop_loss, resistances[0] + (atr * 0.1))

            target_1 = entry_price_und - (atr * t1_mult)
            supports = sorted([lvl.level_price for lvl in key_levels.supports if lvl.level_price < entry_price_und], reverse=True)
            if supports:
                target_1 = max(target_1, supports[0])
            target_2 = target_1 - (atr * (t2_mult - t1_mult)) if supports else None
            
        return stop_loss, target_1, target_2

    def _construct_recommendation_payload(self, directive, processed_data, contracts, und_sl, und_t1, und_t2, bias) -> ActiveRecommendationPayloadV2_5:
        """Constructs and validates the final ActiveRecommendationPayloadV2_5 object."""
        # Simplified for single-leg options
        contract = contracts[0]
        und_entry = processed_data.underlying_data_enriched.price
        opt_entry = contract.get('price') or ((contract.get('bid_price', 0) + contract.get('ask_price', 0)) / 2)
        opt_delta = contract.get('delta', 0.5)

        # Translate underlying SL/TP to option premium SL/TP using delta
        opt_sl = opt_entry - abs(und_sl - und_entry) * abs(opt_delta)
        opt_t1 = opt_entry + abs(und_t1 - und_entry) * abs(opt_delta)
        opt_t2 = (opt_entry + abs(und_t2 - und_entry) * abs(opt_delta)) if und_t2 else None

        return ActiveRecommendationPayloadV2_5(
            recommendation_id=f"rec_{uuid.uuid4().hex[:8]}",
            symbol=processed_data.underlying_data_enriched.symbol,
            timestamp_issued=datetime.now(),
            strategy_type=directive.selected_strategy_type,
            selected_option_details=contracts,
            trade_bias=bias,
            entry_price_initial=opt_entry,
            stop_loss_initial=opt_sl,
            target_1_initial=opt_t1,
            target_2_initial=opt_t2,
            entry_price_actual=None, # To be filled on execution
            stop_loss_current=opt_sl,
            target_1_current=opt_t1,
            target_2_current=opt_t2,
            target_rationale=f"SL/TP based on ATR and Key Levels. Underlying SL: {und_sl:.2f}, T1: {und_t1:.2f}",
            status="ACTIVE_NEW",
            atif_conviction_score_at_issuance=directive.final_conviction_score_from_atif,
            triggering_signals_summary=str(directive.supportive_rationale_components),
            regime_at_issuance=processed_data.underlying_data_enriched.current_market_regime_v2_5
        )
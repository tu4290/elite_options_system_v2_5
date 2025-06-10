# core_analytics_engine/its_orchestrator_v2_5.py
# EOTS v2.5 - S-GRADE, AUTHORITATIVE MASTER ORCHESTRATOR

import logging
import asyncio
from datetime import datetime
from typing import Dict, Optional, Any, List
import pandas as pd
from pydantic import ValidationError as PydanticValidationError

# EOTS V2.5 Data Contracts
from ..data_models.eots_schemas_v2_5 import (
    UnprocessedDataBundleV2_5, ProcessedDataBundleV2_5, FinalAnalysisBundleV2_5, EOTSConfigV2_5,
    RawUnderlyingDataCombinedV2_5, KeyLevelsDataV2_5, SignalPayloadV2_5,
    ATIFStrategyDirectivePayloadV2_5, ActiveRecommendationPayloadV2_5, TradeParametersV2_5
)

# EOTS V2.5 Core Components
from utils.config_manager_v2_5 import ConfigManagerV2_5
from data_management.convexvalue_data_fetcher_v2_5 import ConvexValueDataFetcherV2_5
from data_management.tradier_data_fetcher_v2_5 import TradierDataFetcherV2_5
from data_management.initial_processor_v2_5 import InitialDataProcessorV2_5
from data_management.historical_data_manager_v2_5 import HistoricalDataManagerV2_5
from core_analytics_engine.metrics_calculator_v2_5 import MetricsCalculatorV2_5
from core_analytics_engine.market_regime_engine_v2_5 import MarketRegimeEngineV2_5
from core_analytics_engine.signal_generator_v2_5 import SignalGeneratorV2_5
from core_analytics_engine.adaptive_trade_idea_framework_v2_5 import AdaptiveTradeIdeaFrameworkV2_5
from core_analytics_engine.trade_parameter_optimizer_v2_5 import TradeParameterOptimizerV2_5
# NOTE: TickerContextAnalyzer and KeyLevelIdentifier would be imported here once created.

logger = logging.getLogger(__name__)

class ITSOrchestratorV2_5:
    """
    Main orchestrator for the EOTS v2.5. Controls the entire analysis pipeline
    from data ingestion to final output, enforcing strict data contracts.
    """
    def __init__(self, config_manager: ConfigManagerV2_5, historical_data_manager: HistoricalDataManagerV2_5, performance_tracker: Any,
                 metrics_calculator: MetricsCalculatorV2_5, initial_processor: InitialDataProcessorV2_5,
                 market_regime_engine: MarketRegimeEngineV2_5, signal_generator: SignalGeneratorV2_5,
                 adaptive_trade_idea_framework: AdaptiveTradeIdeaFrameworkV2_5,
                 trade_parameter_optimizer: TradeParameterOptimizerV2_5):
        """Initializes the orchestrator with all subordinate components injected."""
        self.logger = logger.getChild(self.__class__.__name__)
        self.logger.info("Initializing ITSOrchestratorV2_5...")

        # Store injected dependencies
        self.config_manager = config_manager
        self.historical_data_manager = historical_data_manager
        self.performance_tracker = performance_tracker
        self.metrics_calculator = metrics_calculator
        self.initial_processor = initial_processor
        self.market_regime_engine = market_regime_engine
        self.signal_generator = signal_generator
        self.adaptive_trade_idea_framework = adaptive_trade_idea_framework
        self.trade_parameter_optimizer = trade_parameter_optimizer

        # Instantiate components not requiring complex dependency chains
        self.convexvalue_fetcher = ConvexValueDataFetcherV2_5(config_manager)
        self.tradier_fetcher = TradierDataFetcherV2_5(config_manager)
        # self.key_level_identifier = KeyLevelIdentifierV2_5(config_manager)
        # self.ticker_context_analyzer = TickerContextAnalyzerV2_5(config_manager)

        # State Management
        self.active_recommendations: List[ActiveRecommendationPayloadV2_5] = []
        self.current_symbol_being_managed: Optional[str] = None
        self._latest_bundle: Optional[FinalAnalysisBundleV2_5] = None
        
        self.logger.info("ITSOrchestratorV2_5 initialized successfully with all components.")

    def get_latest_analysis_bundle(self) -> Optional[FinalAnalysisBundleV2_5]:
        """Public method for the dashboard to retrieve the last computed bundle."""
        return self._latest_bundle

    def run_full_analysis_cycle(self, symbol: str) -> Optional[FinalAnalysisBundleV2_5]:
        """Executes the entire end-to-end analysis pipeline for a given symbol."""
        self.logger.info(f"---|> Starting Full Analysis Cycle for '{symbol}'...")
        
        if self.current_symbol_being_managed != symbol:
            self.logger.info(f"Symbol context changed to '{symbol}'. Resetting active recommendations.")
            self.active_recommendations = []
            self.current_symbol_being_managed = symbol

        try:
            # 1. DATA FUSION: Fetch raw data from all sources
            raw_data_bundle = self._fetch_and_fuse_data(symbol)
            if not raw_data_bundle: return None

            # 2. METRIC CALCULATION: Process raw data and compute all metrics
            processed_data_bundle = self.initial_processor.process_data_and_calculate_metrics(raw_data_bundle)

            # --- Placeholder for future components ---
            # 3. CONTEXTUALIZATION: Get Ticker-Specific Context
            # ticker_context = self.ticker_context_analyzer.get_context(...)
            # processed_data_bundle.underlying_data_enriched.ticker_context_dict_v2_5 = ticker_context
            
            # 4. DYNAMIC THRESHOLDS: Resolve dynamic thresholds for this cycle
            # dynamic_thresholds = self._resolve_dynamic_thresholds(symbol)
            # processed_data_bundle.underlying_data_enriched.dynamic_thresholds = dynamic_thresholds
            
            # Create DataFrames once for performance
            df_strike = pd.DataFrame(processed_data_bundle.strike_level_data_with_metrics)
            if not df_strike.empty: df_strike.set_index('strike', inplace=True, drop=False)
            df_chain = pd.DataFrame(processed_data_bundle.options_data_with_metrics)

            # 5. MARKET REGIME: Classify the market state
            market_regime = self.market_regime_engine.determine_market_regime(
                und_data=processed_data_bundle.underlying_data_enriched,
                df_strike=df_strike,
                df_chain=df_chain
            )
            processed_data_bundle.underlying_data_enriched.current_market_regime_v2_5 = market_regime
            self.logger.info(f"Market Regime classified as: {market_regime}")

            # 6. KEY LEVELS: Identify all key support/resistance levels
            # key_levels = self.key_level_identifier.identify_all_levels(...)
            from data_models.eots_schemas_v2_5 import KeyLevelsDataV2_5 # Temp import
            key_levels = KeyLevelsDataV2_5(timestamp=datetime.now()) # Placeholder

            # 7. SIGNAL GENERATION: Generate scored signals
            signals = self.signal_generator.generate_all_signals(processed_data_bundle)

            # 8. ATIF - NEW IDEAS: Formulate new trade directives
            new_directives = self.adaptive_trade_idea_framework.generate_trade_directives(
                processed_data=processed_data_bundle,
                scored_signals=signals,
                key_levels=key_levels
            )

            # 9. TPO: Optimize parameters for new ideas
            newly_parameterized_recos = [
                self.trade_parameter_optimizer.optimize_parameters_for_directive(d, processed_data_bundle, key_levels)
                for d in new_directives
            ]
            newly_parameterized_recos = [reco for reco in newly_parameterized_recos if reco is not None]

            # 10. ATIF - MANAGE ACTIVE: Manage lifecycle of existing recommendations
            self._manage_active_recommendations(processed_data_bundle.underlying_data_enriched.price)
            
            # 11. STATE UPDATE: Add new recommendations to the active list
            self.active_recommendations.extend(newly_parameterized_recos)

            # 12. BUNDLE FINALIZATION: Assemble the final output
            final_bundle = FinalAnalysisBundleV2_5(
                processed_data_bundle=processed_data_bundle,
                key_levels_data_v2_5=key_levels,
                scored_signals_v2_5=signals,
                active_recommendations_v2_5=self.active_recommendations,
                bundle_timestamp=datetime.now(),
                target_symbol=symbol,
                system_status_messages=[]
            )
            
            self._latest_bundle = final_bundle # Cache the latest bundle
            self.logger.info(f"---|> Analysis Cycle for '{symbol}' COMPLETE. <|---")
            return final_bundle

        except Exception as e:
            self.logger.critical(f"FATAL ERROR during analysis cycle for '{symbol}': {e}", exc_info=True)
            return None

    def _fetch_and_fuse_data(self, symbol: str) -> Optional[UnprocessedDataBundleV2_5]:
        """Fetches data from all sources and fuses it into a single bundle."""
        async def _async_fusion_fetch():
            async with self.tradier_fetcher as tradier_session:
                # Run CV and Tradier fetches concurrently
                cv_task = self.convexvalue_fetcher.fetch_chain_and_underlying(symbol)
                tradier_task = tradier_session.fetch_underlying_quote(symbol)
                results = await asyncio.gather(cv_task, tradier_task, return_exceptions=True)
            return results
        
        try:
            (cv_result, tradier_result) = asyncio.run(_async_fusion_fetch())

            if isinstance(cv_result, Exception) or cv_result[1] is None:
                raise IOError(f"Critical data failure from ConvexValue: {cv_result}")
            cv_options, cv_underlying = cv_result
            
            tradier_ohlcv = {}
            if isinstance(tradier_result, Exception):
                self.logger.warning(f"Tradier fetch failed: {tradier_result}. OHLCV data will be missing.")
            elif tradier_result:
                tradier_ohlcv = tradier_result
            
            # Fuse data: Start with CV and overwrite/add with Tradier data
            final_underlying_dict = cv_underlying.model_dump()
            final_underlying_dict.update(tradier_ohlcv)
            
            final_underlying_model = RawUnderlyingDataCombinedV2_5(**final_underlying_dict)
            return UnprocessedDataBundleV2_5(
                options_contracts=cv_options or [],
                underlying_data=final_underlying_model,
                fetch_timestamp=datetime.now(),
                errors=[]
            )
        except (IOError, PydanticValidationError) as e:
            self.logger.error(f"Data fusion failed for {symbol}: {e}", exc_info=True)
            return None
    
    def _manage_active_recommendations(self, current_price: float):
        """Manages the lifecycle of active recommendations."""
        if not self.active_recommendations:
            return

        recos_to_keep = []
        for reco in self.active_recommendations:
            # Check for standard SL/TP hits first
            if reco.status.startswith("ACTIVE"):
                management_directive = self.adaptive_trade_idea_framework.get_management_directive(reco, current_price)
                if management_directive and management_directive.action == "EXIT":
                    reco.status = f"EXITED_ATIF_{management_directive.reason}"
                    reco.exit_timestamp = datetime.now()
                    reco.exit_price = current_price # Approximate exit price
                    self.performance_tracker.record_recommendation_outcome(reco)
                    self.logger.info(f"ATIF directed EXIT for {reco.recommendation_id} due to {management_directive.reason}.")
                    continue # Do not add back to active list
                # Logic for other directives (ADJUST_STOPLOSS, etc.) would go here
            
            recos_to_keep.append(reco)
        
        self.active_recommendations = recos_to_keep
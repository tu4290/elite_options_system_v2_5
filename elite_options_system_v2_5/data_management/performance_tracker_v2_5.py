# data_management/performance_tracker_v2_5.py
# EOTS v2.5 - SENTRY-APPROVED CANONICAL SCRIPT

import logging
import os
from typing import List, Dict, Any
import pandas as pd
from filelock import FileLock # For safe multi-process file writes

# EOTS V2.5 Imports
from data_models.eots_schemas_v2_5 import ActiveRecommendationPayloadV2_5

class PerformanceTrackerV2_5:
    """
    Tracks the historical performance of signals and trade setups to provide
    data-driven weights and modifiers to the EOTS analytics engine. This implementation
    uses a local CSV file-based storage system for persistence.
    """

    def __init__(self, config_manager: Any):
        """
        Initializes the PerformanceTracker.

        Args:
            config_manager (Any): The system's configuration manager.
        """
        self.logger = logging.getLogger(__name__).getChild(self.__class__.__name__)
        self.config_manager = config_manager
        self.settings = self.config_manager.get_setting("performance_tracker_settings_v2_5", {})
        
        # Resolve data directory path and ensure it exists
        self.performance_data_dir = self.config_manager.get_resolved_path("performance_tracker_settings_v2_5", "performance_data_directory")
        if not self.performance_data_dir:
            raise ValueError("Performance data directory not specified in configuration.")
        
        os.makedirs(self.performance_data_dir, exist_ok=True)
        self.logger.info(f"PerformanceTrackerV2_5 initialized. Data will be stored in: {self.performance_data_dir}")

    def get_performance_weights_for_signals(self, symbol: str, regime: str, signals: List[str]) -> Dict[str, float]:
        """
        Retrieves performance-based weights for a list of signals based on historical win rates.
        If a signal has no history, a neutral weight of 0.5 is returned.

        Args:
            symbol (str): The underlying symbol.
            regime (str): The current market regime.
            signals (List[str]): A list of signal names to retrieve weights for.

        Returns:
            Dict[str, float]: A dictionary mapping each signal name to its historical win rate.
        """
        filepath = os.path.join(self.performance_data_dir, f"{symbol}_performance.csv")
        default_weight = 0.5  # Neutral win rate for signals with no history
        weights = {signal: default_weight for signal in signals}

        if not os.path.exists(filepath):
            return weights

        try:
            df = pd.read_csv(filepath)
            regime_df = df[df['regime_at_issuance'] == regime]
            if regime_df.empty:
                return weights

            for signal_name in signals:
                # Filter for trades where this signal was part of the trigger
                signal_df = regime_df[regime_df['triggering_signals_summary'].str.contains(signal_name, na=False)]
                if not signal_df.empty:
                    # Calculate win rate for this specific signal in this regime
                    win_rate = signal_df['win'].mean()
                    weights[signal_name] = win_rate
            
            self.logger.debug(f"Calculated performance weights for {symbol} in regime {regime}: {weights}")
            return weights
        except Exception as e:
            self.logger.error(f"Failed to read or process performance data for {symbol}: {e}", exc_info=True)
            return {signal: default_weight for signal in signals} # Return defaults on error

    def get_historical_performance_for_setup(self, symbol: str, regime: str, dominant_bias_category: str) -> Dict[str, Any]:
        """
        Retrieves the historical success rate and trade count for an analogous trade setup.

        Args:
            symbol (str): The underlying symbol.
            regime (str): The current market regime.
            dominant_bias_category (str): The assessed directional bias (e.g., "Bullish", "Bearish").

        Returns:
            Dict[str, Any]: A dictionary containing 'win_rate' and 'trade_count'.
        """
        filepath = os.path.join(self.performance_data_dir, f"{symbol}_performance.csv")
        default_result = {'win_rate': 0.5, 'trade_count': 0}

        if not os.path.exists(filepath):
            return default_result

        try:
            df = pd.read_csv(filepath)
            # Filter for analogous setups by regime and trade bias
            filtered_df = df[(df['regime_at_issuance'] == regime) & (df['trade_bias'] == dominant_bias_category)]
            
            if filtered_df.empty:
                return default_result
            
            win_rate = filtered_df['win'].mean()
            trade_count = len(filtered_df)
            
            result = {'win_rate': win_rate, 'trade_count': trade_count}
            self.logger.debug(f"Found historical performance for {symbol}/{regime}/{dominant_bias_category}: {result}")
            return result
        except Exception as e:
            self.logger.error(f"Failed to get historical setup performance for {symbol}: {e}", exc_info=True)
            return default_result

    def record_recommendation_outcome(self, trade_outcome: ActiveRecommendationPayloadV2_5) -> None:
        """
        Records the final outcome of a trade to a persistent CSV data store.
        This method uses a file lock to ensure safe concurrent writes.

        Args:
            trade_outcome (ActiveRecommendationPayloadV2_5): A Pydantic model containing the
                                                             outcome data of a closed trade.
        """
        try:
            symbol = trade_outcome.symbol
            filepath = os.path.join(self.performance_data_dir, f"{symbol}_performance.csv")
            lock_path = filepath + ".lock"

            # Prepare data row for CSV
            record = {
                'recommendation_id': trade_outcome.recommendation_id,
                'symbol': symbol,
                'timestamp_issued': trade_outcome.timestamp_issued,
                'regime_at_issuance': trade_outcome.regime_at_issuance,
                'trade_bias': trade_outcome.trade_bias,
                'strategy_type': trade_outcome.strategy_type,
                'atif_conviction_score': trade_outcome.atif_conviction_score_at_issuance,
                'triggering_signals_summary': trade_outcome.triggering_signals_summary,
                'pnl_percentage': trade_outcome.pnl_percentage,
                'exit_reason': trade_outcome.exit_reason,
                'win': 1 if trade_outcome.pnl_percentage is not None and trade_outcome.pnl_percentage > 0 else 0
            }
            
            df_record = pd.DataFrame([record])
            
            # Use a file lock to prevent race conditions from multiple processes
            lock = FileLock(lock_path, timeout=10)
            with lock:
                file_exists = os.path.exists(filepath)
                df_record.to_csv(filepath, mode='a', header=not file_exists, index=False)
            
            self.logger.info(f"Successfully recorded outcome for trade ID: {trade_outcome.recommendation_id}")

        except Exception as e:
            self.logger.error(f"Failed to record trade outcome: {e}", exc_info=True)
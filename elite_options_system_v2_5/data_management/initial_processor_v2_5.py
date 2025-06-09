# data_management/initial_processor_v2_5.py
# EOTS v2.5 - SENTRY-REFACRORED CANONICAL SCRIPT
#
# This module is responsible for the initial processing and validation of raw data
# and for orchestrating the main metric calculation process. It adheres to the
# Single Responsibility Principle and uses Pydantic models for data contracts.

import logging
import sys
from datetime import datetime
from typing import Any, Dict, List, Tuple

try:
    import pandas as pd
    import numpy as np
    from pydantic import ValidationError
except ImportError as e_dep:
    print(f"CRITICAL ERROR: initial_processor.py: Essential libraries not found: {e_dep}.", file=sys.stderr)
    sys.exit(1)

# --- EOTS V2.5 Component and Schema Imports ---
# Assumes a project structure where these are importable
from utils.config_manager_v2_5 import ConfigManagerV2_5
from data_models.eots_schemas_v2_5 import (
    UnprocessedDataBundleV2_5,
    ProcessedDataBundleV2_5,
    RawOptionsContractDataV2_5,
    ProcessedContractMetricsV2_5,
    ProcessedStrikeLevelMetricsV2_5,
    ProcessedUnderlyingAggregatesV2_5
)
from core_analytics_engine.metrics_calculator_v2_5 import MetricsCalculatorV2_5

# Module-Specific Logger
logger = logging.getLogger(__name__)

class InitialDataProcessorV2_5:
    """
    Processes raw data bundles from data fetchers.
    1. Converts Pydantic models to a DataFrame for processing.
    2. Validates and prepares inputs.
    3. Invokes the MetricsCalculator to compute all system metrics.
    4. Returns a comprehensive, processed data bundle as a Pydantic model.
    """

    def __init__(self, config_manager: ConfigManagerV2_5, metrics_calculator: MetricsCalculatorV2_5):
        """
        Initializes the InitialDataProcessor.

        Args:
            config_manager: The system's configuration manager instance.
            metrics_calculator: An initialized instance of MetricsCalculatorV2_5.
        """
        self.logger = logger.getChild(self.__class__.__name__)
        self.logger.info("Initializing InitialDataProcessorV2_5...")

        if not isinstance(config_manager, ConfigManagerV2_5):
            raise TypeError("A valid ConfigManagerV2_5 instance is required.")
        if not isinstance(metrics_calculator, MetricsCalculatorV2_5):
            raise TypeError("A valid MetricsCalculatorV2_5 instance is required.")

        self.config_manager = config_manager
        self.metrics_calculator = metrics_calculator

        self.logger.info("InitialDataProcessorV2_5 Initialized successfully.")

    def _prepare_dataframe_from_models(
        self,
        options_contracts: List[RawOptionsContractDataV2_5],
        underlying_data: Dict[str, Any],
        symbol: str,
        current_time: datetime
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Converts Pydantic models to a DataFrame and performs initial preparation."""
        prep_logger = self.logger.getChild("PrepareDataFrame")

        if not options_contracts:
            prep_logger.warning(f"Input options model list for {symbol} is empty.")
            return pd.DataFrame(), underlying_data

        try:
            list_of_dicts = [model.model_dump() for model in options_contracts]
            df = pd.DataFrame(list_of_dicts)
            prep_logger.info(f"Converted {len(options_contracts)} Pydantic models to DataFrame for {symbol}. Shape: {df.shape}")
        except Exception as e:
            raise ValueError(f"Failed to convert Pydantic models to DataFrame for {symbol}: {e}")

        # Add essential context columns
        df["underlying_price_at_fetch"] = float(underlying_data.get("price", np.nan))
        df["processing_time_dt_obj"] = current_time
        df["symbol"] = symbol.upper()

        # Ensure critical numeric types
        # Note: 'open_interest' is not in the v2.5 schema; removed from this loop.
        for col in ["strike"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df, underlying_data

    def process_data_and_calculate_metrics(self, raw_data_bundle: UnprocessedDataBundleV2_5) -> ProcessedDataBundleV2_5:
        """
        Main processing method. Validates raw data, prepares it, and orchestrates metric calculation.

        Args:
            raw_data_bundle: An UnprocessedDataBundleV2_5 object from the data fetching layer.

        Returns:
            A ProcessedDataBundleV2_5 object containing all calculated metrics.
        """
        proc_logger = self.logger.getChild("ProcessAndCalculate")
        
        if not isinstance(raw_data_bundle, UnprocessedDataBundleV2_5):
             raise TypeError("Input must be a valid UnprocessedDataBundleV2_5 object.")

        symbol = raw_data_bundle.underlying_data.symbol.upper()
        proc_logger.info(f"--- InitialProcessor: START for '{symbol}' at {raw_data_bundle.fetch_timestamp.isoformat()} ---")

        try:
            # 1. Prepare DataFrame from Pydantic models
            df_prepared, und_data_prepared = self._prepare_dataframe_from_models(
                options_contracts=raw_data_bundle.options_contracts,
                underlying_data=raw_data_bundle.underlying_data.model_dump(),
                symbol=symbol,
                current_time=raw_data_bundle.fetch_timestamp
            )

            # 2. Invoke the Metrics Calculator with the correct method and arguments
            proc_logger.info(f"Invoking MetricsCalculator for {symbol}...")
            (
                df_chain_all_metrics,
                df_strike_all_metrics,
                und_data_enriched
            ) = self.metrics_calculator.calculate_all_metrics(
                options_df_raw=df_prepared,
                und_data_api_raw=und_data_prepared
            )
            proc_logger.info(f"MetricsCalculator finished for {symbol}.")

            # 3. Assemble the final ProcessedDataBundle Pydantic model
            # This step inherently validates the output of the metrics calculator.
            processed_bundle = ProcessedDataBundleV2_5(
                options_data_with_metrics=[ProcessedContractMetricsV2_5(**row) for row in df_chain_all_metrics.to_dict('records')],
                strike_level_data_with_metrics=[ProcessedStrikeLevelMetricsV2_5(**row) for row in df_strike_all_metrics.to_dict('records')],
                underlying_data_enriched=ProcessedUnderlyingAggregatesV2_5(**und_data_enriched),
                processing_timestamp=datetime.now(),
                errors=raw_data_bundle.errors
            )

            proc_logger.info(f"--- InitialProcessor: END for '{symbol}'. Successfully created ProcessedDataBundle. ---")
            return processed_bundle

        except (ValidationError, ValueError, TypeError, KeyError) as e:
            err_msg = f"Critical data processing or validation error for '{symbol}': {e}"
            proc_logger.fatal(err_msg, exc_info=True)
            # In a real system, you might return a ProcessedDataBundle with an error state
            # For now, we re-raise to halt the cycle, as this indicates a severe issue.
            raise RuntimeError(err_msg) from e
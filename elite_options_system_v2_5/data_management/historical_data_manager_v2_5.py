# data_management/historical_data_manager_v2_5.py
# EOTS v2.5 - SENTRY-APPROVED, STUBBED FOR DRY RUN

import logging
from typing import Dict, Any, Optional
from datetime import date, timedelta
import pandas as pd

from utils.config_manager_v2_5 import ConfigManagerV2_5
from data_management.database_manager_v2_5 import DatabaseManagerV2_5

logger = logging.getLogger(__name__)

class HistoricalDataManagerV2_5:
    """
    STUBBED FOR DRY RUN. Manages the retrieval of historical market data.
    This version simulates functionality by returning empty data structures, allowing
    downstream components like MetricsCalculator to run without a live database.
    """

    def __init__(self, config_manager: ConfigManagerV2_5, db_manager: DatabaseManagerV2_5):
        self.logger = logger.getChild(self.__class__.__name__)
        self.logger.warning("HISTORICAL DATA MANAGER IS IN STUBBED MODE. ALL QUERIES WILL RETURN EMPTY DATA.")
        self.config_manager = config_manager
        
        if not isinstance(db_manager, DatabaseManagerV2_5):
            self.logger.critical("FATAL: Invalid db_manager object provided.")
            raise TypeError("db_manager must be an instance of DatabaseManagerV2_5")
            
        self.db_manager = db_manager
        self.logger.info("HistoricalDataManagerV2_5 initialized in stubbed mode.")

    def get_historical_metric(self, symbol: str, metric_name: str, lookback_days: int) -> Optional[pd.Series]:
        """
        STUBBED: Returns an empty pandas Series to prevent errors in downstream calculations.
        """
        self.logger.debug(f"STUBBED: get_historical_metric for '{metric_name}' called. Returning empty Series.")
        return pd.Series([], dtype='float64')

    def get_historical_ohlcv(self, symbol: str, lookback_days: int) -> Optional[pd.DataFrame]:
        """
        STUBBED: Returns an empty pandas DataFrame to prevent errors in downstream calculations.
        """
        self.logger.debug(f"STUBBED: get_historical_ohlcv for '{symbol}' called. Returning empty DataFrame.")
        return pd.DataFrame()

    def store_daily_eots_metrics(self, symbol: str, metric_date: date, metrics_data: Dict[str, Any]) -> None:
        """
        STUBBED: Simulates storing metrics by logging the request. No data is saved.
        """
        self.logger.info(f"STUBBED: store_daily_eots_metrics for {symbol} on {metric_date} called. No action taken.")
        return
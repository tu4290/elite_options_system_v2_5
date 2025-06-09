# EOD_Archiver_v2_5.py
# EOTS v2.5 - S-GRADE END-OF-DAY DATA ARCHIVAL SCRIPT

import argparse
import asyncio
import logging
from datetime import datetime

import pandas as pd

# EOTS v2.5 Component Imports
# This script assumes it is run from a context where the project root is in the Python path.
from utils.config_manager_v2_5 import ConfigManagerV2_5
from data_management.database_manager_v2_5 import (
    get_db_connection,
    initialize_database_schema,
    insert_batch_data
)
from data_management.historical_data_manager_v2_5 import HistoricalDataManagerV2_5
from data_management.tradier_data_fetcher_v2_5 import TradierDataFetcherV2_5 # Using Tradier as an example fetcher
from core_analytics_engine.metrics_calculator_v2_5 import MetricsCalculatorV2_5

def main(symbol: str, target_date_str: str):
    """
    Main execution function for the EOD Archiver.
    Initializes system components, fetches EOD data, calculates all metrics,
    and persists the results to the historical database.
    """
    
    historical_data_manager = None
    db_connection = None
    
    try:
        # --- 1. INITIALIZATION ---
        logging.info("--- EOTS EOD Archiver v2.5 Initializing ---")
        config = ConfigManagerV2_5()
        
        # Initialize database managers and connection
        db_connection = get_db_connection(config.get_setting("database_settings"))
        if not db_connection:
            raise ConnectionError("Failed to establish database connection. Aborting.")
        initialize_database_schema(db_connection) # Ensure schema exists
        
        # The HistoricalDataManager now receives the connection directly
        historical_data_manager = HistoricalDataManagerV2_5(config, db_connection)
        
        # Initialize analytics and fetching components
        metrics_calculator = MetricsCalculatorV2_5(config, historical_data_manager)
        
        logging.info(f"Initialized all components for EOD archival of {symbol} on {target_date_str}.")

        # --- 2. DATA FETCHING ---
        logging.info(f"Fetching EOD data for {symbol} on {target_date_str}...")
        async def fetch_data():
            async with TradierDataFetcherV2_5(config) as fetcher:
                # Fetching data for a specific expiration date. In a real EOD scenario,
                # this might loop over all relevant expirations. For this blueprint,
                # we assume a single relevant expiration (e.g., the specified date if it's an expiry).
                target_date_obj = datetime.strptime(target_date_str, "%Y-%m-%d").date()
                raw_contracts = await fetcher.fetch_options_chain(symbol, target_date_obj)
                raw_underlying = await fetcher.fetch_underlying_quote(symbol)
                return raw_contracts, raw_underlying

        raw_options, raw_underlying = asyncio.run(fetch_data())
        
        if not raw_options or not raw_underlying:
            raise RuntimeError(f"Data fetch failed: Received no data for options or underlying.")
        
        logging.info(f"Successfully fetched {len(raw_options)} option contracts and underlying quote.")
        
        # Prepare data for calculator
        options_df = pd.DataFrame([c.dict() for c in raw_options])
        und_data_dict = raw_underlying.dict()

        # --- 3. METRIC CALCULATION ---
        logging.info("Calculating final EOD metrics...")
        _, _, und_data_enriched = metrics_calculator.calculate_all_metrics(options_df, und_data_dict)
        logging.info("Successfully calculated final EOD metrics.")
        
        # --- 4. DATA EXTRACTION & PERSISTENCE ---
        target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()
        
        # EOTS METRICS
        logging.info("Extracting and storing daily EOTS aggregate metrics...")
        metrics_to_store = {
            "gib_oi_based_und": und_data_enriched.get("gib_oi_based_und"),
            "ivsdh_und_avg": und_data_enriched.get("ivsdh_und_avg"), # Note: This would come from strike_df aggregation
            "vapi_fa_z_score_und": und_data_enriched.get("vapi_fa_z_score_und"),
            "dwfd_z_score_und": und_data_enriched.get("dwfd_z_score_und"),
            "tw_laf_z_score_und": und_data_enriched.get("tw_laf_z_score_und"),
            "market_regime_summary": und_data_enriched.get("current_market_regime_v2_5", "UNKNOWN")
        }
        historical_data_manager.store_daily_eots_metrics(symbol, target_date, metrics_to_store)
        logging.info("Successfully stored daily EOTS metrics.")
        
        # OHLCV DATA
        logging.info("Extracting and storing daily OHLCV data...")
        ohlcv_data_to_store = [{
            "symbol": symbol,
            "date": target_date,
            "open": und_data_enriched.get("day_open_price_und"),
            "high": und_data_enriched.get("day_high_price_und"),
            "low": und_data_enriched.get("day_low_price_und"),
            "close": und_data_enriched.get("price"), # Assuming last price is close
            "volume": und_data_enriched.get("volume", 0),
        }]
        
        # Note: This uses the generic batch insert method as 'store_daily_ohlcv' is not a dedicated method.
        insert_batch_data(db_connection, 'daily_ohlcv', ohlcv_data_to_store)
        logging.info("Successfully stored daily OHLCV data.")

    except Exception as e:
        logging.critical(f"EOD Archiver failed for {symbol} on {target_date_str}: {e}", exc_info=True)
    finally:
        # --- 5. GRACEFUL SHUTDOWN ---
        if db_connection:
            db_connection.close()
            logging.info("Database connection closed. Shutdown complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EOTS v2.5 End-of-Day Data Archiver.")
    parser.add_argument("--symbol", required=True, type=str, help="The ticker symbol to process (e.g., SPY).")
    parser.add_argument("--date", required=True, type=str, help="The target date in YYYY-MM-DD format.")
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    main(args.symbol, args.date)
# data_management/tradier_data_fetcher_v2_5.py
# EOTS v2.5 - SENTRY-APPROVED, CANONICAL ASYNCHRONOUS IMPLEMENTATION

import logging
from datetime import date
from typing import Any, Dict, List, Optional

import aiohttp
from pydantic import ValidationError

from utils.config_manager_v2_5 import ConfigManagerV2_5
from utils.async_resilience_v2_5 import async_retry
# The schema is not strictly needed here as this fetcher now returns a dict,
# but it's good practice to keep the import for context.
from data_models.eots_schemas_v2_5 import RawUnderlyingDataCombinedV2_5

logger = logging.getLogger(__name__)

class TradierDataFetcherV2_5:
    """
    Asynchronous data fetcher for the Tradier API, using aiohttp and a
    resilient retry mechanism. This is the sole authority for OHLCV data.
    Manages its own ClientSession via async context manager pattern.
    """
    def __init__(self, config_manager: ConfigManagerV2_5):
        self.logger = logger.getChild(self.__class__.__name__)
        settings = config_manager.get_setting("data_fetcher_settings")
        self.api_key = settings.get("tradier_api_key")
        self.base_url = "https://sandbox.tradier.com/v1" # Use sandbox for development
        
        if not self.api_key:
            raise ValueError("Tradier API key is missing from configuration.")
            
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Accept': 'application/json'
        }
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Initializes the aiohttp session when entering an 'async with' block."""
        self.session = aiohttp.ClientSession(headers=self.headers)
        self.logger.debug("aiohttp.ClientSession opened for TradierDataFetcher.")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Closes the aiohttp session when exiting an 'async with' block."""
        if self.session:
            await self.session.close()
            self.logger.debug("aiohttp.ClientSession closed for TradierDataFetcher.")

    @async_retry(max_attempts=3, backoff_factor=0.5)
    async def fetch_underlying_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetches the underlying quote and OHLCV data from Tradier.
        This is the definitive source for this data in the EOTS v2.5 system.

        Returns:
            A dictionary of OHLCV data for merging, NOT a Pydantic model.
        """
        if not self.session:
            raise RuntimeError("Session not initialized. Use this fetcher with 'async with'.")

        params = {'symbols': symbol, 'greeks': 'false'}
        self.logger.debug(f"Fetching Tradier quote for {symbol}...")
        
        async with self.session.get(f"{self.base_url}/markets/quotes", params=params) as response:
            response.raise_for_status() # Raises aiohttp.ClientError for 4XX/5XX status
            data = await response.json()
            
            quotes_data = data.get('quotes')
            # The API can return 'null' or a dict without a 'quote' key if no symbol is found
            if not quotes_data or quotes_data.get('quote') is None:
                 self.logger.warning(f"No 'quote' data returned from Tradier for {symbol}. Response: {data}")
                 return None

            quote_raw = quotes_data['quote']
            # API may return a list for multiple symbols, or a dict for one. We handle both.
            if isinstance(quote_raw, list):
                if not quote_raw:
                    self.logger.warning(f"Tradier returned an empty list for quote data for {symbol}.")
                    return None
                quote_raw = quote_raw[0]

            self.logger.info(f"Successfully fetched OHLCV data from Tradier for {symbol}.")
            # Map Tradier API fields to our internal schema names for consistency
            return {
                "day_open_price_und": quote_raw.get("open"),
                "day_high_price_und": quote_raw.get("high"),
                "day_low_price_und": quote_raw.get("low"),
                "prev_day_close_price_und": quote_raw.get("prevclose"),
                # We also get the 'last' price from tradier, which can be a fallback
                "price": quote_raw.get("last"),
                "volume": quote_raw.get("volume")
            }

    # Note: Other methods like fetch_options_chain from the old file are removed,
    # as that responsibility now lies with the ConvexValue fetcher. This module
    # now has a single, clear responsibility.
# data_management/convexvalue_data_fetcher_v2_5.py
# EOTS v2.5 - SENTRY-APPROVED, CANONICAL V2.5.3 IMPLEMENTATION (UNABRIDGED)

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional

import aiohttp
from pydantic import ValidationError

from utils.config_manager_v2_5 import ConfigManagerV2_5
from utils.async_resilience_v2_5 import async_retry
from data_models.eots_schemas_v2_5 import RawOptionsContractDataV2_5, RawUnderlyingDataCombinedV2_5

logger = logging.getLogger(__name__)

# --- CANONICAL PARAMETER LISTS (UNABRIDGED) ---
# As provided by Prime Operator. This is the ground truth.
UNDERLYING_REQUIRED_PARAMS: List[str] = [
    "price", "volatility", "day_volume", "call_gxoi", "put_gxoi",
    "gammas_call_buy", "gammas_call_sell", "gammas_put_buy", "gammas_put_sell",
    "deltas_call_buy", "deltas_call_sell", "deltas_put_buy", "deltas_put_sell",
    "vegas_call_buy", "vegas_call_sell", "vegas_put_buy", "vegas_put_sell",
    "thetas_call_buy", "thetas_call_sell", "thetas_put_buy", "thetas_put_sell",
    "call_vxoi", "put_vxoi", "value_bs", "volm_bs", "deltas_buy", "deltas_sell",
    "vegas_buy", "vegas_sell", "thetas_buy", "thetas_sell", "volm_call_buy",
    "volm_put_buy", "volm_call_sell", "volm_put_sell", "value_call_buy",
    "value_put_buy", "value_call_sell", "value_put_sell", "vflowratio",
    "dxoi", "gxoi", "vxoi", "txoi", "call_dxoi", "put_dxoi"
]

OPTIONS_CHAIN_REQUIRED_PARAMS: List[str] = [
  "price", "volatility", "multiplier", "oi", "delta", "gamma", "theta", "vega",
  "vanna", "vomma", "charm", "dxoi", "gxoi", "vxoi", "txoi", "vannaxoi", "vommaxoi", "charmxoi",
  "dxvolm", "gxvolm", "vxvolm", "txvolm", "vannaxvolm", "vommaxvolm", "charmxvolm",
  "value_bs", "volm_bs", "deltas_buy", "deltas_sell", "gammas_buy", "gammas_sell",
  "vegas_buy", "vegas_sell", "thetas_buy", "thetas_sell",
  "valuebs_5m", "volmbs_5m", "valuebs_15m", "volmbs_15m",
  "valuebs_30m", "volmbs_30m", "valuebs_60m", "volmbs_60m",
  "volm", "volm_buy", "volm_sell", "value_buy", "value_sell"
]


class ConvexValueDataFetcherV2_5:
    """
    Asynchronous data fetcher for the ConvexValue API, precisely aligned
    with the unabridged canonical parameter lists.
    """
    def __init__(self, config_manager: ConfigManagerV2_5):
        self.logger = logger.getChild(self.__class__.__name__)
        settings = config_manager.get_setting("data_fetcher_settings")
        self.api_key = settings.get("convexvalue_api_key")
        self.base_url = "https://api.convexvalue.com/v2" # Using v2 as per typical API evolution
        if not self.api_key:
            raise ValueError("ConvexValue API key is missing from configuration.")

    def _parse_chain_response(self, json_payload: Dict[str, Any]) -> List[RawOptionsContractDataV2_5]:
        """Parses the get_chain response into a list of Pydantic models."""
        parsed_contracts = []
        raw_contract_list = json_payload.get('data', [])
        
        if not isinstance(raw_contract_list, list):
            self.logger.error(f"Expected 'data' in chain response to be a list, got {type(raw_contract_list)}.")
            return []

        for contract_data in raw_contract_list:
            try:
                # API returns a dictionary with 'params' and 'values' for each contract
                params = contract_data.get('params', [])
                values = contract_data.get('values', [])
                if not params or len(params) != len(values):
                    self.logger.warning(f"Skipping contract due to mismatched params/values. Data: {contract_data}")
                    continue

                contract_dict = {param: value for param, value in zip(params, values)}

                # Add identifier fields that are outside the params/values lists
                contract_dict['contract_symbol'] = contract_data.get('symbol')
                contract_dict['strike'] = contract_data.get('strike')
                contract_dict['opt_kind'] = contract_data.get('type')
                contract_dict['expiration_date'] = contract_data.get('expiration')

                validated_contract = RawOptionsContractDataV2_5(**contract_dict)
                parsed_contracts.append(validated_contract)
            except ValidationError as e:
                self.logger.warning(f"Skipping contract due to validation error: {e}. Data: {contract_data}")
        
        self.logger.info(f"Successfully parsed {len(parsed_contracts)} contracts from get_chain.")
        return parsed_contracts

    def _parse_underlying_response(self, json_payload: Dict[str, Any]) -> Optional[RawUnderlyingDataCombinedV2_5]:
        """Parses the get_und response into a single Pydantic model."""
        data_list = json_payload.get('data')
        if not data_list or not isinstance(data_list, list) or len(data_list) != 1:
            self.logger.error("Underlying response 'data' key is missing, not a list, or has unexpected length.")
            return None
            
        data = data_list[0]
        params_list = data.get('params', [])
        values_list = data.get('values', [])
        
        if len(params_list) != len(values_list):
            self.logger.error("Underlying response params and values lists have mismatched lengths.")
            return None
            
        und_dict = {param: value for param, value in zip(params_list, values_list)}
        
        und_dict['symbol'] = data.get('symbol')
        und_dict['timestamp'] = datetime.fromisoformat(data.get('timestamp')) if data.get('timestamp') else datetime.now()
        
        try:
            return RawUnderlyingDataCombinedV2_5(**und_dict)
        except ValidationError as e:
            self.logger.error(f"Pydantic validation failed for ConvexValue underlying data: {e}", exc_info=True)
            return None

    @async_retry(max_attempts=3, backoff_factor=0.7)
    async def _fetch_raw_data(self, session: aiohttp.ClientSession, endpoint: str, params: Dict[str, Any]) -> Dict:
        """Private helper to perform a single resilient GET request."""
        request_params = params.copy()
        if 'params' in request_params and isinstance(request_params['params'], list):
            request_params['params'] = ','.join(request_params['params'])
        
        request_params['key'] = self.api_key
        
        self.logger.debug(f"Fetching from {self.base_url}/{endpoint}...")
        async with session.get(f"{self.base_url}/{endpoint}", params=request_params) as response:
            response.raise_for_status()
            return await response.json()

    async def fetch_chain_and_underlying(self, symbol: str) -> Tuple[Optional[List[RawOptionsContractDataV2_5]], Optional[RawUnderlyingDataCombinedV2_5]]:
        """Concurrently fetches data from get_chain and get_und using the canonical unabridged parameter lists."""
        
        chain_params = {'symbol': symbol, 'params': OPTIONS_CHAIN_REQUIRED_PARAMS}
        und_params = {'symbol': symbol, 'params': UNDERLYING_REQUIRED_PARAMS}

        async with aiohttp.ClientSession() as session:
            chain_task = self._fetch_raw_data(session, "chain", chain_params)
            und_task = self._fetch_raw_data(session, "underlying", und_params)
            
            results = await asyncio.gather(chain_task, und_task, return_exceptions=True)

        chain_result, und_result = results
        
        parsed_contracts = None
        if isinstance(chain_result, dict):
            parsed_contracts = self._parse_chain_response(chain_result)
        elif isinstance(chain_result, Exception):
            self.logger.error(f"Failed to fetch ConvexValue chain data: {chain_result}", exc_info=True)

        parsed_underlying = None
        if isinstance(und_result, dict):
            parsed_underlying = self._parse_underlying_response(und_result)
        elif isinstance(und_result, Exception):
            self.logger.error(f"Failed to fetch ConvexValue underlying data: {und_result}", exc_info=True)
            
        return parsed_contracts, parsed_underlying
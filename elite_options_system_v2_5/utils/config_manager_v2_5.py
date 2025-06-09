# utils/config_manager_v2_5.py
# EOTS v2.5 - SENTRY-APPROVED, HARDENED CONFIGURATION MANAGER

import json
import os
import logging
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Callable

from jsonschema import validate, ValidationError
from pydantic import ValidationError as PydanticValidationError

from data_models.eots_schemas_v2_5 import EOTSConfigV2_5, SymbolSpecificOverrides

logger = logging.getLogger(__name__)

class ConfigManagerV2_5:
    """
    A robust, Singleton configuration manager for the EOTS v2.5 system.

    This class is responsible for loading, validating, and providing type-safe access
    to the system configuration. It uses a Pydantic-native approach for efficient,
    hierarchical setting lookups and is designed for high performance and testability.
    """
    _instance: Optional['ConfigManagerV2_5'] = None
    _config: Optional[EOTSConfigV2_5] = None
    _project_root: str = ""
    _is_loaded: bool = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ConfigManagerV2_5, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """
        Initializes the ConfigManager instance. Note: File loading is deferred
        to the load_config() method to improve testability.
        """
        if not self._project_root: # Initialize only once
            self._project_root = self._get_project_root()
            logger.info("ConfigManager Singleton instantiated. Ready to load configuration.")

    def load_config(self, config_file_path: str = "config/config_v2_5.json",
                    schema_file_path: str = "config/config.schema.v2.5.json"):
        """
        Loads, validates, and parses the configuration files. This method
        is called once at application startup.
        """
        if self._is_loaded:
            self.logger.warning("Configuration already loaded. Skipping reload.")
            return

        config_path = self.resolve_path(config_file_path)
        schema_path = self.resolve_path(schema_file_path)
        
        logger.info(f"Loading configuration from: {config_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        if not os.path.exists(schema_path):
            raise FileNotFoundError(f"Schema file not found: {schema_path}")

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from a file: {e}")

        try:
            logger.debug("Validating configuration against JSON schema...")
            validate(instance=config_data, schema=schema_data)
            logger.info("JSON schema validation successful.")
        except ValidationError as e:
            raise ValueError(f"Configuration validation failed: {e.message} at path '{'.'.join(map(str, e.path))}'")

        try:
            logger.debug("Parsing validated configuration into Pydantic model...")
            self._config = EOTSConfigV2_5(**config_data)
            self._is_loaded = True
            logger.info("Pydantic model parsing successful. Configuration is now loaded and type-safe.")
        except PydanticValidationError as e:
            raise ValueError(f"Pydantic data contract validation failed after loading config: {e}")

    # Use lru_cache for high-performance on repeated lookups of the same key path.
    @lru_cache(maxsize=256)
    def get_setting(self, *keys: str, symbol_context: Optional[str] = None, default: Any = None) -> Any:
        """
        Retrieves a nested configuration setting with hierarchical overrides using
        a Pydantic-native approach for high performance.

        The lookup order is:
        1. Symbol-specific override (e.g., "SPY")
        2. "DEFAULT" profile override
        3. Global setting

        Args:
            *keys: A sequence of keys to traverse the nested config.
            symbol_context: The symbol (e.g., "SPY") to check for overrides.
            default: The value to return if the key path is not found.

        Returns:
            The resolved configuration value or the default.
        """
        if not self._is_loaded or not self._config:
            self.logger.error("Configuration has not been loaded. Call load_config() first.")
            return default

        key_path_str = " -> ".join(keys)

        # 1. Try symbol-specific override
        if symbol_context and hasattr(self._config.symbol_specific_overrides, symbol_context):
            symbol_specific_config = getattr(self._config.symbol_specific_overrides, symbol_context)
            value = self._traverse_model(symbol_specific_config, keys)
            if value is not None:
                return value

        # 2. Try "DEFAULT" profile override
        if self._config.symbol_specific_overrides.DEFAULT:
            value = self._traverse_model(self._config.symbol_specific_overrides.DEFAULT, keys)
            if value is not None:
                return value

        # 3. Try global setting
        value = self._traverse_model(self._config, keys)
        if value is not None:
            return value

        self.logger.debug(f"Config key path '{key_path_str}' not found at any level. Returning default.")
        return default

    def _traverse_model(self, model_node: Any, path: Tuple[str, ...]) -> Any:
        """Helper to safely traverse a nested Pydantic model using getattr."""
        node = model_node
        for key in path:
            if hasattr(node, key):
                node = getattr(node, key)
            else:
                return None # Path does not exist
        return node

    def get_resolved_path(self, *keys: str, symbol_context: Optional[str] = None) -> Optional[str]:
        """Gets a path setting and resolves it relative to the project root."""
        path_val = self.get_setting(*keys, symbol_context=symbol_context)
        if path_val and isinstance(path_val, str):
            if os.path.isabs(path_val):
                return os.path.normpath(path_val)
            return os.path.normpath(os.path.join(self._project_root, path_val))
        return None
            
    def get_pydantic_config(self) -> EOTSConfigV2_5:
        """Returns the entire configuration as a validated Pydantic model."""
        if not self._is_loaded or not self._config:
            raise RuntimeError("Configuration has not been loaded. Call load_config() first.")
        return self._config

    def _get_project_root(self) -> str:
        """Determines the project's root directory."""
        # Assumes this file is in 'utils'. Two levels up is the project root.
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def get_project_root(self) -> str:
        return self._project_root
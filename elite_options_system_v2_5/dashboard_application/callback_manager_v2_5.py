# dashboard_application/callback_manager.py
# EOTS v2.5 - S-GRADE, AUTHORITATIVE CALLBACK MANAGER

import logging
import json
import importlib
from typing import Any, Optional, List, Dict

import dash
from dash import html, Input, Output, State, ctx, no_update, ALL
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

# EOTS V2.5 Imports
from core_analytics_engine.its_orchestrator_v2_5 import ITSOrchestratorV2_5
from utils.config_manager_v2_5 import ConfigManagerV2_5
from data_models.eots_schemas_v2_5 import FinalAnalysisBundleV2_5
from . import ids
from .utils_dashboard_v2_5 import create_empty_figure

# --- Module-Specific Logger & Global References ---
callback_logger = logging.getLogger(__name__)
ORCHESTRATOR_REF: Optional[ITSOrchestratorV2_5] = None
CONFIG_REF: Optional[ConfigManagerV2_5] = None

def register_v2_5_callbacks(app: dash.Dash, orchestrator: ITSOrchestratorV2_5, config: ConfigManagerV2_5):
    """Registers all v2.5 callbacks with the Dash app instance."""
    global ORCHESTRATOR_REF, CONFIG_REF
    ORCHESTRATOR_REF = orchestrator
    CONFIG_REF = config
    callback_logger.info("Registering EOTS v2.5 authoritative callbacks...")

    # --- Primary Data Fetching and Storage Callback ---
    @app.callback(
        Output(ids.ID_MAIN_DATA_STORE, 'data'),
        Output(ids.ID_STATUS_ALERT_CONTAINER, 'children'),
        Input(ids.ID_MANUAL_REFRESH_BUTTON, 'n_clicks'),
        Input(ids.ID_INTERVAL_LIVE_UPDATE, 'n_intervals'),
        State(ids.ID_SYMBOL_INPUT, 'value'),
        prevent_initial_call=True
    )
    def update_analysis_bundle_store(n_clicks: int, n_intervals: int, symbol: str) -> tuple:
        """
        The primary data callback. Triggered by manual refresh or a timer.
        Calls the orchestrator to get the latest analysis and stores it.
        """
        if not ORCHESTRATOR_REF or not symbol:
            return no_update, no_update

        trigger_id = ctx.triggered_id
        callback_logger.info(f"Data fetch triggered by '{trigger_id}' for symbol '{symbol}'.")
        
        try:
            analysis_bundle = ORCHESTRATOR_REF.run_full_analysis_cycle(symbol)
            if not analysis_bundle:
                alert = dbc.Alert(f"Orchestrator did not return a bundle for {symbol}.", color="warning", duration=4000)
                return no_update, alert
            
            # Serialize the Pydantic model to JSON for storage
            bundle_json = analysis_bundle.model_dump_json()
            status_message = f"Data updated for {symbol} at {analysis_bundle.bundle_timestamp.strftime('%H:%M:%S')}."
            alert = dbc.Alert(status_message, color="success", duration=4000)
            return bundle_json, alert

        except Exception as e:
            callback_logger.error(f"Error running full analysis cycle for {symbol}: {e}", exc_info=True)
            alert = dbc.Alert(f"Failed to fetch data for {symbol}: {e}", color="danger", duration=10000)
            return no_update, alert

    # --- Dynamic Mode and Chart Rendering Callback ---
    @app.callback(
        Output(ids.ID_PAGE_CONTENT, 'children'),
        Input(ids.ID_MAIN_DATA_STORE, 'data'),
        State(ids.ID_URL_LOCATION, 'pathname') # Use URL to determine the mode
    )
    def render_mode_content(bundle_json: Optional[str], pathname: str) -> Any:
        """
        Renders the entire layout for the currently selected mode.
        This is the central UI update callback.
        """
        if not bundle_json:
            return dbc.Alert("Waiting for initial data fetch...", color="info")

        # Determine mode from URL path, default to main
        mode_key = pathname.strip('/').split('/')[0] if pathname else 'main'
        modes_config = CONFIG_REF.get_setting('visualization_settings', 'dashboard', 'modes_detail_config', default={})
        mode_info = modes_config.get(mode_key) or modes_config.get('main')

        if not mode_info:
            return dbc.Alert(f"Configuration for mode '{mode_key}' not found.", color="danger")

        try:
            # Dynamically import the required display module
            display_module = importlib.import_module(f".modes.{mode_info['module_name']}", package='dashboard_application')
            
            # The display module's create_layout function generates the necessary charts and structure
            # It now receives the full data bundle to do its work.
            bundle = FinalAnalysisBundleV2_5.model_validate_json(bundle_json)
            mode_layout = display_module.create_layout(bundle, CONFIG_REF)
            
            return mode_layout
        except ImportError:
            callback_logger.error(f"Could not import display module: {mode_info['module_name']}")
            return dbc.Alert(f"Error loading UI module for mode '{mode_key}'.", color="danger")
        except Exception as e:
            callback_logger.error(f"Error rendering layout for mode '{mode_key}': {e}", exc_info=True)
            return dbc.Alert(f"An unexpected error occurred while rendering the {mode_key} view.", color="danger")
            
    # --- Callback to update Refresh Interval ---
    @app.callback(
        Output(ids.ID_INTERVAL_LIVE_UPDATE, 'interval'),
        Input(ids.ID_REFRESH_INTERVAL_DROPDOWN, 'value')
    )
    def update_refresh_interval(interval_seconds: str) -> int:
        """Updates the dcc.Interval component's refresh rate."""
        return int(interval_seconds) * 1000 if interval_seconds else 60 * 1000

    callback_logger.info("EOTS v2.5 authoritative callbacks registered successfully.")
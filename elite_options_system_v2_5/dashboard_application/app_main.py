# dashboard_application/app_main.py
# EOTS v2.5 - S-GRADE, AUTHORITATIVE APPLICATION ENTRY POINT (FINAL)

import logging
import atexit
import sys

from dash import Dash
import dash_bootstrap_components as dbc

# EOTS V2.5 Core Imports
from utils.config_manager_v2_5 import ConfigManagerV2_5
from data_management.database_manager_v2_5 import DatabaseManagerV2_5
from data_management.historical_data_manager_v2_5 import HistoricalDataManagerV2_5
from data_management.performance_tracker_v2_5 import PerformanceTrackerV2_5
from data_management.initial_processor_v2_5 import InitialDataProcessorV2_5
from core_analytics_engine.metrics_calculator_v2_5 import MetricsCalculatorV2_5
from core_analytics_engine.market_regime_engine_v2_5 import MarketRegimeEngineV2_5
from core_analytics_engine.signal_generator_v2_5 import SignalGeneratorV2_5
from core_analytics_engine.adaptive_trade_idea_framework_v2_5 import AdaptiveTradeIdeaFrameworkV2_5
from core_analytics_engine.trade_parameter_optimizer_v2_5 import TradeParameterOptimizerV2_5
from core_analytics_engine.its_orchestrator_v2_5 import ITSOrchestratorV2_5

# EOTS V2.5 Dashboard Imports
from dashboard_application import layout_manager
from dashboard_application import callback_manager_v2_5
from dashboard_application import utils_dashboard_v2_5 # ADDED IMPORT

def main():
    """
    Initializes and runs the EOTS v2.5 Dash application.
    This function encapsulates the entire application lifecycle, ensuring
    correct dependency injection and robust initialization.
    """
    # --- 1. System Configuration & Logging ---
    logging.basicConfig(
        level=logging.INFO, 
        format='[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)

    try:
        config_manager = ConfigManagerV2_5()
        config_manager.load_config()
        logger.info("Configuration loaded successfully.")
    except (FileNotFoundError, ValueError) as e:
        logger.critical(f"FATAL: Could not load or validate configuration. System cannot start. Error: {e}", exc_info=True)
        sys.exit(1)
        
    # Initialize dashboard-specific utilities (e.g., global plotly template)
    utils_dashboard_v2_5.initialize_dashboard_utils(config_manager)

    # --- 2. Backend Component Initialization (Dependency Hierarchy) ---
    logger.info("Initializing backend components...")

    db_settings = config_manager.get_setting("database_settings", default={})
    if not db_settings:
        logger.warning("Database settings not found in configuration. DB functionality will be stubbed.")
    
    db_manager = DatabaseManagerV2_5(db_settings)
    atexit.register(db_manager.close_connection)

    performance_tracker = PerformanceTrackerV2_5(config_manager)
    historical_data_manager = HistoricalDataManagerV2_5(config_manager, db_manager)
    
    metrics_calculator = MetricsCalculatorV2_5(config_manager, historical_data_manager)
    market_regime_engine = MarketRegimeEngineV2_5(config_manager)
    signal_generator = SignalGeneratorV2_5(config_manager)
    trade_parameter_optimizer = TradeParameterOptimizerV2_5(config_manager)
    adaptive_trade_idea_framework = AdaptiveTradeIdeaFrameworkV2_5(config_manager, performance_tracker)
    
    initial_processor = InitialDataProcessorV2_5(config_manager, metrics_calculator)
    
    logger.info("Initializing Master Orchestrator...")
    orchestrator = ITSOrchestratorV2_5(
        config_manager=config_manager,
        historical_data_manager=historical_data_manager, # Pass all dependencies
        performance_tracker=performance_tracker,
        metrics_calculator=metrics_calculator,
        initial_processor=initial_processor,
        market_regime_engine=market_regime_engine,
        signal_generator=signal_generator,
        adaptive_trade_idea_framework=adaptive_trade_idea_framework,
        trade_parameter_optimizer=trade_parameter_optimizer
    )
    logger.info("All backend components and orchestrator initialized.")

    # --- 3. Dash Application Setup ---
    logger.info("Setting up Dash application...")
    app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG], suppress_callback_exceptions=True)
    app.title = "EOTS v2.5 Apex Predator"

    # CORRECTED: Pass the config_manager instance to the layout function
    app.layout = layout_manager.create_master_layout(config_manager)
    
    callback_manager_v2_5.register_v2_5_callbacks(app, orchestrator, config_manager)
    logger.info("Dashboard layout created and callbacks registered.")

    # --- 4. Application Execution ---
    logger.info("Starting EOTS v2.5 Apex Predator Dashboard...")
    app.run_server(debug=True, host='0.0.0.0', port=8050)

if __name__ == '__main__':
    main()
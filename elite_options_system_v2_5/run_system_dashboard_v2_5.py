# elite_options_system_v2_5/run_system_dashboard_v2_5.py
# EOTS v2.5 - S-GRADE, AUTHORITATIVE ENTRY POINT

"""
Elite Options Trading System Dashboard Runner (V2.5 - "Apex Predator")

This is the primary and definitive launcher for the EOTS v2.5 Dashboard.
Its sole responsibilities are:
1. Loading environment variables from the .env file.
2. Configuring the system-wide logger.
3. Ensuring the Python environment is correctly configured for module resolution.
4. Delegating execution to the main application function.
"""

import os
import sys
import traceback
import logging
from dotenv import load_dotenv

def _setup_environment():
    """
    Performs all critical pre-flight checks and configurations.
    """
    # Step 1: Load environment variables from .env file.
    # This MUST be the first step to ensure API keys and other secrets are available.
    load_dotenv()
    
    # Step 2: Configure the root logger for the entire application.
    # All subsequent logging calls from any module will inherit this configuration.
    log_level = os.getenv("EOTS_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=log_level,
        format='[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        stream=sys.stdout  # Log to standard out
    )
    logging.info(f"Root logger configured to level: {log_level}")

    # Step 3: Ensure project root is in the system path for robust imports.
    try:
        # Assumes this script is in the project root.
        project_root = os.path.dirname(os.path.abspath(__file__))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
            logging.info(f"Path Integrity Check: Project root added to sys.path -> {project_root}")
    except Exception as e:
        logging.critical(f"CRITICAL PATH FAILURE: Could not set project root. {e}", exc_info=True)
        traceback.print_exc()
        sys.exit(1)

def main():
    """
    Definitive entry point for the EOTS v2.5 Dashboard Application.
    """
    _setup_environment()
    
    logging.info("EOTS v2.5 'Apex Predator' Initialization Sequence Activated.")
    
    try:
        # Import is performed *after* path correction to guarantee resolution.
        from dashboard_application.app_main import main as app_main
        
        logging.info("Handoff to Application Core...")
        # Execute the main application function.
        app_main()

    except ImportError as e:
        logging.critical(f"CRITICAL IMPORT FAILURE: The core application module could not be found.", exc_info=True)
        logging.critical("Ensure 'dashboard_application/app_main.py' exists and contains a 'main' function.")
        sys.exit(1)
        
    except Exception as e:
        logging.critical(f"CATASTROPHIC FAILURE: An unhandled exception occurred during application startup.", exc_info=True)
        sys.exit(1)

    logging.info("Execution complete. EOTS shutting down.")

if __name__ == "__main__":
    main()
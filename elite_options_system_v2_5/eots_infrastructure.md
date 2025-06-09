lite_options_system_v2_5/
├── config/
│   ├── config_v2_5.json
│   └── config.schema.v2_5.json
├── core_analytics_engine/
│   ├── __init__.py
│   ├── adaptive_trade_idea_framework_v2_5.py  # S-Grade, Hardened
│   ├── its_orchestrator_v2_5.py                 # S-Grade, Hardened
│   ├── market_regime_engine_v2_5.py             # S-Grade, Hardened
│   ├── market_regime_engine.py                  # Legacy, Deprecated
│   ├── metrics_calculator_v2_5.py               # S-Grade, Hardened
│   ├── metrics_calculator.py                    # Legacy, Deprecated
│   ├── recommendation_logic.py                  # Legacy, Deprecated
│   ├── signal_generator_v2_5.py                 # S-Grade, Hardened
│   ├── signal_generator.py                      # Legacy, Deprecated
│   ├── trade_parameter_optimizer_v2_5.py        # S-Grade, Hardened
│   └── trade_parameter_optimizer.py             # Legacy, Deprecated
├── dashboard_application/
│   ├── __init__.py
│   ├── app_main.py                              # v2.5 Core, Hardened
│   ├── assets/
│   │   ├── __init__.py
│   │   ├── custom.css
│   │   └── styles.css
│   ├── callback_manager.py                      # v2.5 Core, Hardened
│   ├── ids.py                                   # v2.5 Core, Canonical
│   ├── layout_manager.py                        # v2.5 Core, Hardened
│   ├── modes/
│   │   ├── __init__.py
│   │   ├── advanced_flow_mode.py                # v2.5 Implementation (Artisan)
│   │   ├── flow_mode_display.py                 # v2.5 Integrated
│   │   ├── main_dashboard_display.py            # v2.5 Layout
│   │   ├── structure_mode_display.py            # v2.5 Integrated
│   │   ├── time_decay_mode_display.py           # v2.5 Placeholder
│   │   └── volatility_mode_display.py           # v2.5 Placeholder
│   ├── styling .py                              # Legacy
│   └── utils_dashboard.py                       # Legacy
├── data_management/
│   ├── __init__.py
│   ├── convexvalue_data_fetcher_v2_5.py       # S-Grade, Async
│   ├── database_manager_v2_5.py               # S-Grade, Hardened
│   ├── fetcher.py                               # Legacy, Deprecated
│   ├── historical_data_manager_v2_5.py        # S-Grade, Hardened
│   ├── historical_data_manager.py             # Legacy, Deprecated
│   ├── initial_processor_v2_5.py              # v2.5 Component
│   ├── performance_tracker_v2_5.py            # S-Grade, Hardened
│   └── tradier_data_fetcher_v2_5.py           # S-Grade, Async
├── data_models/
│   ├── __init__.py
│   └── eots_schemas_v2_5.py                     # S-Grade, System-wide Data Contracts
├── logs/
│   └── requirements.txt
├── tests/
│   ├── test_advanced_flow_mode.py             # S-Grade
│   ├── test_market_regime_engine_v2_5.py      # S-Grade
│   ├── test_metrics_calculator_v2_5.py          # S-Grade
│   ├── test_signal_generator_v2_5.py          # S-Grade
│   └── test_structure_mode_display.py         # S-Grade
├── utils/
│   ├── __init__.py
│   ├── async_resilience_v2_5.py                 # S-Grade Utility
│   └── config_manager_v2_5.py                   # S-Grade Utility
├── EOD_Archiver_v2_5.py                         # S-Grade, Hardened
└── run_system_dashboard_v2_5.py                 # S-Grade, Final Entry Point
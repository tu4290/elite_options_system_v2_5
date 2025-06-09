# dashboard_application/layout_manager.py
# EOTS v2.5 - S-GRADE, AUTHORITATIVE MASTER LAYOUT DEFINITION

from dash import dcc, html
import dash_bootstrap_components as dbc

from . import ids
from utils.config_manager_v2_5 import ConfigManagerV2_5

def create_header(config: ConfigManagerV2_5) -> dbc.Navbar:
    """Creates the persistent header and navigation bar for the application."""
    
    # Dynamically build navigation links from the config file
    modes_config = config.get_setting('visualization_settings', 'dashboard', 'modes_detail_config', default={})
    nav_links = []
    # Ensure 'main' mode is first if it exists
    if 'main' in modes_config:
        nav_links.append(
            dbc.NavLink(
                modes_config['main']['label'], 
                href="/", 
                active="exact",
                className="nav-link-custom"
            )
        )
    for mode, details in modes_config.items():
        if mode != 'main':
             # The path is the key of the mode config
            nav_links.append(
                dbc.NavLink(
                    details['label'], 
                    href=f"/{mode}", 
                    active="exact",
                    className="nav-link-custom"
                )
            )

    return dbc.Navbar(
        dbc.Container(
            [
                html.A(
                    dbc.Row(
                        [
                            dbc.Col(html.Img(src="/assets/logo.png", height="40px")), # Assumes a logo in assets
                            dbc.Col(dbc.NavbarBrand("EOTS v2.5 Apex Predator", className="ms-2")),
                        ],
                        align="center",
                        className="g-0",
                    ),
                    href="/",
                    style={"textDecoration": "none"},
                ),
                dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
                dbc.Collapse(
                    dbc.Nav(
                        [
                            dbc.Input(id=ids.ID_SYMBOL_INPUT, placeholder="Enter Symbol...", type="text", value="SPY", className="me-2"),
                            dbc.Button("Refresh", id=ids.ID_MANUAL_REFRESH_BUTTON, color="primary", className="me-2"),
                            dbc.Select(
                                id=ids.ID_REFRESH_INTERVAL_DROPDOWN,
                                options=[
                                    {"label": "15s", "value": "15"},
                                    {"label": "30s", "value": "30"},
                                    {"label": "60s", "value": "60"},
                                    {"label": "5m", "value": "300"},
                                ],
                                value="60",
                                style={"width": "100px"},
                            ),
                        ],
                        className="ms-auto",
                        navbar=True,
                    ),
                    id="navbar-collapse",
                    navbar=True,
                ),
                # Navigation links are now part of the navbar
                dbc.Nav(nav_links, className="ms-auto", navbar=True, pills=True),
            ],
            fluid=True
        ),
        id=ids.ID_MASTER_HEADER,
        color="dark",
        dark=True,
        sticky="top",
    )

def create_master_layout(config: ConfigManagerV2_5) -> html.Div:
    """
    Creates the master layout for the entire Dash application.

    This layout includes core non-visual components for state management and routing,
    and defines the main structure including the header and content area.
    """
    return html.Div(
        id='app-container',
        children=[
            # Core non-visual components for state and routing
            dcc.Location(id=ids.ID_URL_LOCATION, refresh=False),
            dcc.Store(id=ids.ID_MAIN_DATA_STORE, storage_type='memory'),
            dcc.Interval(
                id=ids.ID_INTERVAL_LIVE_UPDATE,
                interval=60 * 1000,  # Default to 60s, will be updated by callback
                n_intervals=0,
                disabled=False
            ),
            
            # --- Persistent Header & Navigation ---
            create_header(config),
            
            # --- Status Alert Area ---
            html.Div(id=ids.ID_STATUS_ALERT_CONTAINER, style={"position": "fixed", "top": "80px", "right": "10px", "zIndex": "1050"}),

            # --- Dynamic Content Area ---
            # The content for the selected mode will be rendered here by a callback
            html.Main(
                id='app-body',
                className='app-body p-4', # Add padding for content
                children=[
                    html.Div(id=ids.ID_PAGE_CONTENT, children=[
                        # Initial loading message
                        dbc.Spinner(color="primary", children=html.Div("Initializing and fetching data..."))
                    ])
                ]
            )
        ]
    )
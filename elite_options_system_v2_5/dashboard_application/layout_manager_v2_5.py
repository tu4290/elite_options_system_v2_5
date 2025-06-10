# dashboard_application/layout_manager_v2_5.py
# EOTS v2.5 - S-GRADE, AUTHORITATIVE MASTER LAYOUT DEFINITION

from dash import dcc, html
import dash_bootstrap_components as dbc

from . import ids
from ..utils.config_manager_v2_5 import ConfigManagerV2_5

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
            nav_links.append(
                dbc.NavLink(
                    details['label'], 
                    href=f"/{mode}", 
                    active="exact",
                    className="nav-link-custom"
                )
            )

    # Fetch default symbol and refresh interval from config
    vis_settings = config.get_setting('visualization_settings', 'dashboard', 'defaults', default={})
    default_symbol = vis_settings.get('symbol', 'SPY') # Default to SPY if not in config
    default_refresh_interval = str(vis_settings.get('refresh_interval_seconds', '60')) # Default to '60' if not in config

    return dbc.Navbar(
        dbc.Container(
            [
                html.A(
                    dbc.Row(
                        [
                            dbc.Col(html.Img(src=config.get_setting('visualization_settings', 'dashboard', 'logo_path', default="/assets/logo.png"), height="40px")),
                            dbc.Col(dbc.NavbarBrand(config.get_setting('visualization_settings', 'dashboard', 'application_title', default="EOTS v2.5 Apex Predator"), className="ms-2")),
                        ],
                        align="center",
                        className="g-0",
                    ),
                    href="/",
                    style={"textDecoration": "none"},
                ),
                dbc.NavbarToggler(id="navbar-toggler", n_clicks=0), # For mobile responsiveness
                dbc.Collapse(
                    dbc.Nav(
                        [
                            dbc.Input(id=ids.ID_SYMBOL_INPUT, placeholder="Enter Symbol...", type="text", value=default_symbol, className="me-2", style={"width": "120px"}),
                            dbc.Button("Refresh", id=ids.ID_MANUAL_REFRESH_BUTTON, color="primary", className="me-2"),
                            dbc.Select(
                                id=ids.ID_REFRESH_INTERVAL_DROPDOWN,
                                options=[
                                    {"label": "10s", "value": "10"}, # Added 10s
                                    {"label": "15s", "value": "15"},
                                    {"label": "30s", "value": "30"},
                                    {"label": "60s", "value": "60"},
                                    {"label": "2m", "value": "120"}, # Changed 5m to 2m for more frequent option
                                    {"label": "5m", "value": "300"},
                                    {"label": "Off", "value": "999999999"} # Added Off option
                                ],
                                value=default_refresh_interval, # Use value from config
                                style={"width": "100px"},
                            ),
                        ],
                        className="ms-auto", # Align controls to the right
                        navbar=True,
                    ),
                    id="navbar-collapse",
                    navbar=True,
                ),
                dbc.Nav(nav_links, className="ms-auto", navbar=True, pills=True), # Navigation links next to controls
            ],
            fluid=True # Use full width of the container
        ),
        id=ids.ID_MASTER_HEADER,
        color=config.get_setting('visualization_settings', 'dashboard', 'header_color', default="dark"),
        dark=config.get_setting('visualization_settings', 'dashboard', 'header_dark_theme', default=True),
        sticky="top", # Keep header at the top
    )

def create_master_layout(config: ConfigManagerV2_5) -> html.Div:
    """
    Creates the master layout for the entire Dash application.

    This layout includes core non-visual components for state management and routing,
    and defines the main structure including the header and content area.
    """
    # Fetch default refresh interval for dcc.Interval
    vis_defaults = config.get_setting('visualization_settings', 'dashboard', 'defaults', default={})
    initial_refresh_seconds = int(vis_defaults.get('refresh_interval_seconds', 60))
    initial_refresh_ms = initial_refresh_seconds * 1000

    # Determine if interval should be disabled if "Off" (very large number) is the default
    interval_disabled = True if initial_refresh_seconds >= 999999999 else False

    return html.Div(
        id='app-container',
        children=[
            dcc.Location(id=ids.ID_URL_LOCATION, refresh=False),
            dcc.Store(id=ids.ID_MAIN_DATA_STORE, storage_type='memory'), # Stores the main analysis bundle
            dcc.Interval(
                id=ids.ID_INTERVAL_LIVE_UPDATE,
                interval=initial_refresh_ms,
                n_intervals=0,
                disabled=interval_disabled # Control if interval timer is active
            ),
            
            create_header(config), # Header is persistent
            
            # Area for status alerts (e.g., data updated, errors)
            html.Div(id=ids.ID_STATUS_ALERT_CONTAINER,
                     style={"position": "fixed", "top": "70px", "right": "10px", "zIndex": "1050", "width": "auto"}),

            # Main content area, dynamically updated by callbacks based on URL
            html.Main(
                id='app-body',
                className='app-body container-fluid p-3', # Use container-fluid for responsive padding
                children=[
                    dbc.Container(id=ids.ID_PAGE_CONTENT, fluid=True, children=[ # Ensure page content also uses fluid container
                        dbc.Spinner(color="primary", children=html.Div("Initializing and fetching data..."))
                    ])
                ]
            )
        ]
    )
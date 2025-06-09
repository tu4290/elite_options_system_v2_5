# dashboard_application/utils_dashboard_v2_5.py
# EOTS v2.5 - S-GRADE, AUTHORITATIVE DASHBOARD UTILITY MODULE

import logging
from typing import Any, Optional, Union
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
from dateutil import parser as date_parser

# EOTS v2.5 Imports
from utils.config_manager_v2_5 import ConfigManagerV2_5

# --- Module-level logger ---
logger = logging.getLogger(__name__)

# --- Global Plotly Template (to be initialized) ---
PLOTLY_TEMPLATE = "plotly_dark" # Default, can be overwritten by init function

def initialize_dashboard_utils(config: ConfigManagerV2_5):
    """
    Initializes shared utilities for the dashboard, such as the global Plotly template.
    This function should be called once at application startup in app_main.py.
    """
    global PLOTLY_TEMPLATE
    template_name = config.get_setting(
        'visualization_settings', 'dashboard', 'plotly_defaults', 'layout', 'template', 
        default='plotly_dark'
    )
    PLOTLY_TEMPLATE = template_name
    logger.info(f"Dashboard utilities initialized. Global Plotly template set to '{PLOTLY_TEMPLATE}'.")


def create_empty_figure(title: str, height: Optional[int] = None, reason: str = "No Data Available") -> go.Figure:
    """
    Creates a standardized, empty Plotly figure with a title and a reason.
    This function now uses the ConfigManager singleton directly.
    """
    fig = go.Figure()
    config = ConfigManagerV2_5() # Access the singleton instance

    fig_height = height or config.get_setting("visualization_settings", "dashboard", "default_graph_height", default=400)

    fig.update_layout(
        title_text=title,
        height=fig_height,
        xaxis={'showgrid': False, 'zeroline': False, 'visible': False},
        yaxis={'showgrid': False, 'zeroline': False, 'visible': False},
        annotations=[{
            'text': reason,
            'xref': 'paper',
            'yref': 'paper',
            'showarrow': False,
            'font': {'size': 16, 'color': '#888888'}
        }],
        template=PLOTLY_TEMPLATE
    )
    return fig


def add_timestamp_annotation(fig: go.Figure, timestamp: Union[datetime, str, float, None]) -> go.Figure:
    """
    Adds a standardized timestamp annotation to the bottom of a figure.
    This function now uses the ConfigManager singleton directly.
    """
    config = ConfigManagerV2_5()
    
    if not config.get_setting("visualization_settings", "dashboard", "show_chart_timestamps", default=True):
        return fig
    if timestamp is None:
        return fig

    ts_str = ""
    try:
        if isinstance(timestamp, (int, float)):
            ts_dt = datetime.fromtimestamp(timestamp)
        elif isinstance(timestamp, str):
            ts_dt = date_parser.parse(timestamp)
        elif isinstance(timestamp, datetime):
            ts_dt = timestamp
        else:
            ts_dt = None
        
        if ts_dt:
            ts_str = ts_dt.strftime('%Y-%m-%d %H:%M:%S %Z')

    except (ValueError, TypeError) as e:
        logger.warning(f"Could not format timestamp '{timestamp}': {e}")
        ts_str = str(timestamp) # Fallback to string representation

    cfg_path = ("visualization_settings", "dashboard", "plotly_defaults", "timestamp_annotation")
    annotation_defaults = {
        'x': config.get_setting(*cfg_path, 'x_pos', default=0.5),
        'y': config.get_setting(*cfg_path, 'y_pos', default=-0.15),
        'xref': 'paper', 'yref': 'paper', 'showarrow': False,
        'xanchor': 'center', 'yanchor': 'top',
        'font': {
            'size': config.get_setting(*cfg_path, 'font', 'size', default=9),
            'color': config.get_setting(*cfg_path, 'font', 'color', default='grey')
        }
    }
    fig.add_annotation(text=f"Updated: {ts_str}", **annotation_defaults)
    return fig


def add_price_line(fig: go.Figure, price: Optional[float], orientation: str = 'vertical', **kwargs) -> go.Figure:
    """
    Adds a vertical or horizontal line to a figure, typically for current price.
    This function now uses the ConfigManager singleton directly.
    """
    if price is None or not pd.notna(price):
        return fig

    config = ConfigManagerV2_5()
    cfg_path = ("visualization_settings", "dashboard", "plotly_defaults", "price_line")
    
    line_style = {
        'color': config.get_setting(*cfg_path, 'color', default='rgba(255, 255, 255, 0.5)'),
        'width': config.get_setting(*cfg_path, 'width', default=1),
        'dash': config.get_setting(*cfg_path, 'dash', default='dash')
    }
    line_style.update(kwargs) # Allow runtime overrides

    if orientation == 'vertical':
        fig.add_vline(x=price, line=line_style)
    elif orientation == 'horizontal':
        fig.add_hline(y=price, line=line_style)
    else:
        logger.warning(f"Invalid orientation '{orientation}' for add_price_line. Use 'vertical' or 'horizontal'.")

    return fig
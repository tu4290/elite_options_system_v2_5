# dashboard_application/modes/advanced_flow_mode.py
# EOTS v2.5 - S-GRADE, AUTHORITATIVE ADVANCED FLOW DISPLAY

import logging
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from dash import html, dcc
import dash_bootstrap_components as dbc

# EOTS v2.5 Imports
from data_models.eots_schemas_v2_5 import FinalAnalysisBundleV2_5, ProcessedUnderlyingAggregatesV2_5
from utils.config_manager_v2_5 import ConfigManagerV2_5
from ..utils_dashboard_v2_5 import create_empty_figure, add_timestamp_annotation, PLOTLY_TEMPLATE
from .. import ids

logger = logging.getLogger(__name__)

# --- Helper Function for Chart Generation ---

def _create_z_score_gauge(
    metric_name: str, 
    z_score_value: Optional[float], 
    symbol: str, 
    component_id: str
) -> dcc.Graph:
    """A centralized helper to create the Z-Score gauge chart."""
    
    fig_height = 250 # Standard height for these gauges
    title_text = f"<b>{symbol}</b> - {metric_name}"

    if pd.isna(z_score_value) or z_score_value is None:
        fig = create_empty_figure(title=title_text, height=fig_height, reason=f"{metric_name} N/A")
        return dcc.Graph(id=component_id, figure=fig)

    value = float(z_score_value)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': metric_name, 'font': {'size': 18}},
        number={'font': {'size': 36}},
        gauge={
            'axis': {'range': [-3, 3], 'tickwidth': 1, 'tickcolor': "darkgrey"},
            'bar': {'color': "rgba(0,0,0,0)"}, # Invisible bar, threshold line is the indicator
            'steps': [
                {'range': [-3.0, -2.0], 'color': '#d62728'}, # Strong Negative
                {'range': [-2.0, -0.5], 'color': '#ff9896'}, # Mild Negative
                {'range': [-0.5, 0.5], 'color': '#aec7e8'},   # Neutral
                {'range': [0.5, 2.0], 'color': '#98df8a'},  # Mild Positive
                {'range': [2.0, 3.0], 'color': '#2ca02c'}   # Strong Positive
            ],
            'threshold': {
                'line': {'color': "white", 'width': 5},
                'thickness': 0.95, 'value': value
            }
        }
    ))
    
    fig.update_layout(
        title={'text': title_text, 'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
        height=fig_height,
        margin={'t': 60, 'b': 20, 'l': 20, 'r': 20},
        template=PLOTLY_TEMPLATE
    )

    return dcc.Graph(id=component_id, figure=fig)


# --- Main Layout Function ---

def create_layout(bundle: FinalAnalysisBundleV2_5, config: ConfigManagerV2_5) -> html.Div:
    """
    Creates the complete layout for the "Advanced Flow Analysis" mode.
    This is the single entry point called by the callback manager.
    """
    if not bundle or not bundle.processed_data_bundle:
        return dbc.Alert("Advanced Flow data is not available. Cannot render mode.", color="danger")

    und_data = bundle.processed_data_bundle.underlying_data_enriched
    symbol = bundle.target_symbol

    # Generate all components for this mode
    vapi_gauge = _create_z_score_gauge("VAPI-FA", und_data.vapi_fa_z_score_und, symbol, ids.ID_VAPI_GAUGE)
    dwfd_gauge = _create_z_score_gauge("DWFD", und_data.dwfd_z_score_und, symbol, ids.ID_DWFD_GAUGE)
    tw_laf_gauge = _create_z_score_gauge("TW-LAF", und_data.tw_laf_z_score_und, symbol, ids.ID_TW_LAF_GAUGE)

    # Assemble the layout using the generated components
    return html.Div(
        children=[
            dbc.Container(
                fluid=True,
                children=[
                    dbc.Row(
                        [
                            dbc.Col(vapi_gauge, md=12, lg=4, className="mb-4"),
                            dbc.Col(dwfd_gauge, md=12, lg=4, className="mb-4"),
                            dbc.Col(tw_laf_gauge, md=12, lg=4, className="mb-4"),
                        ],
                        className="mt-4"
                    ),
                    # Future rows for historical trend charts of these metrics can be added here
                ]
            )
        ]
    )
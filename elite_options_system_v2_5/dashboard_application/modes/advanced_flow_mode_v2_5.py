# dashboard_application/modes/advanced_flow_mode_v2_5.py
# EOTS v2.5 - S-GRADE, AUTHORITATIVE ADVANCED FLOW DISPLAY

import logging
from typing import Optional
from datetime import datetime # Added for timestamp type hint

import pandas as pd
import plotly.graph_objects as go
from dash import html, dcc
import dash_bootstrap_components as dbc

# EOTS v2.5 Imports
from ...data_models.eots_schemas_v2_5 import FinalAnalysisBundleV2_5 # ProcessedUnderlyingAggregatesV2_5 is part of this
from ...utils.config_manager_v2_5 import ConfigManagerV2_5
from ..utils_dashboard_v2_5 import create_empty_figure, add_timestamp_annotation, PLOTLY_TEMPLATE
from .. import ids

logger = logging.getLogger(__name__)

# --- Helper Function for Chart Generation ---

def _create_z_score_gauge(
    metric_name: str,
    z_score_value: Optional[float],
    symbol: str,
    component_id: str,
    config: ConfigManagerV2_5, # Added config manager
    timestamp: Optional[datetime] # Added timestamp for annotation
) -> dcc.Graph:
    """A centralized helper to create the Z-Score gauge chart."""

    adv_flow_settings = config.get_setting('visualization_settings', 'dashboard', 'advanced_flow_chart_settings', default={})
    gauge_specific_height_key = f"{metric_name.lower().replace(' ', '_').replace('-', '_')}_gauge_height" # e.g., vapi_fa_gauge_height
    fig_height = adv_flow_settings.get(gauge_specific_height_key, adv_flow_settings.get('default_gauge_height', 250))

    title_text = f"<b>{symbol}</b> - {metric_name}"

    if pd.isna(z_score_value) or z_score_value is None:
        fig = create_empty_figure(title=title_text, height=fig_height, reason=f"{metric_name} N/A")
        # Still add timestamp to empty figures for consistency
        if timestamp:
            fig = add_timestamp_annotation(fig, timestamp)
        return dcc.Graph(id=component_id, figure=fig)

    value = float(z_score_value)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': metric_name, 'font': {'size': 18}}, # Gauge-internal title
        number={'font': {'size': 36}},
        gauge={
            'axis': {'range': [-3, 3], 'tickwidth': 1, 'tickcolor': "darkgrey"},
            'bar': {'color': "rgba(0,0,0,0)"},
            'steps': [
                {'range': [-3.0, -2.0], 'color': adv_flow_settings.get("gauge_color_strong_neg", '#d62728')},
                {'range': [-2.0, -0.5], 'color': adv_flow_settings.get("gauge_color_mild_neg", '#ff9896')},
                {'range': [-0.5, 0.5], 'color': adv_flow_settings.get("gauge_color_neutral", '#aec7e8')},
                {'range': [0.5, 2.0], 'color': adv_flow_settings.get("gauge_color_mild_pos", '#98df8a')},
                {'range': [2.0, 3.0], 'color': adv_flow_settings.get("gauge_color_strong_pos", '#2ca02c')}
            ],
            'threshold': {
                'line': {'color': adv_flow_settings.get("gauge_threshold_line_color", "white"), 'width': 5},
                'thickness': 0.95, 'value': value
            }
        }
    ))

    fig.update_layout(
        title={'text': title_text, 'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'}, # Main figure title
        height=fig_height,
        margin=adv_flow_settings.get("gauge_margin", {'t': 60, 'b': 40, 'l': 20, 'r': 20}), # Configurable margin, added more bottom margin for timestamp
        template=PLOTLY_TEMPLATE
    )

    if timestamp:
        fig = add_timestamp_annotation(fig, timestamp)

    return dcc.Graph(id=component_id, figure=fig)


# --- Main Layout Function ---

def create_layout(bundle: FinalAnalysisBundleV2_5, config: ConfigManagerV2_5) -> html.Div:
    """
    Creates the complete layout for the "Advanced Flow Analysis" mode.
    This is the single entry point called by the callback manager.
    """
    if not bundle or not bundle.processed_data_bundle or not bundle.processed_data_bundle.underlying_data_enriched:
        # Added check for underlying_data_enriched
        logger.warning("Advanced Flow data or underlying_data_enriched is not available. Cannot render mode.")
        return dbc.Alert("Advanced Flow data is not available. Cannot render mode.", color="warning", className="m-4")

    und_data = bundle.processed_data_bundle.underlying_data_enriched
    symbol = bundle.target_symbol
    bundle_timestamp = bundle.bundle_timestamp # Extract timestamp for passing to helpers

    # Generate all components for this mode
    vapi_gauge = _create_z_score_gauge(
        "VAPI-FA", und_data.vapi_fa_z_score_und, symbol,
        ids.ID_VAPI_GAUGE, config, bundle_timestamp
    )
    dwfd_gauge = _create_z_score_gauge(
        "DWFD", und_data.dwfd_z_score_und, symbol,
        ids.ID_DWFD_GAUGE, config, bundle_timestamp
    )
    tw_laf_gauge = _create_z_score_gauge(
        "TW-LAF", und_data.tw_laf_z_score_und, symbol,
        ids.ID_TW_LAF_GAUGE, config, bundle_timestamp
    )

    # Assemble the layout using the generated components
    return html.Div(
        className="advanced-flow-mode-container", # Added class for potential styling
        children=[
            dbc.Container(
                fluid=True,
                children=[
                    dbc.Row(
                        [
                            dbc.Col(vapi_gauge, md=12, lg=4, className="mb-4 gauge-column"), # Added class
                            dbc.Col(dwfd_gauge, md=12, lg=4, className="mb-4 gauge-column"), # Added class
                            dbc.Col(tw_laf_gauge, md=12, lg=4, className="mb-4 gauge-column"), # Added class
                        ],
                        className="mt-4 justify-content-center" # Center row if fewer than 3 gauges
                    ),
                    # Future rows for historical trend charts of these metrics can be added here
                    # Example:
                    # dbc.Row([dbc.Col(html.Div("Placeholder for VAPI Historical Chart"), width=12)], className="mt-4"),
                ]
            )
        ]
    )
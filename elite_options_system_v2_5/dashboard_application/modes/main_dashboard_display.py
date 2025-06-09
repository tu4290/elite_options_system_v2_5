# dashboard_application/modes/main_dashboard_display.py
# EOTS v2.5 - S-GRADE, AUTHORITATIVE MAIN DASHBOARD DISPLAY

import logging
from typing import List, Optional
import pandas as pd
import plotly.graph_objects as go
from dash import html, dash_table
import dash_bootstrap_components as dbc

from .. import ids
from ..utils_dashboard_v2_5 import create_empty_figure, PLOTLY_TEMPLATE, add_timestamp_annotation
from data_models.eots_schemas_v2_5 import FinalAnalysisBundleV2_5, ProcessedUnderlyingAggregatesV2_5, ActiveRecommendationPayloadV2_5
from utils.config_manager_v2_5 import ConfigManagerV2_5

logger = logging.getLogger(__name__)

# --- Helper Functions for Component Generation ---

def _create_regime_display(und_data: ProcessedUnderlyingAggregatesV2_5, config: ConfigManagerV2_5) -> dbc.Card:
    """Creates the Market Regime Indicator display card."""
    regime_text = getattr(und_data, 'current_market_regime_v2_5', 'REGIME UNKNOWN')
    
    # Use config for color mapping for robustness
    colors_map = config.get_setting("visualization_settings", "dashboard", "main_dashboard_settings", "regime_indicator", "regime_colors", default={})
    
    color_class = colors_map.get("default", "secondary")
    for key, color in colors_map.items():
        if key in regime_text.upper():
            color_class = color
            break

    return dbc.Card(
        dbc.CardBody([
            html.H6("Market Regime", className="card-title text-muted text-center"),
            html.H4(regime_text, className=f"card-text text-center text-{color_class}")
        ]),
        id=ids.ID_REGIME_DISPLAY_CARD
    )

def _create_flow_gauge(metric_name: str, value: Optional[float], component_id: str) -> dcc.Graph:
    """Creates a gauge figure for a Z-Score flow metric."""
    if value is None or not pd.notna(value):
        fig = create_empty_figure(title=metric_name, reason="Data N/A")
    else:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(value),
            title={'text': metric_name, 'font': {'size': 16}},
            gauge={
                'axis': {'range': [-3, 3], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "rgba(0,0,0,0)"},
                'steps': [
                    {'range': [-3, -2], 'color': '#d62728'},
                    {'range': [-2, -0.5], 'color': '#ff9896'},
                    {'range': [-0.5, 0.5], 'color': '#aec7e8'},
                    {'range': [0.5, 2], 'color': '#98df8a'},
                    {'range': [2, 3], 'color': '#2ca02c'}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.9, 'value': float(value)
                }
            }
        ))
        fig.update_layout(height=200, margin={'t': 40, 'b': 20, 'l': 20, 'r': 20}, template=PLOTLY_TEMPLATE)

    return dcc.Graph(id=component_id, figure=fig)

def _create_recommendations_table(recommendations: List[ActiveRecommendationPayloadV2_5]) -> dbc.Card:
    """Creates the display for the ATIF recommendations table using dash_table.DataTable."""
    if not recommendations:
        return dbc.Card(dbc.CardBody([
            html.H4("ATIF Recommendations"),
            dbc.Alert("No active recommendations.", color="info")
        ]))

    # Convert Pydantic models to a list of dictionaries for the table
    data_for_table = []
    for reco in recommendations:
        data_for_table.append({
            'Strategy': reco.strategy_type,
            'Bias': reco.trade_bias,
            'Conviction': f"{reco.atif_conviction_score_at_issuance:.2f}",
            'Status': reco.status,
            'Entry': f"{reco.entry_price_initial:.2f}",
            'Stop': f"{reco.stop_loss_current:.2f}",
            'Target 1': f"{reco.target_1_current:.2f}",
            'Rationale': reco.target_rationale[:50] + '...' if reco.target_rationale and len(reco.target_rationale) > 50 else reco.target_rationale
        })

    return dbc.Card(
        dbc.CardBody([
            html.H4("ATIF Recommendations"),
            dash_table.DataTable(
                id=ids.ID_RECOMMENDATIONS_TABLE,
                columns=[{"name": i, "id": i} for i in data_for_table[0].keys()],
                data=data_for_table,
                style_cell={'textAlign': 'left', 'padding': '5px'},
                style_header={'backgroundColor': 'rgb(30, 30, 30)', 'fontWeight': 'bold'},
                style_data={'backgroundColor': 'rgb(50, 50, 50)'},
                style_as_list_view=True,
            )
        ])
    )


# --- Main Layout Function ---

def create_layout(bundle: FinalAnalysisBundleV2_5, config: ConfigManagerV2_5) -> html.Div:
    """
    Creates the complete layout for the Main Dashboard mode.
    This is the single entry point called by the callback manager.
    """
    if not bundle or not bundle.processed_data_bundle:
        return dbc.Alert("Data bundle is not available. Cannot render Main Dashboard.", color="danger")

    und_data = bundle.processed_data_bundle.underlying_data_enriched
    recommendations = bundle.active_recommendations_v2_5

    return html.Div(
        id=ids.ID_MAIN_DASHBOARD_CONTAINER,
        children=[
            dbc.Container(
                fluid=True,
                children=[
                    # Row 1: Top-Level KPI Indicators
                    dbc.Row(
                        [
                            dbc.Col(_create_regime_display(und_data, config), md=6, lg=3, className="mb-4"),
                            dbc.Col(_create_flow_gauge("VAPI-FA", und_data.vapi_fa_z_score_und, ids.ID_VAPI_GAUGE), md=6, lg=3, className="mb-4"),
                            dbc.Col(_create_flow_gauge("DWFD", und_data.dwfd_z_score_und, ids.ID_DWFD_GAUGE), md=6, lg=3, className="mb-4"),
                            dbc.Col(_create_flow_gauge("TW-LAF", und_data.tw_laf_z_score_und, ids.ID_TW_LAF_GAUGE), md=6, lg=3, className="mb-4"),
                        ]
                    ),
                    # Row 2: Primary Recommendations Table
                    dbc.Row(
                        [
                            dbc.Col(_create_recommendations_table(recommendations), width=12),
                        ]
                    ),
                    # Future rows for summary heatmaps or charts can be added here
                ]
            )
        ]
    )
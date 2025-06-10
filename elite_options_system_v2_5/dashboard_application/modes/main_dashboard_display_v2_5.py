# dashboard_application/modes/main_dashboard_display_v2_5.py
# EOTS v2.5 - S-GRADE, AUTHORITATIVE MAIN DASHBOARD DISPLAY

import logging
from typing import List, Optional, Any # Added Any for config flexibility
from datetime import datetime # Added for timestamp type hint

import pandas as pd
import plotly.graph_objects as go
from dash import html, dcc, dash_table # Added dcc for dcc.Graph
import dash_bootstrap_components as dbc

from .. import ids
from ..utils_dashboard_v2_5 import create_empty_figure, PLOTLY_TEMPLATE, add_timestamp_annotation
from ...data_models.eots_schemas_v2_5 import FinalAnalysisBundleV2_5, ProcessedUnderlyingAggregatesV2_5, ActiveRecommendationPayloadV2_5
from ...utils.config_manager_v2_5 import ConfigManagerV2_5

logger = logging.getLogger(__name__)

# --- Helper Functions for Component Generation ---

def _create_regime_display(und_data: ProcessedUnderlyingAggregatesV2_5, config: ConfigManagerV2_5) -> dbc.Card:
    """Creates the Market Regime Indicator display card."""
    regime_text = getattr(und_data, 'current_market_regime_v2_5', 'REGIME UNKNOWN')

    main_dash_settings = config.get_setting("visualization_settings", "dashboard", "main_dashboard_settings", default={})
    regime_indicator_settings = main_dash_settings.get("regime_indicator", {})
    colors_map = regime_indicator_settings.get("regime_colors", {})

    color_class = colors_map.get("default", "secondary") # Default color if no specific match
    for key, color in colors_map.items():
        if key.upper() in regime_text.upper(): # Case-insensitive matching for regime keys
            color_class = color
            break

    card_body_content = [
        html.H6(regime_indicator_settings.get("title", "Market Regime"), className="card-title text-muted text-center"),
        html.H4(regime_text, className=f"card-text text-center text-{color_class}")
    ]
    # Timestamp for regime card is skipped as per plan to avoid layout complexity here.

    return dbc.Card(dbc.CardBody(card_body_content), id=ids.ID_REGIME_DISPLAY_CARD)


def _create_flow_gauge(
    metric_name: str,
    value: Optional[float],
    component_id: str,
    config: ConfigManagerV2_5,
    timestamp: Optional[datetime],
    symbol: str # Added symbol for title
) -> dcc.Graph:
    """Creates a gauge figure for a Z-Score flow metric."""
    main_dash_settings = config.get_setting("visualization_settings", "dashboard", "main_dashboard_settings", default={})
    flow_gauge_settings = main_dash_settings.get("flow_gauge", {})
    fig_height = flow_gauge_settings.get("height", 200)
    gauge_title_text = f"<b>{symbol}</b> - {metric_name}" # Title for the overall figure
    indicator_title_text = metric_name # Title for the indicator itself

    if value is None or not pd.notna(value):
        fig = create_empty_figure(title=gauge_title_text, height=fig_height, reason="Data N/A")
    else:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(value),
            title={'text': indicator_title_text, 'font': {'size': flow_gauge_settings.get("indicator_font_size", 16)}},
            number={'font': {'size': flow_gauge_settings.get("number_font_size", 24)}},
            gauge={
                'axis': {'range': flow_gauge_settings.get("axis_range", [-3, 3]), 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "rgba(0,0,0,0)"}, # Invisible bar
                'steps': flow_gauge_settings.get("steps", [ # Default steps
                    {'range': [-3, -2], 'color': '#d62728'}, {'range': [-2, -0.5], 'color': '#ff9896'},
                    {'range': [-0.5, 0.5], 'color': '#aec7e8'}, {'range': [0.5, 2], 'color': '#98df8a'},
                    {'range': [2, 3], 'color': '#2ca02c'}
                ]),
                'threshold': {
                    'line': {'color': flow_gauge_settings.get("threshold_line_color", "white"), 'width': 4},
                    'thickness': 0.9, 'value': float(value)
                }
            }
        ))
        fig.update_layout(
            title={'text': gauge_title_text, 'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
            height=fig_height,
            margin=flow_gauge_settings.get("margin", {'t': 60, 'b': 40, 'l': 20, 'r': 20}), # Added more bottom margin
            template=PLOTLY_TEMPLATE
        )

    if timestamp: # Add timestamp regardless of data presence for consistency
        fig = add_timestamp_annotation(fig, timestamp)

    return dcc.Graph(id=component_id, figure=fig)

def _create_recommendations_table(
    recommendations: List[ActiveRecommendationPayloadV2_5],
    config: ConfigManagerV2_5,
    timestamp: Optional[datetime],
    symbol: str # Added symbol for unique ID
) -> dbc.Card:
    """Creates the display for the ATIF recommendations table using dash_table.DataTable."""
    main_dash_settings = config.get_setting("visualization_settings", "dashboard", "main_dashboard_settings", default={})
    table_settings = main_dash_settings.get("recommendations_table", {})
    max_rationale_len = table_settings.get('max_rationale_length', 50)
    table_title = table_settings.get("title", "ATIF Recommendations")

    card_body_children: List[Any] = [html.H4(table_title)] # Initialize list for card body

    if not recommendations:
        card_body_children.append(dbc.Alert("No active recommendations.", color="info", className="mt-2"))
    else:
        data_for_table = []
        for reco in recommendations:
            rationale = reco.target_rationale
            if rationale and len(rationale) > max_rationale_len:
                rationale = rationale[:max_rationale_len] + '...'

            data_for_table.append({
                'Strategy': reco.strategy_type,
                'Bias': reco.trade_bias,
                'Conviction': f"{reco.atif_conviction_score_at_issuance:.2f}", # Format score
                'Status': reco.status,
                'Entry': f"{reco.entry_price_initial:.2f}" if reco.entry_price_initial is not None else "N/A",
                'Stop': f"{reco.stop_loss_current:.2f}" if reco.stop_loss_current is not None else "N/A",
                'Target 1': f"{reco.target_1_current:.2f}" if reco.target_1_current is not None else "N/A",
                'Rationale': rationale
            })

        table_component = dash_table.DataTable(
            id=f"{ids.ID_RECOMMENDATIONS_TABLE}-{symbol.lower()}", # Ensure unique ID if multiple instances exist
            columns=[{"name": i, "id": i} for i in data_for_table[0].keys()] if data_for_table else [],
            data=data_for_table,
            style_cell=table_settings.get("style_cell", {'textAlign': 'left', 'padding': '5px', 'minWidth': '80px', 'width': 'auto', 'maxWidth': '200px'}),
            style_header=table_settings.get("style_header", {'backgroundColor': 'rgb(30, 30, 30)', 'fontWeight': 'bold', 'color': 'white'}),
            style_data=table_settings.get("style_data", {'backgroundColor': 'rgb(50, 50, 50)', 'color': 'white'}),
            style_as_list_view=True,
            page_size=table_settings.get("page_size", 5), # Make page size configurable
            sort_action="native",
            filter_action="native",
        )
        card_body_children.append(table_component)

    if timestamp:
        ts_format = config.get_setting("visualization_settings", "dashboard", "timestamp_format", default='%Y-%m-%d %H:%M:%S %Z')
        timestamp_text = f"Last updated: {timestamp.strftime(ts_format)}"
        card_body_children.append(html.Small(timestamp_text, className="text-muted d-block mt-2 text-end"))

    return dbc.Card(dbc.CardBody(card_body_children))


# --- Main Layout Function ---

def create_layout(bundle: FinalAnalysisBundleV2_5, config: ConfigManagerV2_5) -> html.Div:
    """
    Creates the complete layout for the Main Dashboard mode.
    This is the single entry point called by the callback manager.
    """
    if not bundle or not bundle.processed_data_bundle or not bundle.processed_data_bundle.underlying_data_enriched:
        logger.warning("Data bundle or underlying_data_enriched is not available. Cannot render Main Dashboard.")
        return html.Div(dbc.Alert("Data bundle is not available. Cannot render Main Dashboard.", color="danger", className="m-4"))

    und_data = bundle.processed_data_bundle.underlying_data_enriched
    recommendations = bundle.active_recommendations_v2_5
    symbol = bundle.target_symbol # Get symbol for passing to helpers
    bundle_timestamp = bundle.bundle_timestamp # Get timestamp for passing to helpers

    return html.Div(
        id=ids.ID_MAIN_DASHBOARD_CONTAINER, # Ensure this ID exists in ids.py
        children=[
            dbc.Container(
                fluid=True,
                children=[
                    dbc.Row(
                        [
                            dbc.Col(_create_regime_display(und_data, config), md=6, lg=3, className="mb-4"),
                            dbc.Col(_create_flow_gauge("VAPI-FA", und_data.vapi_fa_z_score_und, ids.ID_VAPI_GAUGE, config, bundle_timestamp, symbol), md=6, lg=3, className="mb-4"),
                            dbc.Col(_create_flow_gauge("DWFD", und_data.dwfd_z_score_und, ids.ID_DWFD_GAUGE, config, bundle_timestamp, symbol), md=6, lg=3, className="mb-4"),
                            dbc.Col(_create_flow_gauge("TW-LAF", und_data.tw_laf_z_score_und, ids.ID_TW_LAF_GAUGE, config, bundle_timestamp, symbol), md=6, lg=3, className="mb-4"),
                        ], className="mt-3" # Added some top margin
                    ),
                    dbc.Row(
                        [
                            dbc.Col(_create_recommendations_table(recommendations, config, bundle_timestamp, symbol), width=12, className="mb-4"),
                        ], className="mt-3" # Added some top margin
                    ),
                ]
            )
        ]
    )
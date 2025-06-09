# dashboard_application/modes/time_decay_mode_display.py
# EOTS v2.5 - S-GRADE, AUTHORITATIVE TIME DECAY MODE DISPLAY

import logging
from typing import Dict, Optional
import pandas as pd
import plotly.graph_objects as go
from dash import html, dcc
import dash_bootstrap_components as dbc

from .. import ids
from ..utils_dashboard_v2_5 import create_empty_figure, add_timestamp_annotation, add_price_line, PLOTLY_TEMPLATE
from data_models.eots_schemas_v2_5 import FinalAnalysisBundleV2_5
from utils.config_manager_v2_5 import ConfigManagerV2_5

logger = logging.getLogger(__name__)

# --- Helper Functions for Chart Generation ---

def _generate_d_tdpi_chart(bundle: FinalAnalysisBundleV2_5, config: ConfigManagerV2_5) -> dcc.Graph:
    """Generates the D-TDPI (Dynamic Time Decay Pressure Indicator) profile by strike."""
    chart_name = "D-TDPI Pinning Pressure"
    fig_height = config.get_setting("visualization_settings", "dashboard", "time_decay_mode_settings", "tdpi_chart_height", default=500)
    metric_to_plot = 'd_tdpi_strike'
    
    try:
        strike_data = bundle.processed_data_bundle.strike_level_data_with_metrics
        if not strike_data:
            return dcc.Graph(figure=create_empty_figure(chart_name, fig_height, "Strike level data not available."))

        df_strike = pd.DataFrame([s.model_dump() for s in strike_data])
        
        if df_strike.empty or metric_to_plot not in df_strike.columns:
            return dcc.Graph(figure=create_empty_figure(chart_name, fig_height, f"'{metric_to_plot}' data not found."))

        df_plot = df_strike.dropna(subset=['strike', metric_to_plot]).sort_values('strike')
        
        colors = ['#d62728' if x < 0 else '#2ca02c' for x in df_plot[metric_to_plot]]

        fig = go.Figure(data=[go.Bar(
            x=df_plot['strike'],
            y=df_plot[metric_to_plot],
            name='D-TDPI',
            marker_color=colors,
            hovertemplate='Strike: %{x}<br>D-TDPI: %{y:,.2f}<extra></extra>'
        )])

        current_price = bundle.processed_data_bundle.underlying_data_enriched.price
        
        fig.update_layout(
            title_text=f"<b>{bundle.target_symbol}</b> - {chart_name}",
            height=fig_height,
            template=PLOTLY_TEMPLATE,
            xaxis_title="Strike Price",
            yaxis_title="D-TDPI Score",
            showlegend=False,
            xaxis={'type': 'category'}
        )
        add_price_line(fig, current_price, orientation='vertical', line_width=2, line_color='white')
        add_timestamp_annotation(fig, bundle.bundle_timestamp)

    except Exception as e:
        logger.error(f"Error creating {chart_name}: {e}", exc_info=True)
        fig = create_empty_figure(chart_name, fig_height, f"Error: {e}")

    return dcc.Graph(figure=fig)

def _generate_vci_0dte_chart(bundle: FinalAnalysisBundleV2_5, config: ConfigManagerV2_5) -> dcc.Graph:
    """Generates the VCI (Vanna Concentration Index) gauge for 0DTE options."""
    chart_name = "0DTE Vanna Concentration (VCI)"
    fig_height = config.get_setting("visualization_settings", "dashboard", "time_decay_mode_settings", "vci_chart_height", default=250)
    metric_to_plot = 'vci_0dte_agg'
    
    value = getattr(bundle.processed_data_bundle.underlying_data_enriched, metric_to_plot, None)

    if value is None or not pd.notna(value):
        fig = create_empty_figure(title=chart_name, height=fig_height, reason="0DTE VCI data not available.")
    else:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(value),
            title={'text': "VCI", 'font': {'size': 16}},
            number={'valueformat': '.3f'},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 0.5], 'tickwidth': 1},
                'bar': {'color': '#636EFA'},
                'steps': [
                    {'range': [0, 0.1], 'color': 'lightgray'},
                    {'range': [0.1, 0.25], 'color': 'gray'},
                    {'range': [0.25, 0.5], 'color': 'darkslateblue'}],
            }
        ))
        fig.update_layout(
            title_text=f"<b>{bundle.target_symbol}</b> - {chart_name}",
            height=fig_height,
            template=PLOTLY_TEMPLATE,
            margin={'t': 60, 'b': 20, 'l': 20, 'r': 20},
        )
        add_timestamp_annotation(fig, bundle.bundle_timestamp)

    return dcc.Graph(figure=fig)


# --- Main Layout Function ---

def create_layout(bundle: FinalAnalysisBundleV2_5, config: ConfigManagerV2_5) -> html.Div:
    """
    Creates the complete layout for the "Time Decay & Pinning" mode.
    """
    if not bundle or not bundle.processed_data_bundle:
        return dbc.Alert("Time Decay data is not available. Cannot render Time Decay Mode.", color="danger")

    chart_generators = {
        "tdpi_strike_viz": (_generate_d_tdpi_chart, (bundle, config)),
        "vci_0dte_strike_viz": (_generate_vci_0dte_chart, (bundle, config)),
    }
    
    charts_to_display = config.get_setting('visualization_settings', 'dashboard', 'modes_detail_config', 'timedecay', 'charts', default=[])
    
    chart_components = []
    for chart_id in charts_to_display:
        if chart_id in chart_generators:
            generator_func, args = chart_generators[chart_id]
            # D-TDPI is the main chart, VCI is a supporting KPI
            col_width = 12 if chart_id == "tdpi_strike_viz" else 6
            chart_components.append(dbc.Col(generator_func(*args), md=col_width, className="mb-4"))
        else:
            logger.warning(f"No generator found for chart ID '{chart_id}' in time_decay_mode_display.")

    return html.Div([
        dbc.Container(
            fluid=True,
            children=[dbc.Row(chart_components, justify="center")]
        )
    ])
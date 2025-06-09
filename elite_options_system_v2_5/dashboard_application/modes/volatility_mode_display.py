# dashboard_application/modes/volatility_mode_display.py
# EOTS v2.5 - S-GRADE, AUTHORITATIVE VOLATILITY MODE DISPLAY

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

def _generate_vri_2_0_chart(bundle: FinalAnalysisBundleV2_5, config: ConfigManagerV2_5) -> dcc.Graph:
    """Generates the VRI 2.0 (Volatility Regime Indicator) profile by strike."""
    chart_name = "VRI 2.0 Sensitivity Profile"
    fig_height = config.get_setting("visualization_settings", "dashboard", "volatility_mode_settings", "vri_chart_height", default=500)
    metric_to_plot = 'vri_2_0_strike'
    
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
            name='VRI 2.0',
            marker_color=colors,
            hovertemplate='Strike: %{x}<br>VRI 2.0: %{y:,.2f}<extra></extra>'
        )])

        current_price = bundle.processed_data_bundle.underlying_data_enriched.price
        
        fig.update_layout(
            title_text=f"<b>{bundle.target_symbol}</b> - {chart_name}",
            height=fig_height,
            template=PLOTLY_TEMPLATE,
            xaxis_title="Strike Price",
            yaxis_title="VRI 2.0 Score",
            showlegend=False,
            xaxis={'type': 'category'}
        )
        add_price_line(fig, current_price, orientation='vertical', line_width=2, line_color='white')
        add_timestamp_annotation(fig, bundle.bundle_timestamp)

    except Exception as e:
        logger.error(f"Error creating {chart_name}: {e}", exc_info=True)
        fig = create_empty_figure(chart_name, fig_height, f"Error: {e}")

    return dcc.Graph(figure=fig)

def _generate_0dte_vol_gauge(bundle: FinalAnalysisBundleV2_5, config: ConfigManagerV2_5, metric: str, title: str) -> dcc.Graph:
    """Generic helper to create a gauge for 0DTE volatility metrics."""
    fig_height = config.get_setting("visualization_settings", "dashboard", "volatility_mode_settings", "gauge_height", default=250)
    value = getattr(bundle.processed_data_bundle.underlying_data_enriched, metric, None)

    if value is None or not pd.notna(value):
        fig = create_empty_figure(title=title, height=fig_height, reason="0DTE data N/A")
    else:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(value),
            title={'text': title, 'font': {'size': 16}},
            gauge={'axis': {'range': [-2, 2]}} # Example range, can be configured
        ))
        fig.update_layout(
            title_text=f"<b>{bundle.target_symbol}</b> - {title}",
            height=fig_height,
            template=PLOTLY_TEMPLATE,
            margin={'t': 60, 'b': 20, 'l': 20, 'r': 20},
        )
        add_timestamp_annotation(fig, bundle.bundle_timestamp)
        
    return dcc.Graph(figure=fig)


# --- Main Layout Function ---

def create_layout(bundle: FinalAnalysisBundleV2_5, config: ConfigManagerV2_5) -> html.Div:
    """
    Creates the complete layout for the "Volatility Deep Dive" mode.
    """
    if not bundle or not bundle.processed_data_bundle:
        return dbc.Alert("Volatility data is not available. Cannot render Volatility Mode.", color="danger")

    # This mapping defines which charts to build for this mode.
    chart_generators = {
        "vri_2_0_strike_profile": (_generate_vri_2_0_chart, (bundle, config)),
        "vri_0dte_strike_viz": (_generate_0dte_vol_gauge, (bundle, config, "vri_0dte_und_sum", "0DTE VRI Sum")),
        "vfi_0dte_agg_viz": (_generate_0dte_vol_gauge, (bundle, config, "vfi_0dte_und_sum", "0DTE VFI Sum")),
        "vvr_0dte_agg_viz": (_generate_0dte_vol_gauge, (bundle, config, "vvr_0dte_und_avg", "0DTE VVR Avg")),
    }
    
    charts_to_display = config.get_setting('visualization_settings', 'dashboard', 'modes_detail_config', 'volatility', 'charts', default=[])
    
    # Separate main chart from KPI gauges
    main_chart_components = []
    kpi_components = []

    for chart_id in charts_to_display:
        if chart_id in chart_generators:
            generator_func, args = chart_generators[chart_id]
            # Use chart ID to decide if it's a main chart or a KPI
            if chart_id == "vri_2_0_strike_profile":
                main_chart_components.append(dbc.Col(generator_func(*args), width=12, className="mb-4"))
            else:
                kpi_components.append(dbc.Col(generator_func(*args), md=4, className="mb-4"))
        else:
            logger.warning(f"No generator found for chart ID '{chart_id}' in volatility_mode_display.")

    return html.Div([
        dbc.Container(
            fluid=True,
            children=[
                # Row for 0DTE KPI Gauges
                dbc.Row(kpi_components, justify="center", className="mt-4"),
                # Row for the main VRI 2.0 chart
                dbc.Row(main_chart_components)
            ]
        )
    ])
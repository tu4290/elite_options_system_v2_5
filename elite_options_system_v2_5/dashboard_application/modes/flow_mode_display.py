# dashboard_application/modes/flow_mode_display_v2_5.py
# EOTS v2.5 - S-GRADE, AUTHORITATIVE FLOW MODE DISPLAY

import logging
from typing import Dict, Any, Optional, List
import pandas as pd
import plotly.graph_objects as go
from dash import html, dcc
import dash_bootstrap_components as dbc

from .. import ids
from ..utils_dashboard_v2_5 import (
    create_empty_figure, add_timestamp_annotation, add_price_line, PLOTLY_TEMPLATE
)
from data_models.eots_schemas_v2_5 import FinalAnalysisBundleV2_5
from utils.config_manager_v2_5 import ConfigManagerV2_5

logger = logging.getLogger(__name__)

# --- Helper Functions for Chart Generation ---

def _generate_net_value_heatmap(bundle: FinalAnalysisBundleV2_5, config: ConfigManagerV2_5) -> dcc.Graph:
    """Generates a heatmap of net value pressure by strike and option type."""
    chart_name = "Net Value by Strike"
    fig_height = config.get_setting("visualization_settings", "dashboard", "flow_mode_settings", "net_value_heatmap", "height", default=500)

    try:
        df_chain = pd.DataFrame([c.model_dump() for c in bundle.processed_data_bundle.options_data_with_metrics])
        if df_chain.empty or 'value_bs' not in df_chain.columns:
            return dcc.Graph(figure=create_empty_figure(chart_name, fig_height, "Per-contract 'value_bs' data not available."))

        df_plot = df_chain.dropna(subset=['strike', 'opt_kind', 'value_bs'])
        pivot_df = df_plot.pivot_table(index='strike', columns='opt_kind', values='value_bs', aggfunc='sum').fillna(0)
        
        if 'call' not in pivot_df: pivot_df['call'] = 0
        if 'put' not in pivot_df: pivot_df['put'] = 0
        pivot_df = pivot_df[['put', 'call']].sort_index(ascending=False)

        if pivot_df.empty:
            return dcc.Graph(figure=create_empty_figure(chart_name, fig_height, "No data to pivot."))

        fig = go.Figure(data=go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns.str.capitalize(),
            y=pivot_df.index.astype(str),
            colorscale='RdYlGn',
            zmid=0,
            hoverongaps=False,
            hovertemplate='Strike: %{y}<br>Type: %{x}<br>Net Value: %{z:$,.0f}<extra></extra>'
        ))
        
        fig.update_layout(
            title_text=f"<b>{bundle.target_symbol}</b> - {chart_name}",
            height=fig_height,
            template=PLOTLY_TEMPLATE,
            yaxis=dict(type='category')
        )
        add_price_line(fig, bundle.processed_data_bundle.underlying_data_enriched.price, orientation='horizontal')
        add_timestamp_annotation(fig, bundle.bundle_timestamp)

    except Exception as e:
        logger.error(f"Error creating {chart_name}: {e}", exc_info=True)
        fig = create_empty_figure(chart_name, fig_height, f"Error: {e}")

    return dcc.Graph(figure=fig)

def _generate_greek_flow_chart(bundle: FinalAnalysisBundleV2_5, metric: str, title: str, color: str) -> dcc.Graph:
    """Generic helper to create a bar chart for a net customer Greek flow metric."""
    chart_name = f"{title} by Strike"
    fig = go.Figure()
    
    try:
        df_strike = pd.DataFrame([s.model_dump() for s in bundle.processed_data_bundle.strike_level_data_with_metrics])
        if not df_strike.empty and metric in df_strike.columns:
            df_plot = df_strike.dropna(subset=['strike', metric]).sort_values('strike')
            
            fig.add_trace(go.Bar(
                x=df_plot['strike'],
                y=df_plot[metric],
                name=title,
                marker_color=color
            ))
            fig.update_layout(
                title_text=f"<b>{bundle.target_symbol}</b> - {chart_name}",
                height=400,
                template=PLOTLY_TEMPLATE,
                showlegend=False
            )
            add_price_line(fig, bundle.processed_data_bundle.underlying_data_enriched.price)
            add_timestamp_annotation(fig, bundle.bundle_timestamp)
        else:
            fig = create_empty_figure(chart_name, 400, "Metric data not available.")
    except Exception as e:
        logger.error(f"Error creating {chart_name}: {e}", exc_info=True)
        fig = create_empty_figure(chart_name, 400, f"Error: {e}")

    return dcc.Graph(figure=fig)

# --- Main Layout Function ---

def create_layout(bundle: FinalAnalysisBundleV2_5, config: ConfigManagerV2_5) -> html.Div:
    """
    Creates the complete layout for the "Flow Breakdown" mode.
    This is the single entry point called by the callback manager.
    """
    if not bundle or not bundle.processed_data_bundle:
        return dbc.Alert("Flow data is not available. Cannot render Flow Mode.", color="danger")

    # This mapping defines which charts to build for this mode.
    # It maps the chart ID from the config to the generator function and its specific arguments.
    chart_generators = {
        "net_value_heatmap_viz": (_generate_net_value_heatmap, (bundle, config)),
        "net_cust_delta_flow_viz": (_generate_greek_flow_chart, (bundle, "net_cust_delta_flow_at_strike", "Net Delta Flow", "#17becf")),
        "net_cust_gamma_flow_viz": (_generate_greek_flow_chart, (bundle, "net_cust_gamma_flow_at_strike", "Net Gamma Flow", "#e377c2")),
        "net_cust_vega_flow_viz": (_generate_greek_flow_chart, (bundle, "net_cust_vega_flow_at_strike", "Net Vega Flow", "#ff7f0e")),
    }
    
    # Get the list of charts to display for 'flow' mode from the config
    charts_to_display = config.get_setting('visualization_settings', 'dashboard', 'modes_detail_config', 'flow', 'charts', default=[])
    
    chart_components = []
    for chart_id in charts_to_display:
        if chart_id in chart_generators:
            generator_func, args = chart_generators[chart_id]
            chart_components.append(dbc.Col(generator_func(*args), md=12, lg=6, className="mb-4"))
        else:
            logger.warning(f"No generator found for chart ID '{chart_id}' in flow_mode_display.")

    return html.Div([
        dbc.Container(
            fluid=True,
            children=[dbc.Row(chart_components)]
        )
    ])
# dashboard_application/modes/structure_mode_display_v2_5.py
# EOTS v2.5 - S-GRADE, AUTHORITATIVE STRUCTURE MODE DISPLAY

import logging
from typing import Dict, Optional
import pandas as pd
import plotly.graph_objects as go
from dash import html, dcc
import dash_bootstrap_components as dbc

from .. import ids
from ..utils_dashboard_v2_5 import create_empty_figure, add_timestamp_annotation, add_price_line, PLOTLY_TEMPLATE
from ...data_models.eots_schemas_v2_5 import FinalAnalysisBundleV2_5
from ...utils.config_manager_v2_5 import ConfigManagerV2_5

logger = logging.getLogger(__name__)

# --- Helper Function for Chart Generation ---

def _generate_a_mspi_profile_chart(bundle: FinalAnalysisBundleV2_5, config: ConfigManagerV2_5) -> dcc.Graph:
    """
    Generates the primary visualization for Structure Mode: the Adaptive MSPI profile.
    This chart shows the synthesized structural pressure across different strikes.
    """
    chart_name = "Adaptive MSPI Profile"
    fig_height = config.get_setting("visualization_settings", "dashboard", "structure_mode_settings", "mspi_chart_height", default=600)
    
    try:
        strike_data = bundle.processed_data_bundle.strike_level_data_with_metrics
        if not strike_data:
            return dcc.Graph(figure=create_empty_figure(chart_name, fig_height, "Strike level data not available."))

        df_strike = pd.DataFrame([s.model_dump() for s in strike_data])
        
        # A-MSPI is a summary score calculated at the underlying level, but its components are at strike level.
        # For this visualization, we will use a key component like 'a_dag_strike' as the primary bar chart.
        # A true A-MSPI chart might be a single gauge or an overlay of its components.
        metric_to_plot = 'a_dag_strike' # Using A-DAG as the main visual for structural pressure
        
        if df_strike.empty or metric_to_plot not in df_strike.columns:
            return dcc.Graph(figure=create_empty_figure(chart_name, fig_height, f"'{metric_to_plot}' data not found."))

        df_plot = df_strike.dropna(subset=['strike', metric_to_plot]).sort_values('strike')
        
        colors = ['#d62728' if x < 0 else '#2ca02c' for x in df_plot[metric_to_plot]]

        fig = go.Figure(data=[go.Bar(
            x=df_plot['strike'],
            y=df_plot[metric_to_plot],
            name='A-DAG',
            marker_color=colors,
            hovertemplate='Strike: %{x}<br>A-DAG: %{y:,.2f}<extra></extra>'
        )])

        current_price = bundle.processed_data_bundle.underlying_data_enriched.price
        
        fig.update_layout(
            title_text=f"<b>{bundle.target_symbol}</b> - {chart_name} (using A-DAG)",
            height=fig_height,
            template=PLOTLY_TEMPLATE,
            xaxis_title="Strike Price",
            yaxis_title="Adaptive Delta-Adjusted Gamma (A-DAG)",
            showlegend=False,
            barmode='relative',
            xaxis={'type': 'category'} # Treat strikes as categories for clear separation
        )
        add_price_line(fig, current_price, orientation='vertical', line_width=2, line_color='white')
        add_timestamp_annotation(fig, bundle.bundle_timestamp)

    except Exception as e:
        logger.error(f"Error creating {chart_name}: {e}", exc_info=True)
        fig = create_empty_figure(chart_name, fig_height, f"Error: {e}")

    return dcc.Graph(figure=fig)


# --- Main Layout Function ---

def create_layout(bundle: FinalAnalysisBundleV2_5, config: ConfigManagerV2_5) -> html.Div:
    """
    Creates the complete layout for the "Structure & Dealer Positioning" mode.
    This is the single entry point called by the callback manager.
    """
    if not bundle or not bundle.processed_data_bundle:
        return dbc.Alert("Structural data is not available. Cannot render Structure Mode.", color="danger")

    # This mapping defines which charts to build for this mode.
    chart_generators = {
        "mspi_components_viz": (_generate_a_mspi_profile_chart, (bundle, config)),
        # Add other structural charts here if needed, following the same pattern
        # "sdag_multiplicative_viz": (_generate_sdag_chart, (bundle, config, 'e_sdag_mult_strike')),
    }
    
    charts_to_display = config.get_setting('visualization_settings', 'dashboard', 'modes_detail_config', 'structure', 'charts', default=[])
    
    chart_components = []
    for chart_id in charts_to_display:
        if chart_id in chart_generators:
            generator_func, args = chart_generators[chart_id]
            # For structure mode, we'll make the primary chart full width
            chart_components.append(dbc.Col(generator_func(*args), width=12, className="mb-4"))
        else:
            logger.warning(f"No generator found for chart ID '{chart_id}' in structure_mode_display.")

    return html.Div([
        dbc.Container(
            fluid=True,
            children=[dbc.Row(chart_components)]
        )
    ])
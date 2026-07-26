"""
visualizations.py — All Plotly chart functions for the Pension Fund Analytics Platform.

Every function returns a plotly.graph_objects.Figure with a consistent dark theme.
Dark theme: #0D1117 background, #161B22 plot area, white text, subtle grids.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Shared dark theme definition
# ---------------------------------------------------------------------------
CHART_BG       = "#0D1117"
PLOT_BG        = "#161B22"
GRID_COLOR     = "#30363D"
FONT_COLOR     = "#E6EDF3"
ACCENT_COLOR   = "#00D4FF"
POSITIVE_COLOR = "#00C853"
NEGATIVE_COLOR = "#FF4B4B"
WARN_COLOR     = "#FFB300"

COLOR_SEQUENCE = [
    "#00D4FF", "#7B68EE", "#32CD32", "#FF6B35",
    "#FFD700", "#FF69B4", "#98FB98", "#FFA07A",
    "#87CEEB", "#DDA0DD", "#20B2AA", "#F0E68C",
]

PLOTLY_TEMPLATE = {
    "paper_bgcolor": CHART_BG,
    "plot_bgcolor":  PLOT_BG,
    "font":          {"color": FONT_COLOR, "family": "Inter, system-ui, -apple-system, sans-serif", "size": 12},
    "xaxis":         {"gridcolor": GRID_COLOR, "linecolor": GRID_COLOR, "tickcolor": FONT_COLOR},
    "yaxis":         {"gridcolor": GRID_COLOR, "linecolor": GRID_COLOR, "tickcolor": FONT_COLOR},
    "legend":        {"bgcolor": "rgba(22,27,34,0.8)", "bordercolor": GRID_COLOR, "borderwidth": 1},
    "colorway":      COLOR_SEQUENCE,
    "margin":        {"l": 60, "r": 30, "t": 60, "b": 50},
}


def apply_dark_theme(fig: go.Figure, title: Optional[str] = None) -> go.Figure:
    """Apply the shared dark theme to any Plotly figure."""
    fig.update_layout(
        paper_bgcolor = CHART_BG,
        plot_bgcolor  = PLOT_BG,
        font          = PLOTLY_TEMPLATE["font"],
        legend        = PLOTLY_TEMPLATE["legend"],
        margin        = PLOTLY_TEMPLATE["margin"],
        colorway      = COLOR_SEQUENCE,
    )
    if title:
        fig.update_layout(title={"text": title, "x": 0.02, "font": {"size": 16, "color": FONT_COLOR}})
    fig.update_xaxes(gridcolor=GRID_COLOR, linecolor=GRID_COLOR, tickcolor=FONT_COLOR, showline=True)
    fig.update_yaxes(gridcolor=GRID_COLOR, linecolor=GRID_COLOR, tickcolor=FONT_COLOR, showline=True)
    return fig


# ---------------------------------------------------------------------------
# 1. Portfolio Value Trend
# ---------------------------------------------------------------------------

def plot_portfolio_value_trend(portfolio_value_series: pd.Series) -> go.Figure:
    """
    Line chart of portfolio value over time with gradient fill.
    Y-axis in £M notation.
    """
    series = portfolio_value_series / 1e6  # Convert to £M

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x        = series.index,
        y        = series.values,
        mode     = "lines",
        name     = "Portfolio Value",
        line     = {"color": ACCENT_COLOR, "width": 2},
        fill     = "tozeroy",
        fillcolor= "rgba(0, 212, 255, 0.08)",
        hovertemplate = "£%{y:.2f}M<extra></extra>",
    ))
    fig = apply_dark_theme(fig, "Portfolio Value (£M)")
    fig.update_yaxes(tickprefix="£", ticksuffix="M")
    fig.update_xaxes(title_text="Date")
    return fig


# ---------------------------------------------------------------------------
# 2. Asset Allocation Donut
# ---------------------------------------------------------------------------

def plot_asset_allocation_donut(asset_class_weights: Dict[str, float]) -> go.Figure:
    """Donut chart of asset class weights."""
    labels  = list(asset_class_weights.keys())
    values  = [v * 100 for v in asset_class_weights.values()]
    colors  = [
        {"Equity": "#00D4FF", "ETF": "#7B68EE", "Bond": "#32CD32", "Cash": "#FFD700"}.get(l, "#888888")
        for l in labels
    ]

    fig = go.Figure(go.Pie(
        labels           = labels,
        values           = values,
        hole             = 0.6,
        marker_colors    = colors,
        textinfo         = "label+percent",
        textfont_size    = 11,
        hovertemplate    = "<b>%{label}</b><br>%{value:.1f}%<extra></extra>",
        insidetextorientation = "radial",
    ))
    fig.update_layout(
        annotations = [{"text": "Asset<br>Allocation", "x": 0.5, "y": 0.5,
                        "font_size": 13, "showarrow": False, "font_color": FONT_COLOR}],
        showlegend  = True,
    )
    return apply_dark_theme(fig, "Asset Class Allocation")


# ---------------------------------------------------------------------------
# 3. Sector Allocation Bar
# ---------------------------------------------------------------------------

def plot_sector_allocation_bar(sector_weights: Dict[str, float]) -> go.Figure:
    """Horizontal bar chart of sector weights."""
    df = pd.Series(sector_weights).sort_values(ascending=True) * 100
    fig = go.Figure(go.Bar(
        x           = df.values,
        y           = df.index,
        orientation = "h",
        marker_color = COLOR_SEQUENCE[:len(df)],
        text        = [f"{v:.1f}%" for v in df.values],
        textposition = "outside",
        hovertemplate = "<b>%{y}</b><br>%{x:.2f}%<extra></extra>",
    ))
    fig = apply_dark_theme(fig, "Sector Allocation")
    fig.update_xaxes(title_text="Weight (%)", ticksuffix="%")
    fig.update_yaxes(title_text="Sector")
    fig.update_layout(showlegend=False, bargap=0.25)
    return fig


# ---------------------------------------------------------------------------
# 4. Geographic Allocation
# ---------------------------------------------------------------------------

def plot_geographic_allocation(geo_weights: Dict[str, float]) -> go.Figure:
    """Donut chart of geographic weights."""
    labels = list(geo_weights.keys())
    values = [v * 100 for v in geo_weights.values()]
    fig = go.Figure(go.Pie(
        labels        = labels,
        values        = values,
        hole          = 0.55,
        textinfo      = "label+percent",
        textfont_size = 11,
        hovertemplate = "<b>%{label}</b><br>%{value:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        annotations=[{"text": "Geography", "x": 0.5, "y": 0.5,
                      "font_size": 12, "showarrow": False, "font_color": FONT_COLOR}]
    )
    return apply_dark_theme(fig, "Geographic Allocation")


# ---------------------------------------------------------------------------
# 5. Cumulative Returns vs Benchmark
# ---------------------------------------------------------------------------

def plot_cumulative_returns(
    portfolio_cum: pd.Series,
    benchmark_cum: pd.Series,
    labels: List[str] = None,
) -> go.Figure:
    """Dual line chart: portfolio cumulative return vs benchmark."""
    labels = labels or ["Portfolio", "S&P 500 (SPY)"]
    port_pct  = (portfolio_cum - 1) * 100
    bench_pct = (benchmark_cum - 1) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=port_pct.index, y=port_pct.values, mode="lines",
        name=labels[0], line={"color": ACCENT_COLOR, "width": 2},
        hovertemplate=f"{labels[0]}: %{{y:.2f}}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=bench_pct.index, y=bench_pct.values, mode="lines",
        name=labels[1], line={"color": "#7B68EE", "width": 2, "dash": "dash"},
        hovertemplate=f"{labels[1]}: %{{y:.2f}}%<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color=GRID_COLOR)
    fig = apply_dark_theme(fig, "Cumulative Returns vs Benchmark")
    fig.update_yaxes(title_text="Cumulative Return (%)", ticksuffix="%")
    fig.update_xaxes(title_text="Date")
    return fig


# ---------------------------------------------------------------------------
# 6. Rolling Returns
# ---------------------------------------------------------------------------

def plot_rolling_returns(rolling_returns: pd.Series, window_label: str = "21-Day") -> go.Figure:
    """Line chart of rolling returns with zero reference line."""
    series_pct = rolling_returns * 100
    colors     = [POSITIVE_COLOR if v >= 0 else NEGATIVE_COLOR for v in series_pct.values]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series_pct.index, y=series_pct.values, mode="lines",
        name=f"Rolling {window_label} Return",
        line={"color": ACCENT_COLOR, "width": 1.5},
        hovertemplate="%{y:.2f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color=GRID_COLOR, line_width=1)
    fig = apply_dark_theme(fig, f"Rolling {window_label} Return")
    fig.update_yaxes(title_text="Return (%)", ticksuffix="%")
    fig.update_xaxes(title_text="Date")
    return fig


# ---------------------------------------------------------------------------
# 7. Monthly Return Heatmap
# ---------------------------------------------------------------------------

def plot_monthly_return_heatmap(monthly_returns_pivot: pd.DataFrame) -> go.Figure:
    """
    Heatmap with month columns, year rows.
    Green = positive, Red = negative.
    """
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot = monthly_returns_pivot.copy() * 100  # Convert to %

    # Map column numbers (1-12) to month labels
    col_map = {i+1: m for i, m in enumerate(month_labels)}
    pivot.columns = [col_map.get(c, str(c)) for c in pivot.columns]

    z_values = pivot.values
    text_values = [[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in z_values]

    fig = go.Figure(go.Heatmap(
        z            = z_values,
        x            = list(pivot.columns),
        y            = [str(y) for y in pivot.index],
        text         = text_values,
        texttemplate = "%{text}",
        textfont     = {"size": 10},
        colorscale   = [[0.0, "#FF4B4B"], [0.5, PLOT_BG], [1.0, "#00C853"]],
        zmid         = 0,
        showscale    = True,
        colorbar     = {"title": "%", "ticksuffix": "%"},
        hovertemplate= "Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>",
    ))
    fig = apply_dark_theme(fig, "Monthly Return Heatmap (%)")
    fig.update_xaxes(title_text="Month")
    fig.update_yaxes(title_text="Year")
    return fig


# ---------------------------------------------------------------------------
# 8. Drawdown Chart
# ---------------------------------------------------------------------------

def plot_drawdown(drawdown_series: pd.Series) -> go.Figure:
    """Area chart of drawdown series, filled below zero in red."""
    series_pct = drawdown_series * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x        = series_pct.index,
        y        = series_pct.values,
        mode     = "lines",
        name     = "Drawdown",
        line     = {"color": NEGATIVE_COLOR, "width": 1.5},
        fill     = "tozeroy",
        fillcolor= "rgba(255, 75, 75, 0.25)",
        hovertemplate="%{y:.2f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color=GRID_COLOR, line_width=1)
    fig = apply_dark_theme(fig, "Portfolio Drawdown")
    fig.update_yaxes(title_text="Drawdown (%)", ticksuffix="%")
    fig.update_xaxes(title_text="Date")
    return fig


# ---------------------------------------------------------------------------
# 9. Correlation Matrix
# ---------------------------------------------------------------------------

def plot_correlation_matrix(corr_matrix: pd.DataFrame) -> go.Figure:
    """Annotated heatmap of asset return correlations."""
    labels = list(corr_matrix.columns)
    z      = corr_matrix.values
    text   = [[f"{v:.2f}" for v in row] for row in z]

    fig = go.Figure(go.Heatmap(
        z            = z,
        x            = labels,
        y            = labels,
        text         = text,
        texttemplate = "%{text}",
        textfont     = {"size": 9},
        colorscale   = "RdBu",
        zmid         = 0,
        zmin         = -1,
        zmax         = 1,
        showscale    = True,
        colorbar     = {"title": "ρ"},
        hovertemplate= "%{x} / %{y}<br>ρ = %{z:.3f}<extra></extra>",
    ))
    fig = apply_dark_theme(fig, "Return Correlation Matrix")
    fig.update_layout(margin={"l": 80, "r": 30, "t": 60, "b": 80})
    return fig


# ---------------------------------------------------------------------------
# 10. Return Distribution
# ---------------------------------------------------------------------------

def plot_return_distribution(returns: pd.Series) -> go.Figure:
    """
    Histogram of daily returns with KDE overlay and VaR reference lines.
    """
    try:
        from pension_fund.risk_analytics import calculate_var, calculate_cvar
    except ImportError:
        from risk_analytics import calculate_var, calculate_cvar
    clean   = returns.dropna() * 100  # Convert to %
    var_95  = -calculate_var(returns, 0.95) * 100
    cvar_95 = -calculate_cvar(returns, 0.95) * 100

    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Histogram(
        x          = clean.values,
        nbinsx     = 60,
        name       = "Daily Returns",
        opacity    = 0.6,
        marker_color = ACCENT_COLOR,
        hovertemplate="Return: %{x:.2f}%<br>Count: %{y}<extra></extra>",
    ))

    # KDE overlay
    from scipy.stats import gaussian_kde
    try:
        kde    = gaussian_kde(clean.values, bw_method="silverman")
        x_range = np.linspace(clean.min(), clean.max(), 200)
        kde_y   = kde(x_range)
        # Scale KDE to match histogram counts
        bin_width = (clean.max() - clean.min()) / 60
        kde_scaled = kde_y * len(clean) * bin_width
        fig.add_trace(go.Scatter(
            x=x_range, y=kde_scaled, mode="lines",
            name="KDE", line={"color": WARN_COLOR, "width": 2},
            hovertemplate="Return: %{x:.2f}%<br>Density: %{y:.2f}<extra></extra>",
        ))
    except Exception:
        pass

    # VaR lines
    fig.add_vline(x=var_95, line_dash="dash", line_color=NEGATIVE_COLOR, line_width=1.5,
                  annotation_text=f"VaR 95%: {var_95:.2f}%",
                  annotation_position="top left",
                  annotation_font_color=NEGATIVE_COLOR)
    fig.add_vline(x=cvar_95, line_dash="dot", line_color="#FF8C00", line_width=1.5,
                  annotation_text=f"CVaR 95%: {cvar_95:.2f}%",
                  annotation_position="bottom left",
                  annotation_font_color="#FF8C00")

    fig = apply_dark_theme(fig, "Daily Return Distribution")
    fig.update_xaxes(title_text="Daily Return (%)", ticksuffix="%")
    fig.update_yaxes(title_text="Frequency")
    return fig


# ---------------------------------------------------------------------------
# 11. Risk Contribution
# ---------------------------------------------------------------------------

def plot_risk_contribution(risk_contrib_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of risk contribution per asset."""
    if risk_contrib_df.empty:
        fig = go.Figure()
        return apply_dark_theme(fig, "Risk Contribution")

    df = risk_contrib_df.sort_values("Risk Contribution (%)", ascending=True)
    fig = go.Figure(go.Bar(
        x           = df["Risk Contribution (%)"].values,
        y           = df["Ticker"].values,
        orientation = "h",
        marker_color = COLOR_SEQUENCE[:len(df)],
        text        = [f"{v:.1f}%" for v in df["Risk Contribution (%)"].values],
        textposition = "outside",
        hovertemplate="<b>%{y}</b><br>Risk Contribution: %{x:.2f}%<extra></extra>",
    ))
    fig = apply_dark_theme(fig, "Risk Contribution by Asset")
    fig.update_xaxes(title_text="Risk Contribution (%)", ticksuffix="%")
    fig.update_yaxes(title_text="")
    fig.update_layout(showlegend=False, bargap=0.25)
    return fig


# ---------------------------------------------------------------------------
# 12. Holdings Scatter (Risk vs Return Bubble)
# ---------------------------------------------------------------------------

def plot_holdings_scatter(
    holdings_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    weights: Dict[str, float],
) -> go.Figure:
    """
    Bubble chart: annualised return (y) vs annualised volatility (x).
    Bubble size = portfolio weight.
    """
    try:
        from pension_fund.data_processing import calculate_daily_returns, calculate_annualised_return
    except ImportError:
        from data_processing import calculate_daily_returns, calculate_annualised_return

    records = []
    for _, row in holdings_df.iterrows():
        ticker = row["Ticker"]
        if ticker not in returns_df.columns:
            continue
        rets    = returns_df[ticker].dropna()
        ann_ret = calculate_annualised_return(rets) * 100
        ann_vol = rets.std() * np.sqrt(252) * 100
        weight  = weights.get(ticker, 0.0) * 100
        records.append({
            "Ticker":      ticker,
            "Asset Class": row.get("Asset Class", ""),
            "Sector":      row.get("Sector", ""),
            "Return (%)":  ann_ret,
            "Volatility (%)": ann_vol,
            "Weight (%)":  weight,
        })

    if not records:
        fig = go.Figure()
        return apply_dark_theme(fig, "Risk vs Return")

    df = pd.DataFrame(records)
    fig = px.scatter(
        df, x="Volatility (%)", y="Return (%)",
        size="Weight (%)", color="Sector",
        text="Ticker",
        size_max=60,
        color_discrete_sequence=COLOR_SEQUENCE,
        hover_data={"Ticker": True, "Asset Class": True, "Return (%)": ":.1f",
                    "Volatility (%)": ":.1f", "Weight (%)": ":.1f"},
    )
    fig.update_traces(textposition="top center", textfont_size=9)
    fig = apply_dark_theme(fig, "Risk vs Return (Bubble Size = Weight)")
    fig.update_xaxes(title_text="Annualised Volatility (%)", ticksuffix="%")
    fig.update_yaxes(title_text="Annualised Return (%)", ticksuffix="%")
    return fig


# ---------------------------------------------------------------------------
# 13. Rolling Volatility
# ---------------------------------------------------------------------------

def plot_rolling_volatility(rolling_vol_series: pd.Series) -> go.Figure:
    """Line chart of rolling annualised volatility."""
    series_pct = rolling_vol_series * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series_pct.index, y=series_pct.values, mode="lines",
        name="Rolling Volatility",
        line={"color": WARN_COLOR, "width": 2},
        fill="tozeroy",
        fillcolor="rgba(255, 179, 0, 0.08)",
        hovertemplate="%{y:.2f}%<extra></extra>",
    ))
    fig = apply_dark_theme(fig, "Rolling 21-Day Annualised Volatility")
    fig.update_yaxes(title_text="Annualised Volatility (%)", ticksuffix="%")
    fig.update_xaxes(title_text="Date")
    return fig


# ---------------------------------------------------------------------------
# 14. Performance Attribution
# ---------------------------------------------------------------------------

def plot_performance_attribution(attribution_df: pd.DataFrame) -> go.Figure:
    """Waterfall-style bar chart of each holding's contribution to excess return."""
    if attribution_df.empty:
        fig = go.Figure()
        return apply_dark_theme(fig, "Performance Attribution")

    df = attribution_df.sort_values("Contribution", ascending=False)
    colors = [POSITIVE_COLOR if v >= 0 else NEGATIVE_COLOR for v in df["Contribution"].values]
    values_pct = df["Contribution"] * 100

    fig = go.Figure(go.Bar(
        x           = df["Ticker"].values,
        y           = values_pct.values,
        marker_color = colors,
        text        = [f"{v:.2f}%" for v in values_pct.values],
        textposition = "outside",
        hovertemplate="<b>%{x}</b><br>Contribution: %{y:.3f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color=GRID_COLOR, line_width=1)
    fig = apply_dark_theme(fig, "Performance Attribution vs Benchmark")
    fig.update_yaxes(title_text="Contribution to Excess Return (%)", ticksuffix="%")
    fig.update_xaxes(title_text="Holding")
    fig.update_layout(showlegend=False, bargap=0.3)
    return fig


# ---------------------------------------------------------------------------
# 15. Stress Test Results
# ---------------------------------------------------------------------------

def plot_stress_test_results(scenarios_df: pd.DataFrame) -> go.Figure:
    """
    Bar chart showing portfolio impact (£M) per stress scenario.
    scenarios_df: DataFrame with columns [Scenario, Impact (£M), Impact (%)].
    """
    if scenarios_df.empty:
        fig = go.Figure()
        return apply_dark_theme(fig, "Stress Test Results")

    colors = [NEGATIVE_COLOR if v < 0 else POSITIVE_COLOR for v in scenarios_df["Impact (£M)"].values]

    fig = go.Figure(go.Bar(
        x           = scenarios_df["Scenario"].values,
        y           = scenarios_df["Impact (£M)"].values,
        marker_color = colors,
        text        = [f"£{v:.1f}M" for v in scenarios_df["Impact (£M)"].values],
        textposition = "outside",
        hovertemplate="<b>%{x}</b><br>Impact: £%{y:.1f}M<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color=GRID_COLOR, line_width=1)
    fig = apply_dark_theme(fig, "Stress Test — Portfolio Impact (£M)")
    fig.update_yaxes(title_text="Portfolio Impact (£M)", tickprefix="£", ticksuffix="M")
    fig.update_xaxes(title_text="Scenario")
    fig.update_layout(showlegend=False, bargap=0.3)
    return fig


# ---------------------------------------------------------------------------
# 16. Top Gainers / Losers
# ---------------------------------------------------------------------------

def plot_top_gainers_losers(holdings_returns_df: pd.DataFrame, top_n: int = 5) -> go.Figure:
    """
    Horizontal bar chart showing top N gainers and losers (YTD returns per holding).
    """
    if holdings_returns_df.empty:
        fig = go.Figure()
        return apply_dark_theme(fig, "Top Gainers & Losers")

    df = holdings_returns_df.sort_values("Return (%)")
    gainers = df.tail(top_n)
    losers  = df.head(top_n)
    combined = pd.concat([losers, gainers]).drop_duplicates()

    colors = [POSITIVE_COLOR if v >= 0 else NEGATIVE_COLOR for v in combined["Return (%)"].values]

    fig = go.Figure(go.Bar(
        x           = combined["Return (%)"].values,
        y           = combined["Ticker"].values,
        orientation = "h",
        marker_color = colors,
        text        = [f"{v:+.1f}%" for v in combined["Return (%)"].values],
        textposition = "outside",
        hovertemplate="<b>%{y}</b><br>Return: %{x:+.2f}%<extra></extra>",
    ))
    fig.add_vline(x=0, line_dash="dash", line_color=GRID_COLOR, line_width=1)
    fig = apply_dark_theme(fig, "Top Gainers & Losers (YTD)")
    fig.update_xaxes(title_text="YTD Return (%)", ticksuffix="%")
    fig.update_layout(showlegend=False, bargap=0.25)
    return fig


# ---------------------------------------------------------------------------
# 17. Dividend Calendar
# ---------------------------------------------------------------------------

def plot_dividend_calendar(dividend_df: pd.DataFrame) -> go.Figure:
    """
    Bar chart of expected monthly dividend income.
    dividend_df: DataFrame with columns [Month, Income (£)].
    """
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    if dividend_df.empty:
        fig = go.Figure()
        return apply_dark_theme(fig, "Dividend Calendar [SIMULATED]")

    df = dividend_df.copy()
    if "Month" in df.columns and df["Month"].dtype in (int, float):
        df["Month Label"] = df["Month"].apply(lambda m: month_labels[int(m)-1])
    else:
        df["Month Label"] = df.get("Month", range(1, len(df)+1))

    fig = go.Figure(go.Bar(
        x            = df["Month Label"].values,
        y            = df["Income (£)"].values / 1000,  # Show in £K
        marker_color  = ACCENT_COLOR,
        text         = [f"£{v/1000:.0f}K" for v in df["Income (£)"].values],
        textposition  = "outside",
        hovertemplate = "Month: %{x}<br>Income: £%{y:,.0f}K<extra></extra>",
    ))
    fig = apply_dark_theme(fig, "Expected Monthly Dividend Income [SIMULATED]")
    fig.update_yaxes(title_text="Income (£K)", tickprefix="£", ticksuffix="K")
    fig.update_xaxes(title_text="Month")
    fig.update_layout(showlegend=False, bargap=0.3)
    return fig


# ---------------------------------------------------------------------------
# 18. Income by Sector
# ---------------------------------------------------------------------------

def plot_income_by_sector(income_df: pd.DataFrame) -> go.Figure:
    """
    Donut chart of annual income by sector.
    income_df: DataFrame with columns [Sector, Annual Income (£)].
    """
    if income_df.empty:
        fig = go.Figure()
        return apply_dark_theme(fig, "Income by Sector [SIMULATED]")

    fig = go.Figure(go.Pie(
        labels        = income_df["Sector"].values,
        values        = income_df["Annual Income (£)"].values,
        hole          = 0.55,
        textinfo      = "label+percent",
        textfont_size = 10,
        hovertemplate = "<b>%{label}</b><br>£%{value:,.0f}<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(
        annotations=[{"text": "Income<br>by Sector", "x": 0.5, "y": 0.5,
                      "font_size": 12, "showarrow": False, "font_color": FONT_COLOR}]
    )
    return apply_dark_theme(fig, "Annual Income by Sector [SIMULATED]")


# ---------------------------------------------------------------------------
# 19. Dividend Growth
# ---------------------------------------------------------------------------

def plot_dividend_growth(dividend_history_df: pd.DataFrame) -> go.Figure:
    """
    Line chart with markers showing dividend income history and growth.
    dividend_history_df: DataFrame with columns [Year, Annual Income (£)].
    """
    if dividend_history_df.empty:
        fig = go.Figure()
        return apply_dark_theme(fig, "Dividend Growth [SIMULATED]")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x             = dividend_history_df["Year"].astype(str).values,
        y             = dividend_history_df["Annual Income (£)"].values / 1e6,
        mode          = "lines+markers",
        name          = "Annual Dividend Income",
        line          = {"color": POSITIVE_COLOR, "width": 2},
        marker        = {"size": 10, "color": POSITIVE_COLOR, "line": {"color": "white", "width": 1}},
        hovertemplate = "Year: %{x}<br>Income: £%{y:.2f}M<extra></extra>",
        fill          = "tozeroy",
        fillcolor     = "rgba(0, 200, 83, 0.1)",
    ))
    fig = apply_dark_theme(fig, "Dividend Income History [SIMULATED]")
    fig.update_yaxes(title_text="Annual Income (£M)", tickprefix="£", ticksuffix="M")
    fig.update_xaxes(title_text="Year")
    return fig


# ---------------------------------------------------------------------------
# Bonus: Scenario Impact per Holding
# ---------------------------------------------------------------------------

def plot_scenario_holding_impact(impact_df: pd.DataFrame) -> go.Figure:
    """
    Horizontal bar chart of scenario impact (£M) per holding.
    impact_df: DataFrame with columns [Ticker, Impact (£)].
    """
    if impact_df.empty:
        fig = go.Figure()
        return apply_dark_theme(fig, "Scenario Impact by Holding")

    df = impact_df.sort_values("Impact (£)", ascending=True)
    colors = [POSITIVE_COLOR if v >= 0 else NEGATIVE_COLOR for v in df["Impact (£)"].values]

    fig = go.Figure(go.Bar(
        x           = (df["Impact (£)"] / 1e6).values,
        y           = df["Ticker"].values,
        orientation = "h",
        marker_color = colors,
        text        = [f"£{v/1e6:.2f}M" for v in df["Impact (£)"].values],
        textposition = "outside",
        hovertemplate="<b>%{y}</b><br>Impact: £%{x:.2f}M<extra></extra>",
    ))
    fig.add_vline(x=0, line_dash="dash", line_color=GRID_COLOR, line_width=1)
    fig = apply_dark_theme(fig, "Scenario Impact by Holding (£M)")
    fig.update_xaxes(title_text="Impact (£M)", tickprefix="£", ticksuffix="M")
    fig.update_layout(showlegend=False, bargap=0.2)
    return fig


def plot_scenario_allocation_donut(post_shock_weights: Dict[str, float]) -> go.Figure:
    """Donut chart of post-scenario asset allocation."""
    return plot_asset_allocation_donut(post_shock_weights)
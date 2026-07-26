"""
app.py — Northgate Institutional Pension Fund Analytics Platform
Main Streamlit application. 6-tab dashboard with dark professional theme.

Run with: streamlit run pension_fund/app.py
"""

import sys
import os

# Ensure the parent directory is on the path so pension_fund modules are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, date
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Page config — MUST be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Northgate Institutional Pension Fund",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark professional theme
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
/* ---- Base ---- */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
}
.stApp {
    background-color: #0D1117;
    color: #E6EDF3;
}
section[data-testid="stSidebar"] {
    background-color: #161B22;
    border-right: 1px solid #30363D;
}
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stTextInput label {
    color: #E6EDF3 !important;
}

/* ---- Tabs ---- */
.stTabs [data-baseweb="tab-list"] {
    background-color: #161B22;
    border-bottom: 1px solid #30363D;
    gap: 2px;
    padding: 0 8px;
}
.stTabs [data-baseweb="tab"] {
    background-color: transparent;
    color: #8B949E;
    border-radius: 6px 6px 0 0;
    padding: 10px 20px;
    font-weight: 500;
    font-size: 14px;
    border: none;
    transition: all 0.2s;
}
.stTabs [data-baseweb="tab"]:hover {
    background-color: #21262D;
    color: #E6EDF3;
}
.stTabs [aria-selected="true"] {
    background-color: #0D1117 !important;
    color: #00D4FF !important;
    border-top: 2px solid #00D4FF;
    font-weight: 600;
}
.stTabs [data-baseweb="tab-panel"] {
    background-color: #0D1117;
    padding: 16px 0;
}

/* ---- KPI Cards ---- */
.kpi-card {
    background: linear-gradient(135deg, #161B22 0%, #1C2130 100%);
    border: 1px solid #30363D;
    border-radius: 12px;
    padding: 20px 16px;
    text-align: center;
    border-top: 3px solid #00D4FF;
    transition: border-top-color 0.2s;
    min-height: 120px;
}
.kpi-card:hover {
    border-top-color: #7B68EE;
}
.kpi-icon {
    font-size: 24px;
    margin-bottom: 6px;
}
.kpi-label {
    font-size: 11px;
    color: #8B949E;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 500;
    margin-bottom: 6px;
}
.kpi-value {
    font-size: 24px;
    font-weight: 700;
    color: #E6EDF3;
    line-height: 1.1;
}
.kpi-delta-pos {
    font-size: 12px;
    color: #00C853;
    font-weight: 500;
    margin-top: 4px;
}
.kpi-delta-neg {
    font-size: 12px;
    color: #FF4B4B;
    font-weight: 500;
    margin-top: 4px;
}
.kpi-delta-neutral {
    font-size: 12px;
    color: #8B949E;
    font-weight: 500;
    margin-top: 4px;
}

/* ---- Section headers ---- */
.section-header {
    font-size: 13px;
    font-weight: 600;
    color: #8B949E;
    text-transform: uppercase;
    letter-spacing: 1px;
    border-bottom: 1px solid #30363D;
    padding-bottom: 6px;
    margin: 16px 0 12px 0;
}

/* ---- Metrics ---- */
[data-testid="stMetric"] {
    background-color: #161B22;
    border: 1px solid #30363D;
    border-radius: 10px;
    padding: 16px;
}
[data-testid="stMetricLabel"] { color: #8B949E !important; font-size: 12px !important; }
[data-testid="stMetricValue"] { color: #E6EDF3 !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"] svg { display: none; }

/* ---- DataFrames ---- */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
.dataframe thead { background-color: #21262D !important; color: #E6EDF3 !important; }
.dataframe tbody { background-color: #0D1117 !important; color: #E6EDF3 !important; }

/* ---- Buttons ---- */
.stButton > button {
    background: linear-gradient(135deg, #00D4FF, #0099CC);
    color: #0D1117;
    border: none;
    border-radius: 8px;
    font-weight: 700;
    padding: 10px 24px;
    font-size: 14px;
    letter-spacing: 0.3px;
    transition: all 0.2s;
    width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #33DFFF, #00B8DD);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0, 212, 255, 0.3);
}

/* ---- Info/warning boxes ---- */
[data-testid="stInfo"] { background-color: rgba(0, 212, 255, 0.08) !important; border-color: #00D4FF !important; }
[data-testid="stWarning"] { background-color: rgba(255, 179, 0, 0.08) !important; border-color: #FFB300 !important; }

/* ---- Inputs ---- */
.stTextInput input, .stSelectbox select, .stMultiSelect div[data-baseweb="select"] {
    background-color: #21262D !important;
    border-color: #30363D !important;
    color: #E6EDF3 !important;
    border-radius: 8px;
}

/* ---- Sidebar fund logo ---- */
.fund-logo {
    text-align: center;
    padding: 20px 0 10px;
    border-bottom: 1px solid #30363D;
    margin-bottom: 16px;
}
.fund-name {
    font-size: 15px;
    font-weight: 700;
    color: #00D4FF;
    margin-top: 8px;
    line-height: 1.3;
}
.fund-aum {
    font-size: 12px;
    color: #8B949E;
    margin-top: 4px;
}
.simulated-badge {
    display: inline-block;
    background: rgba(0, 212, 255, 0.12);
    color: #00D4FF;
    border: 1px solid #00D4FF;
    border-radius: 4px;
    padding: 1px 6px;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.5px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Module imports (after sys.path insert)
# ---------------------------------------------------------------------------
from pension_fund.config import (
    PORTFOLIO_NAME, FUND_NAV, HOLDINGS, REAL_TICKERS, SIMULATED_TICKERS,
    BENCHMARK_TICKER, RISK_FREE_RATE, TRADING_DAYS_PER_YEAR,
    DIVIDEND_YIELDS, DIVIDEND_FREQUENCY, STRESS_SCENARIOS, ASSET_CLASS_COLORS,
)
from pension_fund.data_ingestion import (
    load_all_prices, get_benchmark_data, get_simulated_fundamentals, simulate_all_data,
)
from pension_fund.data_processing import (
    calculate_daily_returns, calculate_portfolio_returns, calculate_cumulative_returns,
    calculate_annualised_return, calculate_rolling_volatility, calculate_rolling_return,
    calculate_monthly_returns, calculate_portfolio_weights, calculate_portfolio_value_series,
    calculate_sector_exposure, calculate_asset_class_exposure, calculate_geographic_exposure,
    calculate_performance_attribution, build_holdings_df,
)
from pension_fund.portfolio_analytics import (
    calculate_all_metrics, calculate_ytd_return, calculate_mtd_return,
    calculate_drawdown_series, calculate_holding_returns,
)
from pension_fund.risk_analytics import (
    calculate_var, calculate_cvar, calculate_var_dollar, calculate_cvar_dollar,
    calculate_correlation_matrix, calculate_portfolio_volatility,
    calculate_risk_contribution, stress_test_portfolio, apply_custom_scenario,
    calculate_return_distribution_stats,
)
from pension_fund.visualizations import (
    plot_portfolio_value_trend, plot_asset_allocation_donut, plot_sector_allocation_bar,
    plot_geographic_allocation, plot_cumulative_returns, plot_rolling_returns,
    plot_monthly_return_heatmap, plot_drawdown, plot_correlation_matrix,
    plot_return_distribution, plot_risk_contribution, plot_holdings_scatter,
    plot_rolling_volatility, plot_performance_attribution, plot_stress_test_results,
    plot_top_gainers_losers, plot_dividend_calendar, plot_income_by_sector,
    plot_dividend_growth, plot_scenario_holding_impact, plot_scenario_allocation_donut,
)


# ---------------------------------------------------------------------------
# Helper: currency formatting
# ---------------------------------------------------------------------------
def fmt_gbp(value: float, suffix: str = "") -> str:
    """Format a £ value: £1.23M, £456K, etc."""
    abs_val = abs(value)
    sign    = "-" if value < 0 else ""
    if abs_val >= 1e9:
        return f"{sign}£{abs_val/1e9:.2f}B{suffix}"
    elif abs_val >= 1e6:
        return f"{sign}£{abs_val/1e6:.2f}M{suffix}"
    elif abs_val >= 1e3:
        return f"{sign}£{abs_val/1e3:.1f}K{suffix}"
    else:
        return f"{sign}£{abs_val:.0f}{suffix}"


def fmt_pct(value: float, decimals: int = 2) -> str:
    """Format a decimal return as percentage string."""
    return f"{value*100:+.{decimals}f}%"


def kpi_card(icon: str, label: str, value: str, delta: str = "", delta_pos: bool = True) -> str:
    """Return HTML for a KPI card."""
    delta_class = "kpi-delta-pos" if delta_pos else "kpi-delta-neg"
    if not delta:
        delta_class = "kpi-delta-neutral"
    return f"""
    <div class="kpi-card">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="{delta_class}">{delta}</div>
    </div>
    """


# ---------------------------------------------------------------------------
# Data loading and computation
# ---------------------------------------------------------------------------
ALL_TICKERS = [h["ticker"] for h in HOLDINGS]


@st.cache_data(ttl=3600, show_spinner=False)
def load_data(api_key: str = "") -> dict:
    """
    Load and pre-compute all data needed for the dashboard.
    Returns a dict with all pre-computed metrics in session state.
    Cache for 1 hour — respects Alpha Vantage rate limits via data_ingestion.
    """
    # --- Price data ---
    prices_raw = load_all_prices(api_key if api_key else None, ALL_TICKERS)
    benchmark_raw = get_benchmark_data(api_key if api_key else None)

    # Build a unified adjusted_close DataFrame
    prices_dict = {}
    for ticker, df in prices_raw.items():
        if df is not None and not df.empty and "adjusted_close" in df.columns:
            prices_dict[ticker] = df["adjusted_close"]
        elif df is not None and not df.empty:
            prices_dict[ticker] = df.iloc[:, 0]

    prices_df = pd.DataFrame(prices_dict).sort_index()

    # Benchmark
    if benchmark_raw is not None and "adjusted_close" in benchmark_raw.columns:
        benchmark_prices = benchmark_raw["adjusted_close"]
    elif benchmark_raw is not None:
        benchmark_prices = benchmark_raw.iloc[:, 0]
    else:
        benchmark_prices = prices_df.get("SPY", pd.Series(dtype=float))

    # Align to common dates (inner join on index)
    prices_df = prices_df.dropna(how="all")

    # --- Returns ---
    returns_df        = calculate_daily_returns(prices_df)
    weights           = calculate_portfolio_weights(HOLDINGS, prices_df)
    portfolio_returns = calculate_portfolio_returns(returns_df, weights)
    benchmark_returns = benchmark_prices.pct_change().dropna()

    # Align portfolio and benchmark
    combined_idx      = portfolio_returns.index.intersection(benchmark_returns.index)
    portfolio_returns = portfolio_returns.loc[combined_idx]
    benchmark_returns = benchmark_returns.loc[combined_idx]

    # --- Cumulative returns ---
    port_cum  = calculate_cumulative_returns(portfolio_returns)
    bench_cum = calculate_cumulative_returns(benchmark_returns)

    # --- All metrics ---
    metrics = calculate_all_metrics(portfolio_returns, benchmark_returns, RISK_FREE_RATE)

    # --- Portfolio value series ---
    portfolio_value_series = calculate_portfolio_value_series(HOLDINGS, prices_df)

    # --- Exposures ---
    asset_class_weights = calculate_asset_class_exposure(HOLDINGS, prices_df)
    sector_weights      = calculate_sector_exposure(HOLDINGS, prices_df)
    geo_weights         = calculate_geographic_exposure(HOLDINGS, prices_df)

    # --- Rolling analytics ---
    rolling_vol     = calculate_rolling_volatility(portfolio_returns, 21)
    rolling_ret_21d = calculate_rolling_return(portfolio_returns, 21)
    rolling_ret_63d = calculate_rolling_return(portfolio_returns, 63)

    # --- Drawdown series ---
    drawdown_series = calculate_drawdown_series(portfolio_returns)

    # --- Monthly returns heatmap ---
    monthly_pivot = calculate_monthly_returns(portfolio_returns)

    # --- Fundamentals (simulated) ---
    fundamentals = {h["ticker"]: get_simulated_fundamentals(h["ticker"]) for h in HOLDINGS}

    # --- Holdings table ---
    holdings_df = build_holdings_df(HOLDINGS, prices_df, fundamentals)

    # --- Risk metrics ---
    port_vol    = calculate_portfolio_volatility(returns_df, weights)
    risk_contrib = calculate_risk_contribution(returns_df, weights)
    var_95      = calculate_var_dollar(portfolio_returns, portfolio_value_series.iloc[-1], 0.95)
    cvar_95     = calculate_cvar_dollar(portfolio_returns, portfolio_value_series.iloc[-1], 0.95)
    corr_matrix = calculate_correlation_matrix(returns_df[[t for t in REAL_TICKERS if t in returns_df.columns]])
    dist_stats  = calculate_return_distribution_stats(portfolio_returns)

    # --- Attribution ---
    attribution_df = calculate_performance_attribution(HOLDINGS, prices_df, benchmark_returns)

    # --- Holdings returns for gainers/losers ---
    holding_returns_df = calculate_holding_returns(HOLDINGS, prices_df, "ytd")

    # --- Performance attribution ---
    perf_attr_df = calculate_performance_attribution(HOLDINGS, prices_df, benchmark_returns)

    # --- Dividend simulation ---
    dividend_df      = _simulate_dividends(holdings_df, portfolio_value_series.iloc[-1])
    dividend_cal_df  = _dividend_calendar(dividend_df)
    income_sector_df = _income_by_sector(dividend_df)
    dividend_hist_df = _dividend_history(dividend_df)

    # --- Daily P&L ---
    latest_value    = float(portfolio_value_series.iloc[-1])
    prev_value      = float(portfolio_value_series.iloc[-2]) if len(portfolio_value_series) > 1 else latest_value
    daily_pnl       = latest_value - prev_value
    daily_return    = portfolio_returns.iloc[-1] if len(portfolio_returns) > 0 else 0.0

    # --- Cost basis total ---
    total_cost     = sum(
        (h["shares"] * h["cost_basis"] if h["ticker"] not in ("UK_GILT_10Y", "US_TREAS_10Y", "CASH_GBP")
         else h["cost_basis"])
        for h in HOLDINGS
    )
    unrealised_pnl = latest_value - total_cost

    return {
        # Prices
        "prices_df":             prices_df,
        "benchmark_prices":      benchmark_prices,
        "returns_df":            returns_df,
        # Returns
        "portfolio_returns":     portfolio_returns,
        "benchmark_returns":     benchmark_returns,
        "port_cum":              port_cum,
        "bench_cum":             bench_cum,
        # Portfolio
        "weights":               weights,
        "portfolio_value_series": portfolio_value_series,
        "latest_value":          latest_value,
        "daily_pnl":             daily_pnl,
        "daily_return":          float(daily_return),
        "unrealised_pnl":        unrealised_pnl,
        # Exposures
        "asset_class_weights":   asset_class_weights,
        "sector_weights":        sector_weights,
        "geo_weights":           geo_weights,
        # Rolling
        "rolling_vol":           rolling_vol,
        "rolling_ret_21d":       rolling_ret_21d,
        "rolling_ret_63d":       rolling_ret_63d,
        # Metrics
        "metrics":               metrics,
        "dist_stats":            dist_stats,
        # Risk
        "port_vol":              port_vol,
        "risk_contrib":          risk_contrib,
        "var_95":                var_95,
        "cvar_95":               cvar_95,
        "corr_matrix":           corr_matrix,
        "drawdown_series":       drawdown_series,
        # Misc
        "monthly_pivot":         monthly_pivot,
        "holdings_df":           holdings_df,
        "fundamentals":          fundamentals,
        "attribution_df":        attribution_df,
        "holding_returns_df":    holding_returns_df,
        "perf_attr_df":          perf_attr_df,
        # Dividends
        "dividend_df":           dividend_df,
        "dividend_cal_df":       dividend_cal_df,
        "income_sector_df":      income_sector_df,
        "dividend_hist_df":      dividend_hist_df,
        # P&L
        "total_cost":            total_cost,
        "unrealised_pnl":        unrealised_pnl,
        "load_timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "is_simulated":          not bool(api_key),
    }


def _simulate_dividends(holdings_df: pd.DataFrame, portfolio_value: float) -> pd.DataFrame:
    """
    Generate realistic simulated dividend data for each holding.
    Returns DataFrame with columns: Ticker, Name, Annual Dividend/Share,
    Yield, Annual Income (£), Frequency, Next Payment Date, Sector.
    """
    records = []
    today = datetime.today()
    for h in HOLDINGS:
        ticker    = h["ticker"]
        name      = h["name"]
        div_yield = DIVIDEND_YIELDS.get(ticker, 0.0)
        freq      = DIVIDEND_FREQUENCY.get(ticker, "None")

        # Get current price from holdings_df
        row = holdings_df[holdings_df["Ticker"] == ticker]
        if row.empty:
            continue
        current_price = float(row["Current Price"].iloc[0])
        market_value  = float(row["Market Value (£)"].iloc[0])
        shares        = h["shares"]

        annual_div_per_share = current_price * div_yield
        annual_income        = market_value * div_yield

        # Simulate next payment date based on frequency
        if freq == "Monthly":
            next_date = date(today.year, today.month % 12 + 1, 15)
        elif freq == "Quarterly":
            quarter_months = [3, 6, 9, 12]
            next_month = next((m for m in quarter_months if m > today.month), 3)
            next_date  = date(today.year + (1 if next_month < today.month else 0), next_month, 15)
        elif freq == "Semi-Annual":
            semi_months = [6, 12]
            next_month  = next((m for m in semi_months if m > today.month), 6)
            next_date   = date(today.year + (1 if next_month < today.month else 0), next_month, 1)
        else:
            next_date = None

        records.append({
            "Ticker":                ticker,
            "Name":                  name,
            "Sector":                h["sector"],
            "Asset Class":           h["asset_class"],
            "Annual Div/Share":      annual_div_per_share,
            "Yield (%)":             div_yield * 100,
            "Annual Income (£)":     annual_income,
            "Frequency":             freq,
            "Next Payment":          str(next_date) if next_date else "N/A",
        })

    return pd.DataFrame(records)


def _dividend_calendar(dividend_df: pd.DataFrame) -> pd.DataFrame:
    """Generate monthly dividend income schedule."""
    monthly_income = {m: 0.0 for m in range(1, 13)}
    for _, row in dividend_df.iterrows():
        freq   = row["Frequency"]
        income = row["Annual Income (£)"]
        if freq == "Monthly":
            for m in range(1, 13):
                monthly_income[m] += income / 12
        elif freq == "Quarterly":
            for m in [3, 6, 9, 12]:
                monthly_income[m] += income / 4
        elif freq == "Semi-Annual":
            for m in [6, 12]:
                monthly_income[m] += income / 2
        elif freq == "Annual":
            monthly_income[12] += income
    return pd.DataFrame({"Month": list(monthly_income.keys()), "Income (£)": list(monthly_income.values())})


def _income_by_sector(dividend_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate dividend income by sector."""
    grp = dividend_df.groupby("Sector")["Annual Income (£)"].sum().reset_index()
    return grp[grp["Annual Income (£)"] > 0]


def _dividend_history(dividend_df: pd.DataFrame) -> pd.DataFrame:
    """Generate simulated 5-year dividend income history (growing at ~5% p.a.)."""
    total_income = dividend_df["Annual Income (£)"].sum()
    years  = list(range(datetime.today().year - 4, datetime.today().year + 1))
    incomes = [total_income * (1 / (1.05 ** (4 - i))) for i in range(5)]
    return pd.DataFrame({"Year": years, "Annual Income (£)": incomes})


# ===========================================================================
# SIDEBAR
# ===========================================================================
with st.sidebar:
    st.markdown("""
    <div class="fund-logo">
        <div style="font-size: 52px;">🏛️</div>
        <div class="fund-name">Northgate<br>Institutional Pension Fund</div>
        <div class="fund-aum">AUM Target: £450M</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🔑 Data Source")
    api_key = st.text_input(
        "Alpha Vantage API Key",
        type="password",
        help="Optional. Get a free key at alphavantage.co. Without a key, high-quality simulated data is used.",
        placeholder="Enter API key (optional)",
    )

    load_btn = st.button("⚡ Load / Refresh Data", use_container_width=True)

    st.markdown("---")
    st.markdown("### 📅 Date Range")
    today        = date.today()
    default_start = date(today.year - 2, today.month, today.day)
    date_start   = st.date_input("From", value=default_start, max_value=today)
    date_end     = st.date_input("To",   value=today,          min_value=date_start, max_value=today)

    st.markdown("---")
    st.markdown("""
    <div style="font-size: 11px; color: #8B949E; line-height: 1.6;">
        📌 Data labelled <span class="simulated-badge">SIMULATED</span> is generated for
        illustration using geometric Brownian motion. Real market data is sourced from
        Alpha Vantage (API key required).
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-top: 16px; font-size: 11px; color: #8B949E;">
        🗓️ Today: <b style="color: #E6EDF3;">{today.strftime("%d %b %Y")}</b>
    </div>
    """, unsafe_allow_html=True)


# ===========================================================================
# Data loading with progress bar
# ===========================================================================
if "data" not in st.session_state or load_btn:
    with st.spinner("⏳ Loading portfolio data…"):
        progress_bar = st.progress(0)
        progress_bar.progress(10)
        data = load_data(api_key)
        progress_bar.progress(100)
        st.session_state["data"] = data
        progress_bar.empty()
    if data["is_simulated"]:
        st.info("🔵 **SIMULATED DATA** — All price data is generated using realistic Monte Carlo simulation. "
                "Enter an Alpha Vantage API key in the sidebar to load real market data.")
    else:
        st.success(f"✅ Real market data loaded — Last updated: {data['load_timestamp']}")

data = st.session_state["data"]

# Show last updated in sidebar
with st.sidebar:
    st.markdown(f"""
    <div style="font-size: 11px; color: #8B949E; margin-top: 4px;">
        🔄 Last loaded: <b style="color: #E6EDF3;">{data.get('load_timestamp', 'N/A')}</b>
    </div>
    """, unsafe_allow_html=True)

# Apply date filter to data slices
date_start_ts = pd.Timestamp(date_start)
date_end_ts   = pd.Timestamp(date_end)


def slice_series(s: pd.Series) -> pd.Series:
    """Filter a DatetimeIndex series to the selected date range."""
    if s is None or s.empty:
        return s
    return s.loc[(s.index >= date_start_ts) & (s.index <= date_end_ts)]


def slice_df(df: pd.DataFrame) -> pd.DataFrame:
    """Filter a DataFrame with DatetimeIndex to the selected date range."""
    if df is None or df.empty:
        return df
    return df.loc[(df.index >= date_start_ts) & (df.index <= date_end_ts)]


# Slice filtered versions
port_returns_f = slice_series(data["portfolio_returns"])
bench_returns_f = slice_series(data["benchmark_returns"])
port_cum_f   = calculate_cumulative_returns(port_returns_f)
bench_cum_f  = calculate_cumulative_returns(bench_returns_f)
pv_series_f  = slice_series(data["portfolio_value_series"])
rolling_vol_f = slice_series(data["rolling_vol"])
rolling_ret_21_f = slice_series(data["rolling_ret_21d"])
rolling_ret_63_f = slice_series(data["rolling_ret_63d"])
drawdown_f   = slice_series(data["drawdown_series"])
returns_df_f = slice_df(data["returns_df"])

# Re-compute metrics over filtered range (uses pre-computed if date covers full range)
metrics = calculate_all_metrics(port_returns_f, bench_returns_f, RISK_FREE_RATE)

latest_val = float(pv_series_f.iloc[-1]) if not pv_series_f.empty else data["latest_value"]
prev_val   = float(pv_series_f.iloc[-2]) if len(pv_series_f) > 1 else latest_val
daily_ret  = float(port_returns_f.iloc[-1]) if not port_returns_f.empty else 0.0
daily_pnl  = latest_val - prev_val


# ===========================================================================
# TABS
# ===========================================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏠 Executive Summary",
    "📈 Performance",
    "⚠️ Risk",
    "📋 Holdings",
    "💰 Income",
    "🧪 Stress Testing",
])


# ---------------------------------------------------------------------------
# TAB 1: Executive Summary
# ---------------------------------------------------------------------------
with tab1:
    # Row 1 — KPI Cards
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(kpi_card(
            "💼", "Portfolio Value", fmt_gbp(latest_val),
            f"{fmt_pct(daily_ret)} today", daily_ret >= 0,
        ), unsafe_allow_html=True)

    with c2:
        st.markdown(kpi_card(
            "📅", "Daily Return", f"{daily_ret*100:+.2f}%",
            fmt_gbp(daily_pnl) + " P&L", daily_ret >= 0,
        ), unsafe_allow_html=True)

    with c3:
        mtd = metrics.get("mtd_return", 0.0)
        st.markdown(kpi_card(
            "🗓️", "MTD Return", f"{mtd*100:+.2f}%",
            "Month to date", mtd >= 0,
        ), unsafe_allow_html=True)

    with c4:
        ytd = metrics.get("ytd_return", 0.0)
        st.markdown(kpi_card(
            "📆", "YTD Return", f"{ytd*100:+.2f}%",
            "Year to date", ytd >= 0,
        ), unsafe_allow_html=True)

    with c5:
        unrealised = data["unrealised_pnl"]
        st.markdown(kpi_card(
            "💹", "Unrealised P&L", fmt_gbp(unrealised),
            f"{unrealised/data['total_cost']*100:+.1f}% vs cost", unrealised >= 0,
        ), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2 — Portfolio Value Trend + Asset Allocation
    col_chart, col_donut = st.columns([2, 1])
    with col_chart:
        fig_pv = plot_portfolio_value_trend(pv_series_f)
        st.plotly_chart(fig_pv, use_container_width=True)
    with col_donut:
        fig_alloc = plot_asset_allocation_donut(data["asset_class_weights"])
        st.plotly_chart(fig_alloc, use_container_width=True)

    # Row 3 — Gainers/Losers + Stats Table
    col_gl, col_stats = st.columns(2)
    with col_gl:
        if not data["holding_returns_df"].empty:
            fig_gl = plot_top_gainers_losers(data["holding_returns_df"], top_n=5)
            st.plotly_chart(fig_gl, use_container_width=True)
        else:
            st.info("No holding return data available for selected period.")

    with col_stats:
        st.markdown('<p class="section-header">Portfolio Summary Statistics</p>', unsafe_allow_html=True)
        n_holdings = len(HOLDINGS)
        summary_data = {
            "Metric": [
                "Number of Holdings",
                "Annualised Return",
                "Annualised Volatility",
                "Sharpe Ratio",
                "Sortino Ratio",
                "Max Drawdown",
                "Beta vs S&P 500",
                "Alpha (Jensen's)",
                "Information Ratio",
                "Tracking Error",
            ],
            "Value": [
                str(n_holdings),
                f"{metrics.get('annualised_return', 0)*100:.2f}%",
                f"{metrics.get('annualised_volatility', 0)*100:.2f}%",
                f"{metrics.get('sharpe_ratio', 0):.3f}",
                f"{metrics.get('sortino_ratio', 0):.3f}",
                f"{metrics.get('max_drawdown', 0)*100:.2f}%",
                f"{metrics.get('beta', 1):.3f}",
                f"{metrics.get('alpha', 0)*100:.2f}%",
                f"{metrics.get('information_ratio', 0):.3f}",
                f"{metrics.get('tracking_error', 0)*100:.2f}%",
            ],
        }
        st.dataframe(
            pd.DataFrame(summary_data),
            use_container_width=True,
            hide_index=True,
        )

    if data["is_simulated"]:
        st.info("🔵 **SIMULATED** — All price data shown above is generated using Geometric Brownian Motion "
                "with realistic sector parameters. Provide an Alpha Vantage API key for real market data.")


# ---------------------------------------------------------------------------
# TAB 2: Performance Analytics
# ---------------------------------------------------------------------------
with tab2:
    # Row 1 — Performance KPIs
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        ann_ret = metrics.get("annualised_return", 0.0)
        st.metric("Annualised Return", f"{ann_ret*100:.2f}%", delta=f"{(ann_ret - 0.10)*100:+.2f}% vs 10% target")
    with p2:
        sharpe = metrics.get("sharpe_ratio", 0.0)
        st.metric("Sharpe Ratio", f"{sharpe:.3f}", delta="vs 1.0 benchmark" if sharpe >= 1.0 else "Below 1.0")
    with p3:
        sortino = metrics.get("sortino_ratio", 0.0)
        st.metric("Sortino Ratio", f"{sortino:.3f}")
    with p4:
        max_dd = metrics.get("max_drawdown", 0.0)
        st.metric("Max Drawdown", f"{max_dd*100:.2f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2 — Cumulative returns
    st.markdown('<p class="section-header">Cumulative Returns vs S&P 500 Benchmark</p>', unsafe_allow_html=True)
    if not port_cum_f.empty and not bench_cum_f.empty:
        fig_cum = plot_cumulative_returns(port_cum_f, bench_cum_f)
        st.plotly_chart(fig_cum, use_container_width=True)
    else:
        st.warning("Insufficient data for cumulative returns chart.")

    # Row 3 — Rolling returns + Monthly heatmap
    col_roll, col_heat = st.columns(2)
    with col_roll:
        if not rolling_ret_63_f.empty:
            fig_roll = plot_rolling_returns(rolling_ret_63_f, "63-Day (3M)")
            st.plotly_chart(fig_roll, use_container_width=True)
    with col_heat:
        if not data["monthly_pivot"].empty:
            fig_heat = plot_monthly_return_heatmap(data["monthly_pivot"])
            st.plotly_chart(fig_heat, use_container_width=True)

    # Row 4 — Rolling vol + Attribution
    col_rvol, col_attr = st.columns(2)
    with col_rvol:
        if not rolling_vol_f.empty:
            fig_rvol = plot_rolling_volatility(rolling_vol_f)
            st.plotly_chart(fig_rvol, use_container_width=True)
    with col_attr:
        if not data["perf_attr_df"].empty:
            fig_attr = plot_performance_attribution(data["perf_attr_df"])
            st.plotly_chart(fig_attr, use_container_width=True)


# ---------------------------------------------------------------------------
# TAB 3: Risk Analytics
# ---------------------------------------------------------------------------
with tab3:
    # Row 1 — Risk KPIs
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.metric("Portfolio Volatility (Ann.)", f"{metrics.get('annualised_volatility', 0)*100:.2f}%",
                  help="Annualised standard deviation of daily returns.")
    with r2:
        st.metric("Beta vs S&P 500", f"{metrics.get('beta', 1):.3f}",
                  help="Portfolio sensitivity to S&P 500 movements.")
    with r3:
        var_95 = calculate_var_dollar(port_returns_f, latest_val, 0.95)
        st.metric("VaR 95% (1-Day)", fmt_gbp(var_95),
                  help="Historical VaR: maximum expected loss at 95% confidence over 1 trading day.")
    with r4:
        cvar_95 = calculate_cvar_dollar(port_returns_f, latest_val, 0.95)
        st.metric("CVaR / ES 95%", fmt_gbp(cvar_95),
                  help="Expected Shortfall: average loss beyond the VaR threshold.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2 — Correlation + Distribution
    col_corr, col_dist = st.columns(2)
    with col_corr:
        if data["corr_matrix"] is not None and not data["corr_matrix"].empty:
            fig_corr = plot_correlation_matrix(data["corr_matrix"])
            st.plotly_chart(fig_corr, use_container_width=True)
    with col_dist:
        if not port_returns_f.empty:
            fig_dist = plot_return_distribution(port_returns_f)
            st.plotly_chart(fig_dist, use_container_width=True)

    # Row 3 — Drawdown + Risk Contribution
    col_dd, col_rc = st.columns(2)
    with col_dd:
        if not drawdown_f.empty:
            fig_dd = plot_drawdown(drawdown_f)
            st.plotly_chart(fig_dd, use_container_width=True)
    with col_rc:
        if data["risk_contrib"] is not None and not data["risk_contrib"].empty:
            fig_rc = plot_risk_contribution(data["risk_contrib"])
            st.plotly_chart(fig_rc, use_container_width=True)

    # Row 4 — Rolling volatility full-width
    st.markdown('<p class="section-header">Rolling Volatility</p>', unsafe_allow_html=True)
    if not rolling_vol_f.empty:
        fig_rvol_full = plot_rolling_volatility(rolling_vol_f)
        st.plotly_chart(fig_rvol_full, use_container_width=True)

    # Distribution stats table
    st.markdown('<p class="section-header">Return Distribution Statistics</p>', unsafe_allow_html=True)
    ds = data["dist_stats"]
    if ds:
        dist_table = pd.DataFrame({
            "Statistic": ["Annualised Return", "Annualised Volatility", "Skewness", "Excess Kurtosis",
                          "VaR 95% (1-Day)", "CVaR 95% (1-Day)", "VaR 99% (1-Day)", "CVaR 99% (1-Day)",
                          "Best Day", "Worst Day", "% Positive Days", "Observations"],
            "Value": [
                f"{ds.get('mean', 0):.2%}",
                f"{ds.get('std', 0):.2%}",
                f"{ds.get('skewness', 0):.4f}",
                f"{ds.get('kurtosis', 0):.4f}",
                f"{ds.get('var_95', 0):.2%}",
                f"{ds.get('cvar_95', 0):.2%}",
                f"{ds.get('var_99', 0):.2%}",
                f"{ds.get('cvar_99', 0):.2%}",
                f"{ds.get('max_return', 0):.2%}",
                f"{ds.get('min_return', 0):.2%}",
                f"{ds.get('pct_positive', 0):.1f}%",
                f"{ds.get('n_observations', 0):,}",
            ],
        })
        st.dataframe(dist_table, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# TAB 4: Holdings & Portfolio
# ---------------------------------------------------------------------------
with tab4:
    # Row 1 — Sector + Geographic allocation
    col_sec, col_geo = st.columns(2)
    with col_sec:
        fig_sec = plot_sector_allocation_bar(data["sector_weights"])
        st.plotly_chart(fig_sec, use_container_width=True)
    with col_geo:
        fig_geo = plot_geographic_allocation(data["geo_weights"])
        st.plotly_chart(fig_geo, use_container_width=True)

    # Row 2 — Risk vs Return bubble chart
    st.markdown('<p class="section-header">Risk vs Return Bubble Chart</p>', unsafe_allow_html=True)
    if not returns_df_f.empty and not data["holdings_df"].empty:
        fig_scatter = plot_holdings_scatter(data["holdings_df"], returns_df_f, data["weights"])
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Row 3 — Holdings Table with filters
    st.markdown('<p class="section-header">Holdings Detail</p>', unsafe_allow_html=True)

    search_col, ac_col = st.columns([2, 2])
    with search_col:
        search_term = st.text_input("🔍 Search holdings (ticker or name)", placeholder="e.g. AAPL, Apple…")
    with ac_col:
        asset_classes = data["holdings_df"]["Asset Class"].unique().tolist() if not data["holdings_df"].empty else []
        ac_filter = st.multiselect("Filter by Asset Class", options=asset_classes, default=asset_classes)

    holdings_display = data["holdings_df"].copy()

    if search_term:
        mask = (
            holdings_display["Ticker"].str.contains(search_term, case=False, na=False) |
            holdings_display["Name"].str.contains(search_term, case=False, na=False)
        )
        holdings_display = holdings_display[mask]

    if ac_filter:
        holdings_display = holdings_display[holdings_display["Asset Class"].isin(ac_filter)]

    # Format numeric columns for display
    def format_holdings_df(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "Market Value (£)" in df.columns:
            df["Market Value (£)"] = df["Market Value (£)"].apply(fmt_gbp)
        if "Unrealised P&L (£)" in df.columns:
            df["Unrealised P&L (£)"] = df["Unrealised P&L (£)"].apply(fmt_gbp)
        if "Weight (%)" in df.columns:
            df["Weight (%)"] = df["Weight (%)"].apply(lambda x: f"{x:.2f}%")
        if "Unrealised P&L (%)" in df.columns:
            df["Unrealised P&L (%)"] = df["Unrealised P&L (%)"].apply(lambda x: f"{x:+.2f}%")
        if "Current Price" in df.columns:
            df["Current Price"] = df["Current Price"].apply(lambda x: f"${x:,.2f}")
        if "P/E Ratio" in df.columns:
            df["P/E Ratio"] = df["P/E Ratio"].apply(lambda x: f"{x:.1f}" if x else "—")
        if "Div Yield (%)" in df.columns:
            df["Div Yield (%)"] = df["Div Yield (%)"].apply(lambda x: f"{x:.2f}%" if x else "—")
        if "Beta" in df.columns:
            df["Beta"] = df["Beta"].apply(lambda x: f"{x:.2f}" if x else "—")
        if "Market Cap (£B)" in df.columns:
            df["Market Cap (£B)"] = df["Market Cap (£B)"].apply(lambda x: f"£{x:.1f}B" if x else "—")
        if "EPS" in df.columns:
            df["EPS"] = df["EPS"].apply(lambda x: f"${x:.2f}" if x else "—")
        if "52W High" in df.columns:
            df["52W High"] = df["52W High"].apply(lambda x: f"${x:,.2f}" if x else "—")
        if "52W Low" in df.columns:
            df["52W Low"] = df["52W Low"].apply(lambda x: f"${x:,.2f}" if x else "—")
        if "Cost Basis" in df.columns:
            df["Cost Basis"] = df["Cost Basis"].apply(lambda x: f"${x:,.2f}")
        return df

    formatted = format_holdings_df(holdings_display)
    st.dataframe(formatted, use_container_width=True, hide_index=True, height=450)

    if data["is_simulated"]:
        st.info("🔵 **SIMULATED** — P/E, Beta, EPS and other fundamental metrics are simulated estimates. "
                "Prices and market values are generated via GBM simulation.")


# ---------------------------------------------------------------------------
# TAB 5: Income & Dividends
# ---------------------------------------------------------------------------
with tab5:
    st.info("💡 **Note:** All income and dividend data shown in this tab is **[SIMULATED]** based on "
            "approximate historical yields. It is for illustrative purposes only and does not represent "
            "actual dividend receipts.")

    # Row 1 — Income KPIs
    div_df = data["dividend_df"]
    total_income  = div_df["Annual Income (£)"].sum() if not div_df.empty else 0
    avg_yield     = div_df["Yield (%)"].mean() if not div_df.empty else 0
    n_payers      = (div_df["Annual Income (£)"] > 0).sum() if not div_df.empty else 0

    i1, i2, i3 = st.columns(3)
    with i1:
        st.markdown(kpi_card("💰", "Expected Annual Income [SIM]", fmt_gbp(total_income), "Gross dividend income"), unsafe_allow_html=True)
    with i2:
        st.markdown(kpi_card("📊", "Average Dividend Yield [SIM]", f"{avg_yield:.2f}%", "Portfolio average"), unsafe_allow_html=True)
    with i3:
        st.markdown(kpi_card("🏢", "Dividend Payers [SIM]", str(n_payers), f"of {len(HOLDINGS)} holdings"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2 — Calendar + Income by Sector
    col_cal, col_sec_inc = st.columns(2)
    with col_cal:
        if not data["dividend_cal_df"].empty:
            fig_cal = plot_dividend_calendar(data["dividend_cal_df"])
            st.plotly_chart(fig_cal, use_container_width=True)
    with col_sec_inc:
        if not data["income_sector_df"].empty:
            fig_inc = plot_income_by_sector(data["income_sector_df"])
            st.plotly_chart(fig_inc, use_container_width=True)

    # Row 3 — Dividend growth
    st.markdown('<p class="section-header">Dividend Income History [SIMULATED]</p>', unsafe_allow_html=True)
    if not data["dividend_hist_df"].empty:
        fig_div_hist = plot_dividend_growth(data["dividend_hist_df"])
        st.plotly_chart(fig_div_hist, use_container_width=True)

    # Row 4 — Dividend detail table
    st.markdown('<p class="section-header">Dividend Detail by Holding [SIMULATED]</p>', unsafe_allow_html=True)
    if not div_df.empty:
        div_display = div_df[
            ["Ticker", "Name", "Annual Div/Share", "Yield (%)", "Annual Income (£)", "Frequency", "Next Payment"]
        ].copy()
        div_display["Annual Div/Share"]  = div_display["Annual Div/Share"].apply(lambda x: f"${x:.4f}")
        div_display["Yield (%)"]         = div_display["Yield (%)"].apply(lambda x: f"{x:.2f}%")
        div_display["Annual Income (£)"] = div_display["Annual Income (£)"].apply(fmt_gbp)
        st.dataframe(div_display, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# TAB 6: Scenario & Stress Testing
# ---------------------------------------------------------------------------
with tab6:
    st.markdown("### 🧪 Scenario & Stress Testing")
    st.warning("⚠️ **Note:** Stress testing uses simplified linear shocks applied to current market values. "
               "Results are for illustrative purposes only. **[SIMULATED]**")

    # Scenario selection
    scenario_names = list(STRESS_SCENARIOS.keys()) + ["Custom Scenario"]
    selected_scenario = st.selectbox(
        "Select a stress scenario",
        options=scenario_names,
        index=0,
    )

    # Custom scenario sliders
    if selected_scenario == "Custom Scenario":
        st.markdown('<p class="section-header">Custom Scenario Shocks</p>', unsafe_allow_html=True)
        sc1, sc2 = st.columns(2)
        with sc1:
            eq_shock  = st.slider("Equities (%)",      -50, 50, 0, 1) / 100
            bond_shock = st.slider("Bonds (%)",          -30, 30, 0, 1) / 100
            etf_shock  = st.slider("ETFs (%)",           -50, 50, 0, 1) / 100
            cash_shock = st.slider("Cash (%)",           -10, 10, 0, 1) / 100
        with sc2:
            tech_shock = st.slider("Technology (additional %)", -30, 30, 0, 1) / 100
            fin_shock  = st.slider("Financials (additional %)", -30, 30, 0, 1) / 100

        scenario_result = apply_custom_scenario(
            HOLDINGS, data["prices_df"],
            eq_shock, bond_shock, etf_shock, cash_shock,
            tech_shock, fin_shock,
            portfolio_value=latest_val,
        )
    else:
        scenario_def    = STRESS_SCENARIOS[selected_scenario]
        scenario_result = stress_test_portfolio(
            HOLDINGS, data["prices_df"], scenario_def, latest_val
        )
        st.markdown(f"**{scenario_def['description']}**")

    # Row 2 — Impact KPIs
    new_val    = scenario_result["new_portfolio_value"]
    impact_gbp = scenario_result["total_impact_gbp"]
    impact_pct = scenario_result["total_impact_pct"]

    # Worst hit holding
    hi_df   = pd.DataFrame(scenario_result["holdings_impact"])
    worst   = hi_df.loc[hi_df["Impact (£)"].idxmin()] if not hi_df.empty else None
    worst_str = f"{worst['Ticker']} ({worst['Impact (%)']:+.1f}%)" if worst is not None else "N/A"

    # Estimated new Sharpe (simplified approximation)
    shock_ratio = (1 + impact_pct / 100)
    new_returns = port_returns_f * shock_ratio if shock_ratio > 0 else port_returns_f * 0
    from pension_fund.portfolio_analytics import calculate_sharpe_ratio as _sharpe
    new_sharpe  = _sharpe(new_returns, RISK_FREE_RATE)

    st.markdown("<br>", unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Portfolio Value (Post-Shock)", fmt_gbp(new_val), delta=fmt_gbp(impact_gbp))
    with k2:
        st.metric("Portfolio Impact", f"{impact_pct:+.2f}%", delta=fmt_gbp(impact_gbp))
    with k3:
        orig_sharpe = metrics.get("sharpe_ratio", 0)
        st.metric("New Sharpe Ratio (Est.)", f"{new_sharpe:.3f}", delta=f"{new_sharpe - orig_sharpe:+.3f}")
    with k4:
        st.metric("Worst Hit Holding", worst_str)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 3 — Scenario impact chart + new allocation donut
    col_impact, col_donut = st.columns(2)
    with col_impact:
        if not hi_df.empty:
            fig_impact = plot_scenario_holding_impact(hi_df)
            st.plotly_chart(fig_impact, use_container_width=True)
    with col_donut:
        # Compute post-shock asset class weights
        new_vals  = {row["Asset Class"]: row["New Value"] for _, row in hi_df.iterrows()}
        grp_vals  = {}
        for _, row in hi_df.iterrows():
            ac = row["Asset Class"]
            grp_vals[ac] = grp_vals.get(ac, 0.0) + row["New Value"]
        total_new = sum(grp_vals.values())
        post_shock_weights = {ac: v / total_new for ac, v in grp_vals.items()} if total_new > 0 else {}
        if post_shock_weights:
            fig_post_donut = plot_scenario_allocation_donut(post_shock_weights)
            st.plotly_chart(fig_post_donut, use_container_width=True)

    # Row 4 — Holdings impact table
    st.markdown('<p class="section-header">Holdings Impact Detail</p>', unsafe_allow_html=True)
    if not hi_df.empty:
        hi_display = hi_df[["Ticker", "Name", "Asset Class", "Sector",
                             "Current Value", "Shock (%)", "New Value", "Impact (£)", "Impact (%)"]].copy()
        hi_display["Current Value"] = hi_display["Current Value"].apply(fmt_gbp)
        hi_display["New Value"]     = hi_display["New Value"].apply(fmt_gbp)
        hi_display["Impact (£)"]    = hi_display["Impact (£)"].apply(fmt_gbp)
        hi_display["Shock (%)"]     = hi_display["Shock (%)"].apply(lambda x: f"{x:+.1f}%")
        hi_display["Impact (%)"]    = hi_display["Impact (%)"].apply(lambda x: f"{x:+.1f}%")
        st.dataframe(hi_display, use_container_width=True, hide_index=True)

    # Comparison across all preset scenarios
    st.markdown('<p class="section-header">All Preset Scenarios — Portfolio Impact Comparison</p>', unsafe_allow_html=True)
    scenario_comparison = []
    for sname, sdef in STRESS_SCENARIOS.items():
        result = stress_test_portfolio(HOLDINGS, data["prices_df"], sdef, latest_val)
        scenario_comparison.append({
            "Scenario":     sname,
            "Impact (£M)":  result["total_impact_gbp"] / 1e6,
            "Impact (%)":   result["total_impact_pct"],
        })
    sc_df = pd.DataFrame(scenario_comparison)
    fig_sc = plot_stress_test_results(sc_df)
    st.plotly_chart(fig_sc, use_container_width=True)

    st.info("🔵 **[SIMULATED]** All stress test results use simplified linear shocks applied to current "
            "simulated portfolio values. Real-world stress testing requires full risk-factor models, "
            "non-linear pricing, and liquidity adjustments.")

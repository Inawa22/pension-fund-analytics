"""
portfolio_analytics.py — Core portfolio performance analytics functions.

Provides industry-standard metrics: Sharpe, Sortino, Calmar, Max Drawdown,
Alpha, Beta, Information Ratio, and more.
All return-series inputs should be pandas Series of daily returns.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRADING_DAYS = 252  # Trading days per year


# ---------------------------------------------------------------------------
# Risk-adjusted return metrics
# ---------------------------------------------------------------------------

def calculate_sharpe_ratio(
    returns: pd.Series, risk_free_rate: float = 0.05
) -> float:
    """
    Calculate annualised Sharpe Ratio.

    Sharpe = (Annualised Return - Risk-Free Rate) / Annualised Volatility
    """
    clean = returns.dropna()
    if len(clean) < 2:
        return 0.0
    excess_daily = clean - risk_free_rate / TRADING_DAYS
    annualised_excess = excess_daily.mean() * TRADING_DAYS
    annualised_vol    = clean.std() * np.sqrt(TRADING_DAYS)
    if annualised_vol == 0:
        return 0.0
    return float(annualised_excess / annualised_vol)


def calculate_sortino_ratio(
    returns: pd.Series, risk_free_rate: float = 0.05
) -> float:
    """
    Calculate annualised Sortino Ratio.

    Uses downside deviation (only negative returns) as the denominator.
    Sortino = (Annualised Return - Risk-Free Rate) / Downside Deviation
    """
    clean = returns.dropna()
    if len(clean) < 2:
        return 0.0

    daily_rf          = risk_free_rate / TRADING_DAYS
    excess_returns    = clean - daily_rf
    annualised_excess = excess_returns.mean() * TRADING_DAYS

    # Downside deviation: only negative excess returns contribute
    downside = excess_returns[excess_returns < 0]
    if len(downside) == 0:
        return float("inf")  # No negative returns → infinite Sortino

    downside_deviation = np.sqrt((downside ** 2).mean()) * np.sqrt(TRADING_DAYS)
    if downside_deviation == 0:
        return 0.0
    return float(annualised_excess / downside_deviation)


def calculate_maximum_drawdown(
    returns: pd.Series,
) -> Dict:
    """
    Calculate maximum drawdown and identify peak/trough dates.

    Returns:
        Dict with keys: max_drawdown (float, negative), peak_date, trough_date,
                        recovery_date (or None if not recovered)
    """
    clean  = returns.dropna()
    if len(clean) == 0:
        return {"max_drawdown": 0.0, "peak_date": None, "trough_date": None, "recovery_date": None}

    cumulative = (1 + clean).cumprod()
    rolling_max = cumulative.cummax()
    drawdown    = (cumulative - rolling_max) / rolling_max

    max_dd      = float(drawdown.min())
    trough_idx  = drawdown.idxmin()
    peak_series = rolling_max.loc[:trough_idx]
    peak_idx    = peak_series.idxmax()

    # Look for recovery
    recovery_idx = None
    post_trough  = cumulative.loc[trough_idx:]
    recovered    = post_trough[post_trough >= rolling_max.loc[trough_idx]]
    if not recovered.empty:
        recovery_idx = recovered.index[0]

    return {
        "max_drawdown":   max_dd,
        "peak_date":      peak_idx,
        "trough_date":    trough_idx,
        "recovery_date":  recovery_idx,
    }


def calculate_calmar_ratio(
    returns: pd.Series, risk_free_rate: float = 0.05
) -> float:
    """
    Calculate Calmar Ratio.

    Calmar = Annualised Return / |Max Drawdown|
    """
    try:
        from pension_fund.data_processing import calculate_annualised_return
    except ImportError:
        from data_processing import calculate_annualised_return
    ann_return = calculate_annualised_return(returns)
    dd_info    = calculate_maximum_drawdown(returns)
    max_dd     = abs(dd_info["max_drawdown"])
    if max_dd == 0:
        return 0.0
    return float(ann_return / max_dd)


# ---------------------------------------------------------------------------
# Market metrics (Beta, Alpha, IR)
# ---------------------------------------------------------------------------

def calculate_beta(
    portfolio_returns: pd.Series, benchmark_returns: pd.Series
) -> float:
    """
    Calculate portfolio Beta vs benchmark using OLS regression.

    Beta = Cov(portfolio, benchmark) / Var(benchmark)
    """
    combined = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    if len(combined) < 2:
        return 1.0

    port  = combined.iloc[:, 0]
    bench = combined.iloc[:, 1]

    cov_matrix = np.cov(port, bench)
    var_bench  = cov_matrix[1, 1]
    if var_bench == 0:
        return 1.0
    return float(cov_matrix[0, 1] / var_bench)


def calculate_alpha(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.05,
) -> float:
    """
    Calculate Jensen's Alpha (annualised).

    Alpha = Rp - [Rf + Beta * (Rm - Rf)]
    """
    try:
        from pension_fund.data_processing import calculate_annualised_return
    except ImportError:
        from data_processing import calculate_annualised_return
    beta    = calculate_beta(portfolio_returns, benchmark_returns)
    rp      = calculate_annualised_return(portfolio_returns)
    rm      = calculate_annualised_return(benchmark_returns)
    alpha   = rp - (risk_free_rate + beta * (rm - risk_free_rate))
    return float(alpha)


def calculate_information_ratio(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> Dict:
    """
    Calculate Information Ratio and Tracking Error.

    IR = Active Return / Tracking Error
    Tracking Error = annualised std of (portfolio - benchmark) returns
    """
    combined     = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    active_daily = combined.iloc[:, 0] - combined.iloc[:, 1]
    active_ann   = active_daily.mean() * TRADING_DAYS
    tracking_err = active_daily.std() * np.sqrt(TRADING_DAYS)
    ir           = float(active_ann / tracking_err) if tracking_err != 0 else 0.0
    return {
        "information_ratio": ir,
        "tracking_error":    float(tracking_err),
        "active_return":     float(active_ann),
    }


# ---------------------------------------------------------------------------
# Drawdown series
# ---------------------------------------------------------------------------

def calculate_drawdown_series(returns: pd.Series) -> pd.Series:
    """
    Calculate full drawdown time series (each date's drawdown from prior peak).

    Returns:
        pd.Series with same index as returns, values in [−1, 0].
    """
    clean      = returns.dropna()
    cumulative = (1 + clean).cumprod()
    rolling_max = cumulative.cummax()
    drawdown   = (cumulative - rolling_max) / rolling_max
    return drawdown


# ---------------------------------------------------------------------------
# Composite metrics calculator
# ---------------------------------------------------------------------------

def calculate_all_metrics(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.05,
) -> Dict:
    """
    Calculate all key portfolio performance metrics in one call.

    Returns a dict with:
        annualised_return, annualised_volatility, sharpe_ratio, sortino_ratio,
        max_drawdown, calmar_ratio, beta, alpha, information_ratio, tracking_error,
        ytd_return, mtd_return, skewness, kurtosis
    """
    try:
        from pension_fund.data_processing import calculate_annualised_return
    except ImportError:
        from data_processing import calculate_annualised_return

    clean = portfolio_returns.dropna()

    ann_return = calculate_annualised_return(clean)
    ann_vol    = float(clean.std() * np.sqrt(TRADING_DAYS))
    sharpe     = calculate_sharpe_ratio(clean, risk_free_rate)
    sortino    = calculate_sortino_ratio(clean, risk_free_rate)
    dd_info    = calculate_maximum_drawdown(clean)
    calmar     = calculate_calmar_ratio(clean, risk_free_rate)
    beta       = calculate_beta(clean, benchmark_returns)
    alpha      = calculate_alpha(clean, benchmark_returns, risk_free_rate)
    ir_info    = calculate_information_ratio(clean, benchmark_returns)
    ytd        = calculate_ytd_return(clean)
    mtd        = calculate_mtd_return(clean)

    return {
        "annualised_return":    ann_return,
        "annualised_volatility": ann_vol,
        "sharpe_ratio":         sharpe,
        "sortino_ratio":        sortino,
        "max_drawdown":         dd_info["max_drawdown"],
        "peak_date":            dd_info["peak_date"],
        "trough_date":          dd_info["trough_date"],
        "calmar_ratio":         calmar,
        "beta":                 beta,
        "alpha":                alpha,
        "information_ratio":    ir_info["information_ratio"],
        "tracking_error":       ir_info["tracking_error"],
        "active_return":        ir_info["active_return"],
        "ytd_return":           ytd,
        "mtd_return":           mtd,
        "skewness":             float(clean.skew()),
        "kurtosis":             float(clean.kurtosis()),
    }


# ---------------------------------------------------------------------------
# Periodic return calculations
# ---------------------------------------------------------------------------

def calculate_ytd_return(returns: pd.Series) -> float:
    """
    Calculate Year-to-Date cumulative return.
    Uses returns from the start of the current calendar year to the last date.
    """
    clean = returns.dropna()
    if len(clean) == 0:
        return 0.0
    current_year = clean.index[-1].year
    ytd_returns  = clean[clean.index.year == current_year]
    if len(ytd_returns) == 0:
        return 0.0
    return float((1 + ytd_returns).prod() - 1)


def calculate_mtd_return(returns: pd.Series) -> float:
    """
    Calculate Month-to-Date cumulative return.
    Uses returns from the start of the current calendar month.
    """
    clean = returns.dropna()
    if len(clean) == 0:
        return 0.0
    last        = clean.index[-1]
    mtd_returns = clean[
        (clean.index.year == last.year) & (clean.index.month == last.month)
    ]
    if len(mtd_returns) == 0:
        return 0.0
    return float((1 + mtd_returns).prod() - 1)


# ---------------------------------------------------------------------------
# P&L calculator
# ---------------------------------------------------------------------------

def calculate_daily_pnl(
    holdings: list,
    prices_df: pd.DataFrame,
) -> pd.Series:
    """
    Calculate daily portfolio P&L in £.

    Returns:
        pd.Series of daily P&L (£) with DatetimeIndex.
    """
    try:
        from pension_fund.data_processing import calculate_portfolio_value_series
    except ImportError:
        from data_processing import calculate_portfolio_value_series
    value_series = calculate_portfolio_value_series(holdings, prices_df)
    return value_series.diff()


def calculate_holding_returns(
    holdings: list,
    prices_df: pd.DataFrame,
    period: str = "ytd",
) -> pd.DataFrame:
    """
    Calculate return for each holding over a specified period.

    Args:
        holdings: List of holding dicts.
        prices_df: DataFrame with adjusted_close prices.
        period: One of 'ytd', 'mtd', '1m', '3m', '6m', '1y', 'all'.
    Returns:
        DataFrame with columns: Ticker, Name, Return (%), Sector.
    """
    if prices_df.empty:
        return pd.DataFrame()

    # Filter date range
    today = prices_df.index[-1]
    if period == "ytd":
        start = pd.Timestamp(year=today.year, month=1, day=1)
    elif period == "mtd":
        start = pd.Timestamp(year=today.year, month=today.month, day=1)
    elif period == "1m":
        start = today - pd.DateOffset(months=1)
    elif period == "3m":
        start = today - pd.DateOffset(months=3)
    elif period == "6m":
        start = today - pd.DateOffset(months=6)
    elif period == "1y":
        start = today - pd.DateOffset(years=1)
    else:
        start = prices_df.index[0]

    filtered = prices_df.loc[prices_df.index >= start]
    if filtered.empty:
        return pd.DataFrame()

    records = []
    for h in holdings:
        ticker = h["ticker"]
        if ticker not in filtered.columns:
            continue
        prices = filtered[ticker].dropna()
        if len(prices) < 2:
            continue
        ret = float((prices.iloc[-1] / prices.iloc[0]) - 1) * 100
        records.append({
            "Ticker":     ticker,
            "Name":       h["name"],
            "Return (%)": ret,
            "Sector":     h["sector"],
            "Asset Class": h["asset_class"],
        })

    return pd.DataFrame(records).sort_values("Return (%)", ascending=False)
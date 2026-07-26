"""
risk_analytics.py — Portfolio risk analytics: VaR, CVaR, correlation,
risk contribution, stress testing, and distribution statistics.

Industry-standard implementations suitable for institutional use.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRADING_DAYS = 252
RANDOM_SEED  = 42


# ---------------------------------------------------------------------------
# Value at Risk (VaR)
# ---------------------------------------------------------------------------

def calculate_var(
    returns: pd.Series,
    confidence: float = 0.95,
    method: str = "historical",
    n_simulations: int = 10_000,
) -> float:
    """
    Calculate 1-day Value at Risk (VaR).

    Args:
        returns: Daily return series.
        confidence: Confidence level (e.g. 0.95 for 95% VaR).
        method: 'historical', 'parametric', or 'monte_carlo'.
        n_simulations: Number of MC simulations (used if method='monte_carlo').
    Returns:
        VaR as a positive fraction (e.g. 0.02 = 2% loss).
    """
    clean = returns.dropna()
    if len(clean) == 0:
        return 0.0

    if method == "historical":
        # Historical simulation: empirical quantile
        var = -np.percentile(clean, (1 - confidence) * 100)

    elif method == "parametric":
        # Parametric (Gaussian): uses mean and std
        mu    = clean.mean()
        sigma = clean.std()
        z     = scipy_stats.norm.ppf(1 - confidence)
        var   = -(mu + z * sigma)

    elif method == "monte_carlo":
        # Monte Carlo simulation of 1-day returns
        rng         = np.random.default_rng(RANDOM_SEED)
        mu, sigma   = clean.mean(), clean.std()
        simulated   = rng.normal(mu, sigma, n_simulations)
        var         = -np.percentile(simulated, (1 - confidence) * 100)

    else:
        raise ValueError(f"Unknown VaR method: {method}")

    return float(max(var, 0.0))


def calculate_cvar(
    returns: pd.Series, confidence: float = 0.95
) -> float:
    """
    Calculate Conditional VaR (Expected Shortfall, CVaR) at given confidence level.

    CVaR = mean of returns below VaR threshold (Expected Shortfall).
    Returns a positive fraction (e.g. 0.03 = 3% expected loss beyond VaR).
    """
    clean     = returns.dropna()
    if len(clean) == 0:
        return 0.0
    threshold = np.percentile(clean, (1 - confidence) * 100)
    tail      = clean[clean <= threshold]
    if len(tail) == 0:
        return 0.0
    return float(-tail.mean())


def calculate_var_dollar(
    returns: pd.Series,
    portfolio_value: float,
    confidence: float = 0.95,
    method: str = "historical",
) -> float:
    """
    Calculate 1-day VaR in absolute £ terms.

    Returns:
        VaR as a positive £ value (e.g. £4,500,000 = £4.5M at risk).
    """
    var_pct = calculate_var(returns, confidence, method)
    return float(var_pct * portfolio_value)


def calculate_cvar_dollar(
    returns: pd.Series, portfolio_value: float, confidence: float = 0.95
) -> float:
    """Calculate CVaR/Expected Shortfall in £ terms."""
    cvar_pct = calculate_cvar(returns, confidence)
    return float(cvar_pct * portfolio_value)


# ---------------------------------------------------------------------------
# Correlation and covariance
# ---------------------------------------------------------------------------

def calculate_correlation_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Pearson correlation matrix for all return series.

    Returns:
        DataFrame (n_assets × n_assets) with values in [-1, 1].
    """
    clean = returns_df.dropna()
    return clean.corr()


def calculate_covariance_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate annualised covariance matrix."""
    clean = returns_df.dropna()
    return clean.cov() * TRADING_DAYS


# ---------------------------------------------------------------------------
# Portfolio volatility and risk contribution
# ---------------------------------------------------------------------------

def calculate_portfolio_volatility(
    returns_df: pd.DataFrame, weights: Dict[str, float]
) -> float:
    """
    Calculate annualised portfolio volatility using covariance matrix.

    σ_p = sqrt(w^T Σ w) × sqrt(252)
    """
    clean = returns_df.dropna(how="all")
    tickers = [t for t in weights if t in clean.columns]
    if not tickers:
        return 0.0

    w   = np.array([weights.get(t, 0.0) for t in tickers])
    cov = clean[tickers].cov().values

    if np.any(np.isnan(cov)):
        # Fallback: use average of individual volatilities
        vols = clean[tickers].std().values * np.sqrt(TRADING_DAYS)
        return float(np.dot(w, vols))

    port_var = w @ cov @ w
    return float(np.sqrt(max(port_var, 0)) * np.sqrt(TRADING_DAYS))


def calculate_risk_contribution(
    returns_df: pd.DataFrame, weights: Dict[str, float]
) -> pd.DataFrame:
    """
    Calculate marginal risk contribution (MRC) per asset.

    MRC_i = w_i × (Σw)_i / σ_p

    Returns:
        DataFrame with columns: Ticker, Weight, MRC, Risk Contribution (%),
                                 Marginal Contribution to Vol
    """
    clean   = returns_df.dropna(how="all")
    tickers = [t for t in weights if t in clean.columns]
    if not tickers:
        return pd.DataFrame()

    w   = np.array([weights.get(t, 0.0) for t in tickers])
    cov = clean[tickers].cov().values

    port_vol = np.sqrt(max(w @ cov @ w, 0)) * np.sqrt(TRADING_DAYS)
    if port_vol == 0:
        return pd.DataFrame()

    # Marginal contribution to risk (∂σ/∂w_i)
    mcr = (cov @ w) * np.sqrt(TRADING_DAYS) / port_vol
    # Component risk contribution = w_i × MCR_i
    crc = w * mcr

    df = pd.DataFrame({
        "Ticker":                 tickers,
        "Weight":                 w,
        "Marginal Contribution":  mcr,
        "Risk Contribution":      crc,
        "Risk Contribution (%)":  crc / crc.sum() * 100 if crc.sum() > 0 else 0,
    })
    return df.sort_values("Risk Contribution (%)", ascending=False)


def calculate_component_var(
    returns_df: pd.DataFrame,
    weights: Dict[str, float],
    confidence: float = 0.95,
) -> pd.DataFrame:
    """
    Calculate component VaR per asset (as fraction of portfolio VaR).

    Component VaR_i = w_i × Beta_i × Portfolio VaR
    where Beta_i = Cov(r_i, r_p) / Var(r_p)

    Returns:
        DataFrame with Ticker, Weight, Component VaR (%), Component VaR (fraction).
    """
    clean     = returns_df.dropna(how="all")
    tickers   = [t for t in weights if t in clean.columns]
    if not tickers:
        return pd.DataFrame()

    w         = np.array([weights.get(t, 0.0) for t in tickers])
    port_ret  = clean[tickers].values @ w  # Portfolio daily returns
    port_std  = port_ret.std()
    if port_std == 0:
        return pd.DataFrame()

    var_pct   = calculate_var(pd.Series(port_ret), confidence, "historical")

    records   = []
    for i, ticker in enumerate(tickers):
        asset_ret = clean[ticker].values
        cov_ap    = np.cov(asset_ret, port_ret)[0, 1]
        beta_i    = cov_ap / (port_std ** 2) if port_std != 0 else 0
        comp_var  = w[i] * beta_i * var_pct
        records.append({
            "Ticker":               ticker,
            "Weight":               w[i],
            "Beta to Portfolio":    beta_i,
            "Component VaR (frac)": comp_var,
            "Component VaR (%)":    comp_var * 100,
        })

    return pd.DataFrame(records).sort_values("Component VaR (%)", ascending=False)


# ---------------------------------------------------------------------------
# Stress testing
# ---------------------------------------------------------------------------

def stress_test_portfolio(
    holdings: List[Dict],
    prices_df: pd.DataFrame,
    scenario: Dict,
    portfolio_value: float,
) -> Dict:
    """
    Apply a stress-test scenario to the portfolio and return impacted values.

    Args:
        holdings: List of holding dicts.
        prices_df: Current price DataFrame.
        scenario: Dict with 'shocks_by_asset_class' and 'shocks_by_sector' keys.
                  Each is a dict mapping asset class/sector → shock fraction (e.g. -0.10).
        portfolio_value: Current portfolio value in £.
    Returns:
        Dict with keys:
            new_portfolio_value, total_impact_gbp, total_impact_pct,
            holdings_impact: list of dicts per holding with Ticker, Name,
                             Current Value, Shock %, New Value, Impact £, Impact %
    """
    try:
        from pension_fund.data_processing import calculate_portfolio_weights, calculate_portfolio_value_series
    except ImportError:
        from data_processing import calculate_portfolio_weights, calculate_portfolio_value_series

    weights      = calculate_portfolio_weights(holdings, prices_df)
    value_series = calculate_portfolio_value_series(holdings, prices_df)
    current_val  = float(value_series.iloc[-1]) if not value_series.empty else portfolio_value

    shocks_ac = scenario.get("shocks_by_asset_class", {})
    shocks_sec = scenario.get("shocks_by_sector", {})

    holding_impacts = []
    total_impact    = 0.0

    for h in holdings:
        ticker    = h["ticker"]
        ac        = h["asset_class"]
        sector    = h["sector"]
        w         = weights.get(ticker, 0.0)
        holding_val = w * current_val

        # Determine combined shock
        shock_ac  = shocks_ac.get(ac, 0.0)
        shock_sec = shocks_sec.get(sector, 0.0)
        # Apply asset class shock first, then additional sector shock
        combined_shock = shock_ac + shock_sec * (1 + shock_ac)
        combined_shock = max(min(combined_shock, 1.0), -1.0)  # Clamp to [-100%, +100%]

        new_val    = holding_val * (1 + combined_shock)
        impact_gbp = new_val - holding_val
        impact_pct = combined_shock * 100

        holding_impacts.append({
            "Ticker":        ticker,
            "Name":          h["name"],
            "Asset Class":   ac,
            "Sector":        sector,
            "Current Value": holding_val,
            "Shock (%)":     impact_pct,
            "New Value":     new_val,
            "Impact (£)":    impact_gbp,
            "Impact (%)":    impact_pct,
        })
        total_impact += impact_gbp

    new_portfolio_value = current_val + total_impact
    total_impact_pct    = total_impact / current_val * 100 if current_val != 0 else 0.0

    return {
        "current_portfolio_value": current_val,
        "new_portfolio_value":     new_portfolio_value,
        "total_impact_gbp":        total_impact,
        "total_impact_pct":        total_impact_pct,
        "holdings_impact":         holding_impacts,
    }


def apply_custom_scenario(
    holdings: List[Dict],
    prices_df: pd.DataFrame,
    equity_shock: float,
    bond_shock: float,
    etf_shock: float,
    cash_shock: float,
    tech_additional: float = 0.0,
    financials_additional: float = 0.0,
    portfolio_value: float = 450_000_000,
) -> Dict:
    """
    Apply a custom scenario with per-asset-class and per-sector shocks.
    Wrapper around stress_test_portfolio for the UI sliders.
    """
    scenario = {
        "shocks_by_asset_class": {
            "Equity": equity_shock,
            "Bond":   bond_shock,
            "ETF":    etf_shock,
            "Cash":   cash_shock,
        },
        "shocks_by_sector": {
            "Technology": tech_additional,
            "Financials": financials_additional,
        },
    }
    return stress_test_portfolio(holdings, prices_df, scenario, portfolio_value)


# ---------------------------------------------------------------------------
# Tracking error
# ---------------------------------------------------------------------------

def calculate_tracking_error(
    portfolio_returns: pd.Series, benchmark_returns: pd.Series
) -> float:
    """
    Calculate annualised tracking error.

    TE = std(portfolio_returns - benchmark_returns) × sqrt(252)
    """
    combined     = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    active_daily = combined.iloc[:, 0] - combined.iloc[:, 1]
    return float(active_daily.std() * np.sqrt(TRADING_DAYS))


# ---------------------------------------------------------------------------
# Return distribution statistics
# ---------------------------------------------------------------------------

def calculate_return_distribution_stats(returns: pd.Series) -> Dict:
    """
    Compute comprehensive return distribution statistics.

    Returns:
        Dict with: mean, std, skewness, kurtosis, var_95, cvar_95,
                   var_99, cvar_99, min_return, max_return, pct_positive,
                   normality_pvalue (Shapiro-Wilk, sample ≤ 5000)
    """
    clean = returns.dropna()
    if len(clean) == 0:
        return {}

    try:
        # Shapiro-Wilk test (use sample for large datasets)
        sample = clean if len(clean) <= 5000 else clean.sample(5000, random_state=RANDOM_SEED)
        _, normality_p = scipy_stats.shapiro(sample)
    except Exception:
        normality_p = None

    return {
        "mean":             float(clean.mean() * TRADING_DAYS),  # Annualised
        "std":              float(clean.std() * np.sqrt(TRADING_DAYS)),  # Annualised
        "skewness":         float(clean.skew()),
        "kurtosis":         float(clean.kurtosis()),
        "var_95":           calculate_var(clean, 0.95),
        "cvar_95":          calculate_cvar(clean, 0.95),
        "var_99":           calculate_var(clean, 0.99),
        "cvar_99":          calculate_cvar(clean, 0.99),
        "min_return":       float(clean.min()),
        "max_return":       float(clean.max()),
        "pct_positive":     float((clean > 0).mean() * 100),
        "normality_pvalue": float(normality_p) if normality_p is not None else None,
        "n_observations":   len(clean),
    }
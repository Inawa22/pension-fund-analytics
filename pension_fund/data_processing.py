"""
data_processing.py — Core data transformation and portfolio calculation functions.

All functions are pure (no side effects) and operate on pandas DataFrames/Series.
Returns are documented per function.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRADING_DAYS = 252  # Calendar-adjusted trading days per year


# ---------------------------------------------------------------------------
# Return calculations
# ---------------------------------------------------------------------------

def calculate_daily_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily percentage returns for each ticker.

    Args:
        prices_df: DataFrame with DatetimeIndex and tickers as columns (prices).
    Returns:
        DataFrame of daily simple returns (NaN for first row).
    """
    return prices_df.pct_change()


def calculate_portfolio_returns(
    returns_df: pd.DataFrame, weights: Dict[str, float]
) -> pd.Series:
    """
    Calculate weighted portfolio daily return series.

    Args:
        returns_df: DataFrame of daily returns (columns = tickers).
        weights: Dict mapping ticker → portfolio weight (should sum to 1.0).
    Returns:
        pd.Series of weighted daily portfolio returns.
    """
    # Align weights to DataFrame columns; missing tickers get 0 weight
    w = pd.Series(weights).reindex(returns_df.columns).fillna(0.0)
    # Normalise to sum to 1.0
    if w.sum() > 0:
        w = w / w.sum()
    return returns_df.mul(w, axis=1).sum(axis=1)


def calculate_cumulative_returns(returns_series: pd.Series) -> pd.Series:
    """
    Calculate cumulative product return series (base = 1.0 at start).

    Args:
        returns_series: pd.Series of daily returns.
    Returns:
        pd.Series of cumulative returns (1.0 = no change, 1.10 = +10%).
    """
    return (1 + returns_series).cumprod()


def calculate_annualised_return(returns_series: pd.Series) -> float:
    """
    Calculate annualised geometric return from a daily return series.

    Formula: (product(1 + r_i))^(252/N) - 1
    """
    returns_clean = returns_series.dropna()
    if len(returns_clean) == 0:
        return 0.0
    n = len(returns_clean)
    cumulative = (1 + returns_clean).prod()
    return float(cumulative ** (TRADING_DAYS / n) - 1)


def calculate_rolling_volatility(
    returns_series: pd.Series, window: int = 21
) -> pd.Series:
    """
    Calculate rolling annualised volatility.

    Args:
        returns_series: Daily return series.
        window: Rolling window in trading days (default: 21 ≈ 1 month).
    Returns:
        pd.Series of annualised rolling volatility.
    """
    return returns_series.rolling(window).std() * np.sqrt(TRADING_DAYS)


def calculate_rolling_return(
    returns_series: pd.Series, window: int = 21
) -> pd.Series:
    """
    Calculate rolling cumulative return over a given window.

    Args:
        returns_series: Daily return series.
        window: Rolling window in trading days.
    Returns:
        pd.Series of rolling cumulative returns (not annualised).
    """
    return (1 + returns_series).rolling(window).apply(
        lambda x: x.prod() - 1, raw=True
    )


def calculate_moving_averages(
    price_series: pd.Series, windows: List[int] = None
) -> Dict[str, pd.Series]:
    """
    Calculate simple moving averages for a price series.

    Args:
        price_series: Price series (DatetimeIndex).
        windows: List of window lengths in trading days. Default: [20, 50, 200].
    Returns:
        Dict mapping window label → MA series. E.g. {"MA20": ..., "MA50": ..., "MA200": ...}
    """
    if windows is None:
        windows = [20, 50, 200]
    return {f"MA{w}": price_series.rolling(w).mean() for w in windows}


def calculate_monthly_returns(returns_series: pd.Series) -> pd.DataFrame:
    """
    Calculate monthly returns and return a pivot table of year × month.

    Returns:
        DataFrame with years as index and month numbers (1-12) as columns.
        Values are cumulative returns for that month.
    """
    monthly = (
        (1 + returns_series)
        .resample("ME")
        .prod() - 1
    )
    monthly.index = pd.PeriodIndex(monthly.index, freq="M")
    pivot = monthly.groupby([monthly.index.year, monthly.index.month]).first().unstack()
    pivot.columns.name = "Month"
    pivot.index.name   = "Year"
    return pivot


# ---------------------------------------------------------------------------
# Portfolio weight and value calculations
# ---------------------------------------------------------------------------

def calculate_portfolio_weights(
    holdings: List[Dict], prices_df: pd.DataFrame
) -> Dict[str, float]:
    """
    Calculate current market-value weights for each holding.

    Args:
        holdings: List of holding dicts from config.HOLDINGS.
        prices_df: DataFrame of prices (latest row used for current price).
    Returns:
        Dict mapping ticker → weight (0-1), normalised to sum to 1.0.
    """
    if prices_df.empty:
        return {}

    latest_prices = prices_df.iloc[-1]
    market_values: Dict[str, float] = {}

    for h in holdings:
        ticker = h["ticker"]
        shares = h["shares"]
        cb     = h["cost_basis"]

        if ticker in latest_prices.index:
            price = latest_prices[ticker]
        elif ticker in ("UK_GILT_10Y", "US_TREAS_10Y"):
            # Bond: shares=1, cost_basis is total notional; price is index (normalised to 1)
            # Market value = cost_basis * (price / start_price)
            price = cb  # Simplified: use cost_basis as market value
        elif ticker == "CASH_GBP":
            price = cb
        else:
            price = cb  # Fallback

        # For bonds/cash: shares=1, cost_basis is already total value
        if ticker in ("UK_GILT_10Y", "US_TREAS_10Y", "CASH_GBP"):
            if ticker in latest_prices.index:
                # Price here is the adjusted_close (normalised 0-100 for bonds, 1.0 for cash)
                raw_price = latest_prices[ticker]
                if ticker == "CASH_GBP":
                    mv = cb * raw_price  # cb is initial value, raw_price is growth factor
                else:
                    mv = cb * (raw_price / 100.0)  # Bond price as % of par
            else:
                mv = cb
        else:
            mv = shares * float(price)

        market_values[ticker] = max(mv, 0.0)

    total = sum(market_values.values())
    if total == 0:
        return {t: 0.0 for t in market_values}
    return {t: mv / total for t, mv in market_values.items()}


def calculate_portfolio_value_series(
    holdings: List[Dict], prices_df: pd.DataFrame
) -> pd.Series:
    """
    Calculate total portfolio value over time.

    Args:
        holdings: List of holding dicts.
        prices_df: DataFrame with adjusted_close prices for all tickers.
    Returns:
        pd.Series of portfolio value (£) indexed by date.
    """
    value_series = pd.Series(0.0, index=prices_df.index)

    for h in holdings:
        ticker = h["ticker"]
        shares = h["shares"]
        cb     = h["cost_basis"]

        if ticker not in prices_df.columns:
            # Use constant value for missing tickers
            value_series += cb
            continue

        price_col = prices_df[ticker]

        if ticker in ("UK_GILT_10Y", "US_TREAS_10Y"):
            # Bond: market value = notional * (price / 100)
            value_series += cb * (price_col / 100.0)
        elif ticker == "CASH_GBP":
            # Cash: value = initial_value * cumulative_growth
            value_series += cb * price_col
        else:
            # Equity / ETF: market value = shares * price
            value_series += shares * price_col

    return value_series


def calculate_sector_exposure(
    holdings: List[Dict], prices_df: pd.DataFrame
) -> Dict[str, float]:
    """
    Calculate sector weights as a fraction of total portfolio.

    Returns:
        Dict mapping sector → weight (0-1).
    """
    weights = calculate_portfolio_weights(holdings, prices_df)
    sector_weights: Dict[str, float] = {}

    for h in holdings:
        ticker = h["ticker"]
        sector = h["sector"]
        w      = weights.get(ticker, 0.0)
        sector_weights[sector] = sector_weights.get(sector, 0.0) + w

    total = sum(sector_weights.values())
    if total > 0:
        sector_weights = {s: v / total for s, v in sector_weights.items()}
    return sector_weights


def calculate_asset_class_exposure(
    holdings: List[Dict], prices_df: pd.DataFrame
) -> Dict[str, float]:
    """
    Calculate asset class weights.

    Returns:
        Dict mapping asset_class → weight (0-1).
    """
    weights = calculate_portfolio_weights(holdings, prices_df)
    ac_weights: Dict[str, float] = {}

    for h in holdings:
        ticker = h["ticker"]
        ac     = h["asset_class"]
        w      = weights.get(ticker, 0.0)
        ac_weights[ac] = ac_weights.get(ac, 0.0) + w

    total = sum(ac_weights.values())
    if total > 0:
        ac_weights = {ac: v / total for ac, v in ac_weights.items()}
    return ac_weights


def calculate_geographic_exposure(
    holdings: List[Dict], prices_df: pd.DataFrame
) -> Dict[str, float]:
    """
    Calculate geographic weights.

    Returns:
        Dict mapping geography → weight (0-1).
    """
    weights = calculate_portfolio_weights(holdings, prices_df)
    geo_weights: Dict[str, float] = {}

    for h in holdings:
        ticker = h["ticker"]
        geo    = h["geography"]
        w      = weights.get(ticker, 0.0)
        geo_weights[geo] = geo_weights.get(geo, 0.0) + w

    total = sum(geo_weights.values())
    if total > 0:
        geo_weights = {g: v / total for g, v in geo_weights.items()}
    return geo_weights


def calculate_performance_attribution(
    holdings: List[Dict],
    prices_df: pd.DataFrame,
    benchmark_returns: pd.Series,
) -> pd.DataFrame:
    """
    Calculate performance attribution: each holding's contribution to excess return vs benchmark.

    Method: Brinson-Hood-Beebower simplified.
    - Active weight = portfolio weight - benchmark weight (benchmark = equal-weight of equities)
    - Contribution = weight × (holding return - benchmark return)

    Returns:
        DataFrame with columns: ticker, name, weight, holding_return, benchmark_return,
                                 active_return, contribution
    """
    weights = calculate_portfolio_weights(holdings, prices_df)
    daily_returns = calculate_daily_returns(prices_df)

    bench_ret = float(benchmark_returns.dropna().sum())

    records = []
    for h in holdings:
        ticker = h["ticker"]
        name   = h["name"]
        w      = weights.get(ticker, 0.0)

        if ticker in daily_returns.columns:
            holding_ret = float(daily_returns[ticker].dropna().sum())
        else:
            holding_ret = 0.0

        active_ret   = holding_ret - bench_ret
        contribution = w * active_ret

        records.append({
            "Ticker":           ticker,
            "Name":             name,
            "Weight":           w,
            "Holding Return":   holding_ret,
            "Benchmark Return": bench_ret,
            "Active Return":    active_ret,
            "Contribution":     contribution,
        })

    df = pd.DataFrame(records).sort_values("Contribution", ascending=False)
    return df


# ---------------------------------------------------------------------------
# Holdings table builder
# ---------------------------------------------------------------------------

def build_holdings_df(
    holdings: List[Dict],
    prices_df: pd.DataFrame,
    fundamentals: Dict[str, Dict],
) -> pd.DataFrame:
    """
    Build a comprehensive holdings DataFrame for display in Tab 4.

    Columns: Ticker, Name, Asset Class, Sector, Geography, Shares, Current Price,
             Market Value (£), Weight (%), Cost Basis, Unrealised P&L (£),
             Unrealised P&L (%), P/E Ratio, Div Yield (%), Beta,
             Market Cap (£B), EPS, 52W High, 52W Low
    """
    latest_prices = prices_df.iloc[-1] if not prices_df.empty else pd.Series(dtype=float)
    weights       = calculate_portfolio_weights(holdings, prices_df)
    total_value   = calculate_portfolio_value_series(holdings, prices_df).iloc[-1]

    records = []
    for h in holdings:
        ticker = h["ticker"]
        shares = h["shares"]
        cb     = h["cost_basis"]
        name   = h["name"]

        # Current price
        if ticker in latest_prices.index:
            current_price = float(latest_prices[ticker])
        else:
            current_price = cb

        # Market value
        if ticker in ("UK_GILT_10Y", "US_TREAS_10Y"):
            if ticker in latest_prices.index:
                mv = cb * (float(latest_prices[ticker]) / 100.0)
            else:
                mv = cb
        elif ticker == "CASH_GBP":
            if ticker in latest_prices.index:
                mv = cb * float(latest_prices[ticker])
            else:
                mv = cb
        else:
            mv = shares * current_price

        # Cost / P&L
        if ticker in ("UK_GILT_10Y", "US_TREAS_10Y", "CASH_GBP"):
            total_cost = cb
        else:
            total_cost = shares * cb

        pnl      = mv - total_cost
        pnl_pct  = (pnl / total_cost * 100) if total_cost != 0 else 0.0

        # Fundamentals
        fund = fundamentals.get(ticker, {})
        pe        = fund.get("PE_ratio", None)
        div_yield = fund.get("DividendYield", None)
        beta      = fund.get("Beta", None)
        mkt_cap   = fund.get("MarketCap", None)
        eps       = fund.get("EPS", None)
        high_52w  = fund.get("52WeekHigh", None)
        low_52w   = fund.get("52WeekLow", None)

        records.append({
            "Ticker":               ticker,
            "Name":                 name,
            "Asset Class":          h["asset_class"],
            "Sector":               h["sector"],
            "Geography":            h["geography"],
            "Shares":               f"{shares:,}",
            "Current Price":        current_price,
            "Market Value (£)":     mv,
            "Weight (%)":           weights.get(ticker, 0.0) * 100,
            "Cost Basis":           cb,
            "Unrealised P&L (£)":   pnl,
            "Unrealised P&L (%)":   pnl_pct,
            "P/E Ratio":            pe,
            "Div Yield (%)":        (div_yield * 100) if div_yield is not None else None,
            "Beta":                 beta,
            "Market Cap (£B)":      (mkt_cap / 1e9) if mkt_cap else None,
            "EPS":                  eps,
            "52W High":             high_52w,
            "52W Low":              low_52w,
        })

    return pd.DataFrame(records)
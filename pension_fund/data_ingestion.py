"""
data_ingestion.py — Alpha Vantage data fetcher with file-based caching and simulated fallback.

Features:
  - File-based caching (JSON/CSV in ./pension_fund/cache/) — expires after 24 hours
  - Rate limiting: 5 calls/minute for Alpha Vantage free tier with automatic delay
  - Simulated fallback using Geometric Brownian Motion (GBM) with realistic parameters
  - Full simulated data mode when no API key is provided
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Module-level configuration
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

CACHE_DIR         = Path("./pension_fund/cache")
CACHE_TTL_HOURS   = 24
RATE_LIMIT_CALLS  = 5       # Alpha Vantage free tier: 5 calls/min
RATE_LIMIT_WINDOW = 60.0    # seconds

AV_BASE_URL = "https://www.alphavantage.co/query"

# Track API call timestamps for rate limiting
_api_call_times: List[float] = []

# Simulation start date
SIM_START_DATE = "2022-01-01"

# Random seed for reproducibility
RANDOM_SEED = 42

# GBM parameters per sector
GBM_PARAMS = {
    "Technology":      {"mu": 0.15,  "sigma": 0.28},
    "Financials":      {"mu": 0.10,  "sigma": 0.20},
    "Healthcare":      {"mu": 0.08,  "sigma": 0.16},
    "Energy":          {"mu": 0.12,  "sigma": 0.25},
    "Consumer Disc.":  {"mu": 0.09,  "sigma": 0.22},
    "Broad Market":    {"mu": 0.10,  "sigma": 0.18},  # SPY, QQQ
    "Fixed Income":    {"mu": 0.02,  "sigma": 0.05},  # AGG
    "Government Bond": {"mu": 0.03,  "sigma": 0.04},  # Bonds
    "Cash":            {"mu": 0.05,  "sigma": 0.001}, # Near-zero vol
}

# Starting prices for simulation (approximate 2022-01-01 prices)
SIM_START_PRICES = {
    "AAPL":         182.0,
    "MSFT":         335.0,
    "GOOGL":        144.0,
    "NVDA":         294.0,
    "JPM":          165.0,
    "JNJ":          170.0,
    "XOM":          106.0,
    "UNH":          510.0,
    "HD":           368.0,
    "BRK-B":        310.0,
    "SPY":          473.0,
    "QQQ":          394.0,
    "AGG":           94.0,
    "UK_GILT_10Y":  100.0,  # Par value
    "US_TREAS_10Y": 100.0,  # Par value
    "CASH_GBP":       1.0,  # Normalised unit value
}

# Ticker → sector mapping (for GBM param lookup)
TICKER_SECTOR = {
    "AAPL":  "Technology", "MSFT":  "Technology", "GOOGL": "Technology",
    "NVDA":  "Technology", "QQQ":   "Technology",
    "JPM":   "Financials", "BRK-B": "Financials",
    "JNJ":   "Healthcare", "UNH":   "Healthcare",
    "XOM":   "Energy",
    "HD":    "Consumer Disc.",
    "SPY":   "Broad Market",
    "AGG":   "Fixed Income",
    "UK_GILT_10Y":  "Government Bond",
    "US_TREAS_10Y": "Government Bond",
    "CASH_GBP":     "Cash",
}

# Technology stock correlation matrix (AAPL, MSFT, GOOGL, NVDA, QQQ)
TECH_CORRELATION = np.array([
    [1.00, 0.78, 0.72, 0.65, 0.85],  # AAPL
    [0.78, 1.00, 0.76, 0.68, 0.88],  # MSFT
    [0.72, 0.76, 1.00, 0.70, 0.82],  # GOOGL
    [0.65, 0.68, 0.70, 1.00, 0.75],  # NVDA
    [0.85, 0.88, 0.82, 0.75, 1.00],  # QQQ
])
TECH_TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "QQQ"]

# ---------------------------------------------------------------------------
# Cache utilities
# ---------------------------------------------------------------------------

def _cache_path(key: str, ext: str = "json") -> Path:
    """Return path to a cache file for the given key."""
    safe_key = key.replace("/", "_").replace("\\", "_").replace("-", "_")
    return CACHE_DIR / f"{safe_key}.{ext}"


def _is_cache_valid(path: Path, ttl_hours: int = CACHE_TTL_HOURS) -> bool:
    """Return True if cache file exists and is younger than ttl_hours."""
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return datetime.now() - mtime < timedelta(hours=ttl_hours)


def _load_cache_json(path: Path) -> Optional[dict]:
    """Load JSON from cache file."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Cache load error {path}: {e}")
        return None


def _save_cache_json(path: Path, data: dict) -> None:
    """Save data as JSON to cache file."""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)
    except Exception as e:
        logger.warning(f"Cache save error {path}: {e}")


def _load_cache_csv(path: Path) -> Optional[pd.DataFrame]:
    """Load a CSV from cache."""
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df
    except Exception as e:
        logger.warning(f"Cache CSV load error {path}: {e}")
        return None


def _save_cache_csv(path: Path, df: pd.DataFrame) -> None:
    """Save DataFrame as CSV to cache."""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(path)
    except Exception as e:
        logger.warning(f"Cache CSV save error {path}: {e}")


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

def _rate_limit_wait() -> None:
    """Block until it is safe to make another Alpha Vantage API call (5/min)."""
    global _api_call_times
    now = time.time()
    # Remove timestamps older than the rate-limit window
    _api_call_times = [t for t in _api_call_times if now - t < RATE_LIMIT_WINDOW]
    if len(_api_call_times) >= RATE_LIMIT_CALLS:
        wait_time = RATE_LIMIT_WINDOW - (now - _api_call_times[0]) + 1.0
        if wait_time > 0:
            logger.info(f"Rate limit reached — waiting {wait_time:.1f}s")
            time.sleep(wait_time)
    _api_call_times.append(time.time())


# ---------------------------------------------------------------------------
# Alpha Vantage API calls
# ---------------------------------------------------------------------------

def get_daily_prices(ticker: str, api_key: str) -> Optional[pd.DataFrame]:
    """
    Fetch daily OHLCV prices from Alpha Vantage (TIME_SERIES_DAILY_ADJUSTED).

    Returns a DataFrame with DatetimeIndex and columns:
        open, high, low, close, adjusted_close, volume
    Returns None if the API call fails.
    """
    # Normalise ticker for cache key (BRK-B → BRK_B)
    cache_key = f"prices_{ticker}"
    cache_path = _cache_path(cache_key, "csv")

    if _is_cache_valid(cache_path):
        logger.info(f"Cache hit: {ticker} prices")
        return _load_cache_csv(cache_path)

    _rate_limit_wait()
    try:
        import requests
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": ticker,
            "outputsize": "full",
            "apikey": api_key,
        }
        response = requests.get(AV_BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if "Time Series (Daily)" not in data:
            error_msg = data.get("Note", data.get("Information", "Unknown error"))
            logger.warning(f"Alpha Vantage error for {ticker}: {error_msg}")
            return None

        ts = data["Time Series (Daily)"]
        records = []
        for date_str, ohlcv in ts.items():
            records.append({
                "date":           pd.Timestamp(date_str),
                "open":           float(ohlcv.get("1. open", 0)),
                "high":           float(ohlcv.get("2. high", 0)),
                "low":            float(ohlcv.get("3. low", 0)),
                "close":          float(ohlcv.get("4. close", 0)),
                "adjusted_close": float(ohlcv.get("5. adjusted close", 0)),
                "volume":         float(ohlcv.get("6. volume", 0)),
            })

        df = pd.DataFrame(records).set_index("date").sort_index()
        _save_cache_csv(cache_path, df)
        logger.info(f"Fetched {len(df)} rows for {ticker}")
        return df

    except Exception as e:
        logger.error(f"Failed to fetch prices for {ticker}: {e}")
        return None


def get_fundamentals(ticker: str, api_key: str) -> Dict:
    """
    Fetch company overview / fundamentals from Alpha Vantage (OVERVIEW endpoint).

    Returns a dict with keys:
        MarketCap, PE_ratio, DividendYield, Beta, EPS, 52WeekHigh, 52WeekLow,
        Sector, Industry, Description
    Returns an empty dict if the API call fails.
    """
    cache_key  = f"fundamentals_{ticker}"
    cache_path = _cache_path(cache_key, "json")

    if _is_cache_valid(cache_path):
        data = _load_cache_json(cache_path)
        if data:
            return data

    _rate_limit_wait()
    try:
        import requests
        params = {
            "function": "OVERVIEW",
            "symbol":   ticker,
            "apikey":   api_key,
        }
        response = requests.get(AV_BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        raw = response.json()

        if not raw or "Symbol" not in raw:
            logger.warning(f"Fundamentals unavailable for {ticker}")
            return {}

        result = {
            "MarketCap":     _safe_float(raw.get("MarketCapitalization")),
            "PE_ratio":      _safe_float(raw.get("PERatio")),
            "DividendYield": _safe_float(raw.get("DividendYield")),
            "Beta":          _safe_float(raw.get("Beta")),
            "EPS":           _safe_float(raw.get("EPS")),
            "52WeekHigh":    _safe_float(raw.get("52WeekHigh")),
            "52WeekLow":     _safe_float(raw.get("52WeekLow")),
            "Sector":        raw.get("Sector", ""),
            "Industry":      raw.get("Industry", ""),
            "Description":   raw.get("Description", ""),
        }
        _save_cache_json(cache_path, result)
        return result

    except Exception as e:
        logger.error(f"Failed to fetch fundamentals for {ticker}: {e}")
        return {}


# ---------------------------------------------------------------------------
# Simulated data generators
# ---------------------------------------------------------------------------

def _trading_date_range(start: str = SIM_START_DATE, end: Optional[str] = None) -> pd.DatetimeIndex:
    """Return a business-day DatetimeIndex from start to end (default: today)."""
    end_date = end or datetime.today().strftime("%Y-%m-%d")
    return pd.bdate_range(start=start, end=end_date)


def _gbm_prices(
    S0: float,
    mu: float,
    sigma: float,
    n_steps: int,
    rng: np.random.Generator,
    dt: float = 1 / 252,
) -> np.ndarray:
    """
    Simulate price path using Geometric Brownian Motion.

    S_t = S_{t-1} * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    """
    Z = rng.standard_normal(n_steps)
    log_returns = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    prices = S0 * np.exp(np.cumsum(log_returns))
    return np.concatenate([[S0], prices[:-1]])  # Shift so first value = S0


def get_simulated_bond_data(
    ticker: str, years: int = 3
) -> pd.DataFrame:
    """
    Generate realistic bond price time series.
    Prices mean-revert around 100 (par) with slight drift toward yield income.
    Returns DataFrame with adjusted_close column.
    """
    rng   = np.random.default_rng(RANDOM_SEED + hash(ticker) % 1000)
    dates = _trading_date_range()
    n     = len(dates)

    params = GBM_PARAMS.get("Government Bond", {"mu": 0.03, "sigma": 0.04})
    S0     = SIM_START_PRICES.get(ticker, 100.0)

    # Mean-reverting Ornstein-Uhlenbeck process around par
    kappa   = 0.3   # Mean-reversion speed
    theta   = S0    # Long-run mean (par)
    dt      = 1 / 252
    prices  = [S0]
    for _ in range(n - 1):
        prev    = prices[-1]
        shock   = rng.standard_normal()
        dp      = kappa * (theta - prev) * dt + params["sigma"] * np.sqrt(dt) * shock * theta
        prices.append(max(prev + dp, 50.0))  # Floor at 50

    prices = np.array(prices)
    df = pd.DataFrame(
        {
            "open":           prices,
            "high":           prices * (1 + rng.uniform(0, 0.002, n)),
            "low":            prices * (1 - rng.uniform(0, 0.002, n)),
            "close":          prices,
            "adjusted_close": prices,
            "volume":         np.zeros(n),
        },
        index=dates,
    )
    return df


def get_simulated_cash_data(
    start_date: Optional[str] = None, end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate flat cash position growing at the risk-free rate (daily compounding).
    Returns DataFrame with adjusted_close column.
    """
    dates = _trading_date_range(
        start=start_date or SIM_START_DATE,
        end=end_date,
    )
    n    = len(dates)
    rate = GBM_PARAMS["Cash"]["mu"]  # 5% p.a.
    dt   = 1 / 252

    # Compound daily: V_t = V_0 * (1 + rate*dt)^t
    growth = np.array([(1 + rate * dt) ** i for i in range(n)])
    prices = growth  # Normalised to 1.0 base

    df = pd.DataFrame(
        {
            "open":           prices,
            "high":           prices,
            "low":            prices,
            "close":          prices,
            "adjusted_close": prices,
            "volume":         np.zeros(n),
        },
        index=dates,
    )
    return df


def _simulate_correlated_tech_prices(dates: pd.DatetimeIndex, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """
    Simulate correlated price paths for technology stocks using Cholesky decomposition.
    Ensures realistic co-movement between AAPL, MSFT, GOOGL, NVDA, QQQ.
    """
    n      = len(dates)
    dt     = 1 / 252
    params = {t: GBM_PARAMS.get(TICKER_SECTOR.get(t, "Broad Market")) for t in TECH_TICKERS}

    # Cholesky decomposition for correlated normals
    L = np.linalg.cholesky(TECH_CORRELATION)
    Z = rng.standard_normal((n, len(TECH_TICKERS)))
    corr_Z = Z @ L.T  # Shape: (n, 5)

    results = {}
    for i, ticker in enumerate(TECH_TICKERS):
        mu    = params[ticker]["mu"]
        sigma = params[ticker]["sigma"]
        S0    = SIM_START_PRICES.get(ticker, 100.0)
        log_r = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * corr_Z[:, i]
        prices = S0 * np.exp(np.cumsum(log_r))
        results[ticker] = np.concatenate([[S0], prices[:-1]])

    return results


def simulate_all_data() -> Dict[str, pd.DataFrame]:
    """
    Generate FULL realistic simulated data for ALL holdings without an API key.

    Returns a dict mapping ticker → DataFrame with columns:
        open, high, low, close, adjusted_close, volume

    Uses:
    - Correlated GBM for technology stocks
    - Independent GBM for non-tech equities
    - Ornstein-Uhlenbeck for bonds
    - Deterministic compounding for cash
    All paths start from historically-plausible 2022-01-01 prices.
    """
    rng   = np.random.default_rng(RANDOM_SEED)
    dates = _trading_date_range()
    n     = len(dates)
    dt    = 1 / 252

    # --- Correlated tech prices ---
    tech_prices = _simulate_correlated_tech_prices(dates, rng)

    all_prices: Dict[str, pd.DataFrame] = {}

    try:
        from pension_fund.config import HOLDINGS, SIMULATED_TICKERS
    except ImportError:
        from config import HOLDINGS, SIMULATED_TICKERS
    try:
        from pension_fund.config import EQUITY_TICKERS, ETF_TICKERS
    except ImportError:
        from config import EQUITY_TICKERS, ETF_TICKERS

    for holding in HOLDINGS:
        ticker = holding["ticker"]
        sector = holding["sector"]

        if ticker in SIMULATED_TICKERS:
            if "GILT" in ticker or "TREAS" in ticker:
                all_prices[ticker] = get_simulated_bond_data(ticker)
            else:  # CASH
                all_prices[ticker] = get_simulated_cash_data()
            continue

        if ticker in TECH_TICKERS and ticker in tech_prices:
            prices = tech_prices[ticker]
        else:
            # Independent GBM for non-tech assets
            params = GBM_PARAMS.get(sector, {"mu": 0.08, "sigma": 0.18})
            S0     = SIM_START_PRICES.get(ticker, 100.0)
            # Use a deterministic sub-RNG per ticker
            sub_rng = np.random.default_rng(RANDOM_SEED + abs(hash(ticker)) % 10_000)
            prices  = _gbm_prices(S0, params["mu"], params["sigma"], n, sub_rng, dt)

        # Add OHLC spread and volume
        spread_pct = 0.005  # 0.5% daily OHLC spread
        sub_rng    = np.random.default_rng(RANDOM_SEED + abs(hash(ticker + "_v")) % 10_000)
        highs  = prices * (1 + sub_rng.uniform(0, spread_pct, n))
        lows   = prices * (1 - sub_rng.uniform(0, spread_pct, n))
        opens  = prices * (1 + sub_rng.uniform(-0.002, 0.002, n))
        volume = sub_rng.uniform(5_000_000, 50_000_000, n)

        all_prices[ticker] = pd.DataFrame(
            {
                "open":           opens,
                "high":           highs,
                "low":            lows,
                "close":          prices,
                "adjusted_close": prices,
                "volume":         volume,
            },
            index=dates,
        )

    return all_prices


def load_all_prices(
    api_key: Optional[str],
    tickers: List[str],
) -> Dict[str, pd.DataFrame]:
    """
    Load prices for all tickers.
    - For REAL_TICKERS: tries Alpha Vantage if api_key provided; falls back to simulation.
    - For SIMULATED_TICKERS: always uses simulated data.

    Returns: dict mapping ticker → DataFrame with adjusted_close column.
    """
    try:
        from pension_fund.config import SIMULATED_TICKERS, REAL_TICKERS
    except ImportError:
        from config import SIMULATED_TICKERS, REAL_TICKERS

    results: Dict[str, pd.DataFrame] = {}

    # Always simulate bonds and cash
    for ticker in SIMULATED_TICKERS:
        if "GILT" in ticker or "TREAS" in ticker:
            results[ticker] = get_simulated_bond_data(ticker)
        else:
            results[ticker] = get_simulated_cash_data()

    if not api_key:
        # Full simulation mode
        sim_data = simulate_all_data()
        for ticker in tickers:
            if ticker not in results:
                if ticker in sim_data:
                    results[ticker] = sim_data[ticker]
        return results

    # Try real API with fallback
    sim_data = None  # Lazy init to avoid unnecessary computation

    for ticker in tickers:
        if ticker in results:
            continue  # Already have simulated bond/cash data
        if ticker not in REAL_TICKERS:
            continue

        df = get_daily_prices(ticker, api_key)
        if df is not None and not df.empty:
            results[ticker] = df
        else:
            # API failed → simulate
            if sim_data is None:
                sim_data = simulate_all_data()
            if ticker in sim_data:
                results[ticker] = sim_data[ticker]
                logger.warning(f"{ticker}: API failed, using simulated data")

    return results


def get_benchmark_data(api_key: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Return SPY price series (benchmark).
    Uses real Alpha Vantage data if api_key provided; otherwise simulates.
    """
    if api_key:
        df = get_daily_prices("SPY", api_key)
        if df is not None:
            return df

    # Simulated benchmark
    sim_data = simulate_all_data()
    return sim_data.get("SPY")


def get_simulated_fundamentals(ticker: str) -> Dict:
    """
    Return simulated fundamental data for a ticker (used as fallback).
    Values are plausible but fictional — labelled SIMULATED in the app.
    """
    try:
        from pension_fund.config import DIVIDEND_YIELDS
    except ImportError:
        from config import DIVIDEND_YIELDS

    # Base data per ticker (approximate 2024 estimates)
    base_data = {
        "AAPL":  {"MarketCap": 3_000_000_000_000, "PE_ratio": 31.0, "Beta": 1.25, "EPS": 6.57, "52WeekHigh": 237.0, "52WeekLow": 164.0},
        "MSFT":  {"MarketCap": 3_100_000_000_000, "PE_ratio": 36.0, "Beta": 0.88, "EPS": 11.45, "52WeekHigh": 468.0, "52WeekLow": 310.0},
        "GOOGL": {"MarketCap": 2_100_000_000_000, "PE_ratio": 27.0, "Beta": 1.05, "EPS": 8.04, "52WeekHigh": 196.0, "52WeekLow": 129.0},
        "NVDA":  {"MarketCap": 2_900_000_000_000, "PE_ratio": 65.0, "Beta": 1.68, "EPS": 16.84, "52WeekHigh": 974.0, "52WeekLow": 403.0},
        "JPM":   {"MarketCap":  580_000_000_000, "PE_ratio": 12.0, "Beta": 1.12, "EPS": 18.22, "52WeekHigh": 220.0, "52WeekLow": 144.0},
        "JNJ":   {"MarketCap":  365_000_000_000, "PE_ratio": 15.0, "Beta": 0.52, "EPS": 9.98, "52WeekHigh": 163.0, "52WeekLow": 143.0},
        "XOM":   {"MarketCap":  520_000_000_000, "PE_ratio": 14.0, "Beta": 0.80, "EPS": 8.89, "52WeekHigh": 123.0, "52WeekLow":  95.0},
        "UNH":   {"MarketCap":  470_000_000_000, "PE_ratio": 22.0, "Beta": 0.56, "EPS": 27.56, "52WeekHigh": 565.0, "52WeekLow": 450.0},
        "HD":    {"MarketCap":  380_000_000_000, "PE_ratio": 26.0, "Beta": 1.02, "EPS": 15.14, "52WeekHigh": 395.0, "52WeekLow": 274.0},
        "BRK-B": {"MarketCap":  880_000_000_000, "PE_ratio": 10.0, "Beta": 0.88, "EPS": 31.20, "52WeekHigh": 395.0, "52WeekLow": 317.0},
        "SPY":   {"MarketCap":  450_000_000_000, "PE_ratio": 22.0, "Beta": 1.00, "EPS": 22.00, "52WeekHigh": 540.0, "52WeekLow": 420.0},
        "QQQ":   {"MarketCap":  250_000_000_000, "PE_ratio": 31.0, "Beta": 1.10, "EPS": 18.00, "52WeekHigh": 495.0, "52WeekLow": 360.0},
        "AGG":   {"MarketCap":  100_000_000_000, "PE_ratio":  0.0, "Beta": -0.05, "EPS": 3.90, "52WeekHigh": 100.0, "52WeekLow":  90.0},
        "UK_GILT_10Y":  {"MarketCap": 0, "PE_ratio": 0, "Beta": -0.10, "EPS": 0, "52WeekHigh": 102.0, "52WeekLow": 94.0},
        "US_TREAS_10Y": {"MarketCap": 0, "PE_ratio": 0, "Beta": -0.08, "EPS": 0, "52WeekHigh": 102.0, "52WeekLow": 95.0},
        "CASH_GBP":     {"MarketCap": 0, "PE_ratio": 0, "Beta":  0.00, "EPS": 0, "52WeekHigh": 1.0,   "52WeekLow": 1.0},
    }
    data = base_data.get(ticker, {"MarketCap": 0, "PE_ratio": 0, "Beta": 1.0, "EPS": 0, "52WeekHigh": 0, "52WeekLow": 0})
    data["DividendYield"] = DIVIDEND_YIELDS.get(ticker, 0.0)
    return data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(value) -> Optional[float]:
    """Convert a value to float, returning None if conversion fails."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
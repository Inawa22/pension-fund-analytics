"""
config.py — Portfolio configuration for Northgate Pension Fund (simulated institutional fund).
Defines holdings, weights, sector mappings, geographic mappings, and global constants.
"""

PORTFOLIO_NAME = "Northgate Institutional Pension Fund"
FUND_NAV = 450_000_000  # £450M AUM

# ---------------------------------------------------------------------------
# Holdings: (ticker, name, asset_class, sector, geography, shares, cost_basis_per_share)
# ---------------------------------------------------------------------------
HOLDINGS = [
    # Equities - Large Cap US
    {"ticker": "AAPL",        "name": "Apple Inc.",                    "asset_class": "Equity", "sector": "Technology",      "geography": "North America", "shares": 250_000, "cost_basis": 145.0},
    {"ticker": "MSFT",        "name": "Microsoft Corp.",               "asset_class": "Equity", "sector": "Technology",      "geography": "North America", "shares": 180_000, "cost_basis": 280.0},
    {"ticker": "GOOGL",       "name": "Alphabet Inc.",                 "asset_class": "Equity", "sector": "Technology",      "geography": "North America", "shares":  95_000, "cost_basis": 130.0},
    {"ticker": "NVDA",        "name": "NVIDIA Corp.",                  "asset_class": "Equity", "sector": "Technology",      "geography": "North America", "shares": 120_000, "cost_basis": 450.0},
    {"ticker": "JPM",         "name": "JPMorgan Chase & Co.",          "asset_class": "Equity", "sector": "Financials",      "geography": "North America", "shares": 280_000, "cost_basis": 155.0},
    {"ticker": "JNJ",         "name": "Johnson & Johnson",             "asset_class": "Equity", "sector": "Healthcare",      "geography": "North America", "shares": 160_000, "cost_basis": 165.0},
    {"ticker": "XOM",         "name": "ExxonMobil Corp.",              "asset_class": "Equity", "sector": "Energy",          "geography": "North America", "shares": 220_000, "cost_basis":  95.0},
    {"ticker": "UNH",         "name": "UnitedHealth Group Inc.",       "asset_class": "Equity", "sector": "Healthcare",      "geography": "North America", "shares":  55_000, "cost_basis": 480.0},
    {"ticker": "HD",          "name": "Home Depot Inc.",               "asset_class": "Equity", "sector": "Consumer Disc.",  "geography": "North America", "shares":  90_000, "cost_basis": 310.0},
    {"ticker": "BRK-B",       "name": "Berkshire Hathaway B",          "asset_class": "Equity", "sector": "Financials",      "geography": "North America", "shares": 300_000, "cost_basis": 310.0},
    # ETFs
    {"ticker": "SPY",         "name": "SPDR S&P 500 ETF",             "asset_class": "ETF",    "sector": "Broad Market",    "geography": "North America", "shares": 150_000, "cost_basis": 420.0},
    {"ticker": "QQQ",         "name": "Invesco QQQ Trust",             "asset_class": "ETF",    "sector": "Technology",      "geography": "North America", "shares":  80_000, "cost_basis": 350.0},
    {"ticker": "AGG",         "name": "iShares Core US Agg Bond",      "asset_class": "ETF",    "sector": "Fixed Income",    "geography": "North America", "shares": 200_000, "cost_basis":  98.0},
    # Simulated Bonds
    {"ticker": "UK_GILT_10Y", "name": "UK Gilt 10Y (Simulated)",       "asset_class": "Bond",   "sector": "Government Bond", "geography": "Europe",        "shares": 1,       "cost_basis": 35_000_000},
    {"ticker": "US_TREAS_10Y","name": "US Treasury 10Y (Simulated)",   "asset_class": "Bond",   "sector": "Government Bond", "geography": "North America", "shares": 1,       "cost_basis": 40_000_000},
    # Cash
    {"ticker": "CASH_GBP",    "name": "Cash & Money Market (GBP)",     "asset_class": "Cash",   "sector": "Cash",            "geography": "Europe",        "shares": 1,       "cost_basis": 22_500_000},
]

# ---------------------------------------------------------------------------
# Ticker groupings
# ---------------------------------------------------------------------------
EQUITY_TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "JPM", "JNJ", "XOM", "UNH", "HD", "BRK-B"]
ETF_TICKERS    = ["SPY", "QQQ", "AGG"]
REAL_TICKERS   = EQUITY_TICKERS + ETF_TICKERS   # Tickers with real Alpha Vantage data
SIMULATED_TICKERS = ["UK_GILT_10Y", "US_TREAS_10Y", "CASH_GBP"]

BENCHMARK_TICKER = "SPY"   # Use SPY as S&P 500 proxy

# ---------------------------------------------------------------------------
# Market / risk parameters
# ---------------------------------------------------------------------------
RISK_FREE_RATE         = 0.05   # 5% annualised (approx UK/US short-term rate 2024)
TRADING_DAYS_PER_YEAR  = 252
VAR_CONFIDENCE         = 0.95   # Default VaR confidence level
CACHE_TTL_HOURS        = 24     # Cache expiry in hours

# ---------------------------------------------------------------------------
# Geometric Brownian Motion parameters per sector (used in simulation)
# ---------------------------------------------------------------------------
GBM_PARAMS = {
    "Technology":     {"mu": 0.15, "sigma": 0.28},
    "Financials":     {"mu": 0.10, "sigma": 0.20},
    "Healthcare":     {"mu": 0.08, "sigma": 0.16},
    "Energy":         {"mu": 0.12, "sigma": 0.25},
    "Consumer Disc.": {"mu": 0.09, "sigma": 0.22},
    "Broad Market":   {"mu": 0.10, "sigma": 0.18},  # SPY, QQQ
    "Fixed Income":   {"mu": 0.02, "sigma": 0.05},  # AGG
    "Government Bond":{"mu": 0.03, "sigma": 0.04},  # UK Gilt, US Treasury
    "Cash":           {"mu": 0.05, "sigma": 0.001}, # Cash — grows at risk-free rate
}

# ---------------------------------------------------------------------------
# Dividend yields per ticker (annualised, approximate)
# ---------------------------------------------------------------------------
DIVIDEND_YIELDS = {
    "AAPL":   0.005,   # 0.5%
    "MSFT":   0.008,   # 0.8%
    "GOOGL":  0.005,   # 0.5%
    "NVDA":   0.0003,  # 0.03%
    "JPM":    0.025,   # 2.5%
    "JNJ":    0.030,   # 3.0%
    "XOM":    0.035,   # 3.5%
    "UNH":    0.015,   # 1.5%
    "HD":     0.023,   # 2.3%
    "BRK-B":  0.000,   # No dividend
    "SPY":    0.013,   # 1.3% (index ETF)
    "QQQ":    0.006,   # 0.6% (tech ETF)
    "AGG":    0.040,   # 4.0% (bond ETF)
    "UK_GILT_10Y":  0.042,  # 4.2% yield
    "US_TREAS_10Y": 0.045,  # 4.5% yield
    "CASH_GBP":     0.050,  # 5.0% MMF rate
}

# Dividend payment frequency per ticker
DIVIDEND_FREQUENCY = {
    "AAPL":   "Quarterly",
    "MSFT":   "Quarterly",
    "GOOGL":  "Quarterly",
    "NVDA":   "Quarterly",
    "JPM":    "Quarterly",
    "JNJ":    "Quarterly",
    "XOM":    "Quarterly",
    "UNH":    "Quarterly",
    "HD":     "Quarterly",
    "BRK-B":  "None",
    "SPY":    "Quarterly",
    "QQQ":    "Quarterly",
    "AGG":    "Monthly",
    "UK_GILT_10Y":  "Semi-Annual",
    "US_TREAS_10Y": "Semi-Annual",
    "CASH_GBP":     "Monthly",
}

# ---------------------------------------------------------------------------
# Preset stress-test scenarios
# ---------------------------------------------------------------------------
STRESS_SCENARIOS = {
    "Equity Market -10%": {
        "description": "Broad equity market decline of 10%.",
        "shocks_by_asset_class": {"Equity": -0.10, "ETF": -0.08},
        "shocks_by_sector": {},
    },
    "Technology Sector -20%": {
        "description": "Technology sector sell-off of 20%.",
        "shocks_by_asset_class": {"Equity": -0.05},
        "shocks_by_sector": {"Technology": -0.20},
    },
    "Interest Rate Rise +200bps": {
        "description": "200bp rate hike: bonds fall ~8%, equities fall ~5%.",
        "shocks_by_asset_class": {"Equity": -0.05, "ETF": -0.04, "Bond": -0.08},
        "shocks_by_sector": {"Fixed Income": -0.08, "Government Bond": -0.08},
    },
    "Inflation Shock": {
        "description": "High inflation: energy +15%, bonds -12%, real estate -10%.",
        "shocks_by_asset_class": {"Bond": -0.12},
        "shocks_by_sector": {"Energy": +0.15, "Fixed Income": -0.12, "Government Bond": -0.12},
    },
    "2008 GFC Replay": {
        "description": "Global financial crisis: equities -35%, bonds +5% (flight to quality).",
        "shocks_by_asset_class": {"Equity": -0.35, "ETF": -0.30, "Bond": +0.05},
        "shocks_by_sector": {"Financials": -0.50, "Technology": -0.40},
    },
    "COVID-19 Crash Replay": {
        "description": "COVID crash: equities -30%, then partial recovery.",
        "shocks_by_asset_class": {"Equity": -0.30, "ETF": -0.28, "Bond": +0.03},
        "shocks_by_sector": {},
    },
}

# ---------------------------------------------------------------------------
# Colour palettes
# ---------------------------------------------------------------------------
ASSET_CLASS_COLORS = {
    "Equity": "#00D4FF",
    "ETF":    "#7B68EE",
    "Bond":   "#32CD32",
    "Cash":   "#FFD700",
}

SECTOR_COLORS = {
    "Technology":      "#00D4FF",
    "Financials":      "#7B68EE",
    "Healthcare":      "#32CD32",
    "Energy":          "#FF6B35",
    "Consumer Disc.":  "#FFD700",
    "Broad Market":    "#FF69B4",
    "Fixed Income":    "#98FB98",
    "Government Bond": "#90EE90",
    "Cash":            "#F0E68C",
}

CHART_COLOR_SEQUENCE = [
    "#00D4FF", "#7B68EE", "#32CD32", "#FF6B35",
    "#FFD700", "#FF69B4", "#98FB98", "#FFA07A",
    "#87CEEB", "#DDA0DD", "#20B2AA", "#F0E68C",
]

# IBKR Portfolio Tracker

Portfolio analysis and tracking tool for Interactive Brokers account data with automated analytics, performance metrics, visualizations, and rebalancing recommendations.

## Project Structure

```
IBKR_portfolio_tracker/
├── data/                   # CSV data files from IBKR (gitignored)
├── src/                    # Core library modules
│   ├── data_loader.py      # Load and parse IBKR transactions and positions
│   ├── metrics.py          # Calculate portfolio performance metrics
│   ├── visualization.py    # Generate charts and summary reports
│   └── rebalancing.py      # Calculate rebalancing trades
└── jupyter/                # Analysis notebooks
    ├── portfolio_analysis.ipynb          # Main automated analysis notebook
    └── verify_calculations.ipynb         # Verification and testing notebook
```

## Installation

```bash
uv pip install polars yfinance matplotlib numpy
```

## Usage

### 1. Prepare Your Data

Export your IBKR account data as CSV and place files in the `data/` directory:
```
data/UXXXXXXXX_YYYYMMDD_YYYYMMDD.csv
```

You can have multiple CSV files covering different time periods (e.g., multiple years). The tool will process all CSV files in the `data/` directory.

### 2. Run Analysis

Open `jupyter/portfolio_analysis.ipynb` and run all cells to generate portfolio metrics, performance charts, and rebalancing recommendations.

## Key Metrics

- **Simple Return**: `(Current Equity / Total Contributed) - 1`
- **TWR (time-weighted)**: `Π(1 + r_t) - 1`, with daily `r_t = (E_t - Flow_t) / E_{t-1} - 1`, so deposit timing doesn't affect it
- **Annualized TWR**: `[(1 + TWR)^(1/years) - 1] × 100`
- **XIRR (money-weighted)**: annual rate that discounts all contributions to the current value; reported in USD and JPY
- **Holding Ret%**: `(Unrealized + Net Dividends) / Cost Basis × 100`, current holdings only

Prices come from Yahoo Finance daily closes that are split-adjusted but not dividend-adjusted (`auto_adjust=False`); dividends are booked as cash. Statement timestamps are converted from `Asia/Tokyo` to New York trade dates (`STATEMENT_TZ` in `src/data_loader.py`).

## Disclaimer

For informational purposes only. Not financial advice.

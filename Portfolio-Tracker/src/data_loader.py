"""
Data loading utilities for portfolio transactions and positions.
"""

import os
import re
import polars as pl
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# Get the project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# IBKR statements print trade timestamps in the account's timezone; trades are dated by the exchange's session
STATEMENT_TZ = ZoneInfo("Asia/Tokyo")
EXCHANGE_TZ = ZoneInfo("America/New_York")


def get_csv_files():
    """Get all CSV files from the data directory matching the date pattern."""
    files = [str(DATA_DIR / f) for f in os.listdir(DATA_DIR)
             if re.search(r"\d{8}_\d{8}\.csv$", f)]
    return sorted(files, key=lambda p: re.search(r"(\d{8})_(\d{8})", os.path.basename(p)).groups())


def clean(val):
    """Clean numeric values from CSV files."""
    if val is None or str(val) == '--':
        return 0.0
    return float(str(val).replace(",", ""))


def extract_ticker(desc):
    """Extract ticker symbol from description string."""
    match = re.match(r"([A-Z]+)\(", str(desc))
    return match.group(1) if match else None


def trade_date(timestamp):
    """Convert an IBKR 'YYYY-MM-DD, HH:MM:SS' statement timestamp to the exchange trade date."""
    dt = datetime.strptime(str(timestamp), "%Y-%m-%d, %H:%M:%S").replace(tzinfo=STATEMENT_TZ)
    return dt.astimezone(EXCHANGE_TZ).strftime("%Y-%m-%d")


def not_total(col):
    """Filter out the 'Total' / 'Total in JPY' summary rows of a section."""
    return pl.col(col).is_not_null() & ~pl.col(col).str.starts_with("Total")


def read_csv(filepath):
    """Read a CSV file with error handling."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pl.read_csv(filepath, has_header=False, infer_schema_length=1000,
                          ignore_errors=True, null_values=["--", ""], truncate_ragged_lines=True)


def load_transactions():
    """
    Load all transactions from CSV files.

    Returns:
        pl.DataFrame: DataFrame containing all transactions with columns:
            - date: Transaction date
            - type: Transaction type (Trade, FX, Dividend, Tax, Deposit, Fee)
            - symbol: Stock symbol (if applicable)
            - quantity: Number of shares
            - amount: Transaction amount
            - commission: Commission paid
            - currency: Currency of transaction
    """
    transactions = []

    for filepath in get_csv_files():
        df = read_csv(filepath)

        # Trades
        for r in df.filter((pl.col("column_1") == "Trades") & (pl.col("column_2") == "Data") &
                          (pl.col("column_3") == "Order") & (pl.col("column_4") == "Stocks")).iter_rows():
            transactions.append({'date': trade_date(r[6]), 'type': 'Trade', 'symbol': str(r[5]),
                               'quantity': clean(r[7]), 'amount': clean(r[10]),
                               'commission': abs(clean(r[11])), 'currency': str(r[4])})

        # Forex
        for r in df.filter((pl.col("column_1") == "Trades") & (pl.col("column_2") == "Data") &
                          (pl.col("column_3") == "Order") & (pl.col("column_4") == "Forex")).iter_rows():
            transactions.append({'date': trade_date(r[6]), 'type': 'FX', 'symbol': str(r[5]),
                               'quantity': clean(r[7]), 'amount': clean(r[10]), 'commission': 0.0, 'currency': str(r[4])})

        # Dividends
        for r in df.filter((pl.col("column_1") == "Dividends") & (pl.col("column_2") == "Data") &
                          not_total("column_3")).iter_rows():
            transactions.append({'date': str(r[3]), 'type': 'Dividend', 'symbol': extract_ticker(r[4]),
                               'quantity': 0.0, 'amount': clean(r[5]), 'commission': 0.0, 'currency': str(r[2])})

        # Withholding Tax
        for r in df.filter((pl.col("column_1") == "Withholding Tax") & (pl.col("column_2") == "Data") &
                          not_total("column_3")).iter_rows():
            transactions.append({'date': str(r[3]), 'type': 'Tax', 'symbol': extract_ticker(r[4]),
                               'quantity': 0.0, 'amount': clean(r[5]), 'commission': 0.0, 'currency': str(r[2])})

        # Deposits
        for r in df.filter((pl.col("column_1") == "Deposits & Withdrawals") & (pl.col("column_2") == "Data") &
                          not_total("column_3")).iter_rows():
            transactions.append({'date': str(r[3]), 'type': 'Deposit', 'symbol': None,
                               'quantity': 0.0, 'amount': clean(r[5]), 'commission': 0.0, 'currency': str(r[2])})

        # Fees
        for r in df.filter((pl.col("column_1") == "Fees") & (pl.col("column_2") == "Data") &
                          (pl.col("column_3") == "Other Fees")).iter_rows():
            transactions.append({'date': str(r[4]) if len(r) > 4 else None, 'type': 'Fee', 'symbol': None,
                               'quantity': 0.0, 'amount': clean(r[6]) if len(r) > 6 else 0.0,
                               'commission': 0.0, 'currency': str(r[3]) if len(r) > 3 else 'USD'})

    df = pl.DataFrame(transactions)
    df = df.with_columns(pl.col('date').str.strptime(pl.Date, '%Y-%m-%d', strict=False))
    return df.filter(pl.col('date').is_not_null()).sort('date')


def load_positions():
    """
    Load current positions from the most recent CSV file.

    Returns:
        pl.DataFrame: DataFrame containing current positions with columns:
            - symbol: Stock symbol
            - shares: Number of shares held (fractional shares preserved)
            - cost_basis: Total cost basis
            - current_price: Statement close price
            - market_value: Statement market value
            - unrealized_pnl: Unrealized profit/loss
            - as_of: Statement end date the prices refer to
    """
    for filepath in reversed(get_csv_files()):
        df = read_csv(filepath)
        rows = df.filter((pl.col("column_1") == "Open Positions") & (pl.col("column_2") == "Data") &
                        (pl.col("column_3") == "Summary"))

        if rows.height > 0:
            as_of = datetime.strptime(re.search(r"_(\d{8})\.csv$", filepath).group(1), "%Y%m%d").date()
            return pl.DataFrame([{'symbol': str(r[5]), 'shares': clean(r[6]),
                                 'cost_basis': clean(r[9]), 'current_price': clean(r[10]),
                                 'market_value': clean(r[11]), 'unrealized_pnl': clean(r[12]),
                                 'as_of': as_of}
                                for r in rows.iter_rows()])
    return pl.DataFrame()

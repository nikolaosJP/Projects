"""
Portfolio metrics calculation utilities.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import polars as pl

FX_TICKER = 'JPY=X'  # Yahoo Finance USD/JPY (yen per dollar)


def download_closes(tickers, start, end):
    """
    Download daily closes and forward-fill them onto a calendar-day index.

    Uses auto_adjust=False so 'Close' is split-adjusted but NOT dividend-adjusted:
    dividends are booked as cash separately, so adjusted prices would count them twice.
    """
    data = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=False)['Close']
    if isinstance(data, pd.Series):
        data = data.to_frame(tickers[0])
    data.index = pd.to_datetime(data.index).tz_localize(None).normalize()
    return data.reindex(pd.date_range(start, end, freq='D')).ffill()


def xirr(cash_flows, lo=-0.99, hi=10.0, iterations=200):
    """
    Annualized money-weighted return (IRR) of dated cash flows, found by bisection.

    Args:
        cash_flows (list): (date, amount) pairs; contributions negative, final value positive

    Returns:
        float: Annual rate (0.17 = 17%), or None if there is no sign change to solve for
    """
    t0 = cash_flows[0][0]

    def npv(rate):
        return sum(v / (1 + rate) ** ((d - t0).days / 365.25) for d, v in cash_flows)

    if npv(lo) * npv(hi) > 0:
        return None
    for _ in range(iterations):
        mid = (lo + hi) / 2
        if npv(lo) * npv(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def money_weighted_return(portfolio_history, value_col='equity', invested_col='contributed'):
    """
    XIRR from a portfolio history: daily changes in `invested_col` are the external flows,
    the last `value_col` is the terminal value.
    """
    dates = portfolio_history['date'].to_list()
    invested = portfolio_history[invested_col].to_list()
    flows = [(dates[0], -invested[0])]
    flows += [(d, -(c - prev)) for d, c, prev in zip(dates[1:], invested[1:], invested[:-1]) if abs(c - prev) > 1e-9]
    flows.append((dates[-1], portfolio_history[value_col][-1]))
    return xirr(flows)


def calculate_portfolio_metrics(transactions):
    """
    Calculate portfolio metrics over time from transaction history.

    Args:
        transactions (pl.DataFrame): DataFrame of transactions from load_transactions()

    Returns:
        pl.DataFrame: Daily portfolio history with columns including:
            - date: Date
            - contributed: Total capital contributed (USD deposits + JPY deposits at the deposit-day market rate)
            - invested_capital: Current cost basis of holdings
            - cash: Available cash
            - equity: Total portfolio value
            - holdings_value: Market value of holdings
            - unrealized_pnl: Unrealized profit/loss
            - realized_pnl: Realized profit/loss
            - total_investment_pnl: Total investment P&L
            - portfolio_pnl: Total portfolio P&L
            - simple_return_pct: Simple return on contributed capital = (Equity/Contributed - 1) × 100
            - return_pct: Alias of simple_return_pct (kept for backwards compatibility)
            - twr_pct: Time-weighted return, daily returns chain-linked with contributions removed
            - dividends: Cumulative gross dividends
            - withholding_tax: Cumulative withholding tax (positive = cost)
            - fees: Cumulative account fees (positive = cost)
            - fx_pnl: FX gain/loss on deposited yen: USD obtained from conversions plus unconverted
              JPY at today's rate, minus the deposits' value on their deposit days
            - fx_rate: USD/JPY market rate (yen per dollar)
            - deposited_jpy: Cumulative deposits in JPY
            - equity_jpy: Total portfolio value in JPY
            - {symbol}_shares: Shares held for each symbol
            - {symbol}_value: Market value for each symbol
    """
    # Get symbols and download prices
    all_symbols = transactions.filter(
        (pl.col('type') == 'Trade') & (pl.col('symbol').is_not_null())
    )['symbol'].unique().to_list()

    start, today = transactions['date'].min(), datetime.now().date()
    price_data = download_closes(all_symbols + [FX_TICKER], start=start, end=today)

    # Fall back to the rates of executed FX trades if the market rate is unavailable
    fx_trades = transactions.filter((pl.col('type') == 'FX') & (pl.col('symbol') == 'USD.JPY') & (pl.col('quantity') != 0))
    trade_rates = pd.Series([abs(a) / abs(q) for a, q in zip(fx_trades['amount'], fx_trades['quantity'])],
                            index=pd.to_datetime(fx_trades['date'].to_list()))
    trade_rates = trade_rates.groupby(level=0).last().reindex(price_data.index).ffill()
    market_rates = price_data[FX_TICKER] if FX_TICKER in price_data.columns else pd.Series(index=price_data.index)
    fx_rates = market_rates.fillna(trade_rates).bfill()

    # Initialize
    contributed = jpy_cash = usd_cash = dividends = realized_pnl = deposited_jpy = 0.0
    taxes = fees = usd_from_fx = jpy_deposits_usd = jpy_other = 0.0
    twr_index = 1.0
    prev_equity = prev_contributed = 0.0
    holdings = {}
    current_holdings_cost = {}
    daily_values = []

    def add_cash(currency, amount, fx_rate):
        """Book a non-deposit cash flow; returns its USD value."""
        nonlocal jpy_cash, usd_cash, jpy_other
        if currency == 'JPY':
            jpy_cash += amount
            jpy_other += amount
            return amount / fx_rate
        usd_cash += amount
        return amount

    # Generate dates and process
    current = start
    while current <= today:
        fx_rate = float(fx_rates.loc[pd.Timestamp(current)])

        # Process day's transactions
        for row in transactions.filter(pl.col('date') == current).iter_rows(named=True):
            if row['type'] == 'Deposit':
                # Deposits are the only external flows; JPY is valued once, at the deposit-day rate
                if row['currency'] == 'JPY':
                    jpy_cash += row['amount']
                    deposited_jpy += row['amount']
                    jpy_deposits_usd += row['amount'] / fx_rate
                    contributed += row['amount'] / fx_rate
                else:
                    usd_cash += row['amount']
                    deposited_jpy += row['amount'] * fx_rate
                    contributed += row['amount']
            elif row['type'] == 'FX':
                # Converting JPY to USD moves money inside the account; it is not a contribution
                usd_from_fx += row['quantity']
                usd_cash += row['quantity']
                jpy_cash += row['amount']
            elif row['type'] == 'Trade' and row['symbol']:
                sym, qty = row['symbol'], row['quantity']
                cost = abs(row['amount']) + abs(row['commission'])
                usd_cash += row['amount'] - row['commission']

                if qty > 0:  # Buy
                    current_holdings_cost[sym] = current_holdings_cost.get(sym, 0) + cost
                else:  # Sell
                    if sym in current_holdings_cost and holdings.get(sym, 0) > 0:
                        sold_cost = (current_holdings_cost[sym] / holdings[sym]) * abs(qty)
                        realized_pnl += (row['amount'] - row['commission']) - sold_cost
                        current_holdings_cost[sym] -= sold_cost

                holdings[sym] = holdings.get(sym, 0) + qty
                if holdings[sym] == 0:
                    holdings.pop(sym, None)
                    current_holdings_cost.pop(sym, None)
            elif row['type'] == 'Dividend':
                dividends += add_cash(row['currency'], row['amount'], fx_rate)
            elif row['type'] == 'Tax':
                taxes -= add_cash(row['currency'], row['amount'], fx_rate)
            elif row['type'] == 'Fee':
                fees -= add_cash(row['currency'], row['amount'], fx_rate)

        # Calculate holdings value per symbol and store holdings + values
        holdings_value = 0.0
        day_data = {'date': current}

        for sym in all_symbols:
            shares = holdings.get(sym, 0)
            sym_value = 0.0
            if shares > 0 and sym in price_data.columns:
                price = price_data.at[pd.Timestamp(current), sym]
                if pd.notna(price):
                    sym_value = shares * price
                    holdings_value += sym_value
            day_data[f'{sym}_shares'] = shares
            day_data[f'{sym}_value'] = sym_value

        total_cash = (jpy_cash / fx_rate) + usd_cash
        invested_capital = sum(current_holdings_cost.values())
        unrealized_pnl = holdings_value - invested_capital
        equity = total_cash + holdings_value

        fx_pnl = usd_from_fx + (jpy_cash - jpy_other) / fx_rate - jpy_deposits_usd

        # Time-weighted return: today's change in contributed capital is an external flow, not performance
        if prev_equity > 0:
            twr_index *= (equity - (contributed - prev_contributed)) / prev_equity
        prev_equity, prev_contributed = equity, contributed

        simple_return_pct = ((equity / contributed - 1) * 100) if contributed > 0 else 0

        day_data.update({
            'contributed': contributed, 'invested_capital': invested_capital,
            'cash': total_cash, 'equity': equity, 'holdings_value': holdings_value,
            'unrealized_pnl': unrealized_pnl, 'realized_pnl': realized_pnl,
            'total_investment_pnl': unrealized_pnl + realized_pnl,
            'portfolio_pnl': equity - contributed,
            'simple_return_pct': simple_return_pct,
            'return_pct': simple_return_pct,
            'twr_pct': (twr_index - 1) * 100,
            'dividends': dividends, 'withholding_tax': taxes, 'fees': fees, 'fx_pnl': fx_pnl,
            'fx_rate': fx_rate, 'deposited_jpy': deposited_jpy, 'equity_jpy': equity * fx_rate
        })

        daily_values.append(day_data)
        current += timedelta(days=1)

    return pl.DataFrame(daily_values)

"""
Portfolio visualization and reporting utilities.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import polars as pl

try:
    from .metrics import money_weighted_return
except ImportError:  # imported as a top-level module (notebooks add src/ to sys.path)
    from metrics import money_weighted_return


def plot_portfolio_performance(portfolio_history, transactions):
    """
    Plot portfolio performance with value, P&L, and asset allocation charts.

    Args:
        portfolio_history (pl.DataFrame): DataFrame from calculate_portfolio_metrics()
        transactions (pl.DataFrame): DataFrame from load_transactions()
    """
    # Extract data
    dates = portfolio_history['date'].to_list()

    # Get asset symbols from column names - EXCLUDE 'holdings_value'
    all_symbols = [col.replace('_value', '') for col in portfolio_history.columns
                   if col.endswith('_value') and col != 'holdings_value']

    data = {k: portfolio_history[k].to_list() for k in
            ['equity', 'contributed', 'invested_capital', 'cash', 'twr_pct', 'portfolio_pnl', 'dividends']}

    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True,
                                         gridspec_kw={"height_ratios": [2, 1, 1.5]})
    fig.suptitle("Portfolio Performance (Daily Close Prices)", fontsize=14, fontweight='bold')

    # Subplot 1: Portfolio value
    ax1.step(dates, data['equity'], 'b-', label='Total Value', linewidth=2, where='post')
    ax1.step(dates, data['contributed'], 'grey', linestyle='--', label='Contributed', linewidth=1.5, where='post')
    ax1.step(dates, data['invested_capital'], 'purple', linestyle='--', label='Invested Capital',
             linewidth=1.5, alpha=0.7, where='post')
    ax1.step(dates, data['cash'], ':', label='Cash', color='black', linewidth=1.5, alpha=0.7, where='post')

    eq, contrib = np.array(data['equity']), np.array(data['contributed'])
    ax1.fill_between(dates, contrib, eq, where=eq >= contrib, alpha=0.2, color='green', step='post')
    ax1.fill_between(dates, contrib, eq, where=eq < contrib, alpha=0.2, color='red', step='post')

    # Buy/sell markers
    for row in transactions.filter(pl.col('type') == 'Trade').iter_rows(named=True):
        if row['date'] in dates:
            idx = dates.index(row['date'])
            marker, color = ('^', 'green') if row['quantity'] > 0 else ('v', 'red')
            ax1.scatter(row['date'], data['equity'][idx], marker=marker, s=100, color=color,
                       edgecolors='dark'+color, linewidths=1.5, zorder=5, alpha=0.8)

    ax1.set_ylabel('Value (USD)', fontsize=11)
    ax1.grid(alpha=0.3, linestyle='--')

    # Secondary axis for time-weighted return %
    ax1b = ax1.twinx()
    ax1b.step(dates, data['twr_pct'], '--', color='green', label='Time-Weighted Return %', linewidth=2, alpha=0.7, where='post')
    ax1b.set_ylabel('Time-Weighted Return % (TWR)', fontsize=11, color='green')
    ax1b.tick_params(axis='y', labelcolor='green')

    # Legend with buy/sell markers
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines1b, labels1b = ax1b.get_legend_handles_labels()
    markers = [plt.Line2D([0], [0], marker=m, color='w', markerfacecolor=c,
                          markeredgecolor='dark'+c, markersize=8, label=l)
               for m, c, l in [('^', 'green', 'Buy'), ('v', 'red', 'Sell')]]
    ax1.legend(lines1 + lines1b + markers, labels1 + labels1b + ['Buy', 'Sell'],
               loc='upper left', fontsize=9)

    # Subplot 2: P&L and Dividends
    ax2.step(dates, data['portfolio_pnl'], 'purple', label='Portfolio P&L', linewidth=2, where='post')
    ax2.axhline(0, color='black', linewidth=1, alpha=0.5)
    pnl = np.array(data['portfolio_pnl'])
    ax2.fill_between(dates, 0, pnl, where=pnl >= 0, alpha=0.3, color='green', step='post')
    ax2.fill_between(dates, 0, pnl, where=pnl < 0, alpha=0.3, color='red', step='post')
    ax2.set_ylabel('P&L (USD)', fontsize=11)
    ax2.grid(alpha=0.3, linestyle='--')

    ax2b = ax2.twinx()
    ax2b.step(dates, data['dividends'], ':', color='orange', label='Cumulative Dividends (gross)',
             linewidth=2, alpha=0.8, where='post')
    ax2b.set_ylabel('Cumulative Gross Dividends (USD)', fontsize=11, color='orange')
    ax2b.tick_params(axis='y', labelcolor='orange')

    lines2, labels2 = ax2.get_legend_handles_labels()
    lines2b, labels2b = ax2b.get_legend_handles_labels()
    ax2.legend(lines2 + lines2b, labels2 + labels2b, loc='upper left', fontsize=9)

    # Subplot 3: Asset allocation
    asset_values_over_time = {sym: portfolio_history[f'{sym}_value'].to_list() for sym in all_symbols}
    cash_over_time = data['cash']

    final_values = {sym: asset_values_over_time[sym][-1] for sym in all_symbols}
    final_values['Cash'] = cash_over_time[-1]
    sorted_symbols = sorted(final_values.keys(), key=lambda s: final_values[s], reverse=True)

    stack_data = [np.array(cash_over_time if sym == 'Cash' else asset_values_over_time[sym])
                  for sym in sorted_symbols]

    ax3.stackplot(dates, *stack_data, labels=sorted_symbols,
                 colors=plt.cm.tab10(range(len(sorted_symbols))), alpha=0.8)
    ax3.set_ylabel('Value (USD)', fontsize=11)
    ax3.set_xlabel('Date', fontsize=11)
    ax3.grid(alpha=0.3, linestyle='--')
    ax3.legend(loc='upper left', fontsize=9, ncol=min(len(sorted_symbols), 4))

    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax3.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.show()


def print_portfolio_summary(transactions, positions, portfolio_history, mode='full'):
    """
    Print comprehensive portfolio summary with mathematical formulas for each metric.

    Args:
        transactions (pl.DataFrame): DataFrame from load_transactions()
        positions (pl.DataFrame): DataFrame from load_positions()
        portfolio_history (pl.DataFrame): DataFrame from calculate_portfolio_metrics()
        mode (str): 'full' for the since-inception summary, 'annual' for a breakdown by
            calendar year (Jan 1 - Dec 31) with a since-inception 'Total' column
    """
    if mode == 'annual':
        return _print_annual_summary(transactions, portfolio_history)
    if mode != 'full':
        raise ValueError(f"mode must be 'full' or 'annual', got {mode!r}")

    def fmt_num(val, decimal=2, prefix='', na_str='N/A'):
        """Helper to format numeric values, handling None gracefully."""
        if val is None:
            return na_str
        return f"{prefix}{val:,.{decimal}f}"

    def fmt_pct(val, decimal=2, na_str='N/A'):
        """Helper to format percentage values, handling None gracefully."""
        if val is None:
            return na_str
        return f"{val:.{decimal}f}%"

    def by_type(kind):
        return transactions.filter(pl.col('type') == kind)

    final = portfolio_history.row(-1, named=True)
    fx_rate = final['fx_rate']

    # Aggregate dividends and taxes by symbol (per-holding table only)
    div_agg = by_type('Dividend').group_by('symbol').agg(pl.col('amount').sum().alias('total_dividends'))
    tax_agg = by_type('Tax').group_by('symbol').agg(pl.col('amount').sum().alias('withholding_tax'))

    # Table weights use the statement's market values plus current cash
    holdings_mv = positions['market_value'].sum() or 0
    table_total = holdings_mv + final['cash']

    # Build enriched holdings summary
    summary = (positions
        .join(div_agg, on='symbol', how='left')
        .join(tax_agg, on='symbol', how='left')
        .with_columns([
            pl.col('total_dividends').fill_null(0),
            pl.col('withholding_tax').fill_null(0),
            (pl.col('cost_basis') / pl.col('shares')).alias('avg_price'),
            (pl.col('market_value') / table_total * 100).alias('portfolio_weight_pct'),
            (pl.col('market_value') / holdings_mv * 100).alias('asset_weight_pct'),
            (pl.col('total_dividends') + pl.col('withholding_tax')).alias('net_dividends'),
            ((pl.col('unrealized_pnl') + pl.col('total_dividends') + pl.col('withholding_tax')) /
             pl.col('cost_basis') * 100).alias('total_return_pct')
        ])
    )

    # Account-wide totals cover all transactions, including closed positions
    total_commissions = by_type('Trade')['commission'].sum()
    gross_dividends = final['dividends']
    total_withholding_tax = final['withholding_tax']
    total_fees = final['fees']
    fx_pnl = final['fx_pnl']

    # Returns
    start_date = portfolio_history['date'].min()
    end_date = portfolio_history['date'].max()
    days_invested = (end_date - start_date).days
    years_invested = days_invested / 365.25

    def annualize(total_pct):
        return ((1 + total_pct / 100) ** (1 / years_invested) - 1) * 100 if years_invested > 0 else 0

    twr_annualized = annualize(final['twr_pct'])
    usd_xirr = money_weighted_return(portfolio_history, 'equity', 'contributed')
    jpy_xirr = money_weighted_return(portfolio_history, 'equity_jpy', 'deposited_jpy')
    pct_or_na = lambda rate: fmt_pct(rate * 100 if rate is not None else None)

    cash_weight = (final['cash'] / final['equity'] * 100) if final['equity'] else 0
    as_of = positions['as_of'][0] if 'as_of' in positions.columns and positions.height else None

    print("\n" + "=" * 120)
    print("INVESTMENT TIMELINE")
    print("=" * 120)
    print(f"Start Date:                              {start_date}")
    print(f"End Date:                                {end_date}")
    print(f"Days Invested:                           {days_invested} days ({years_invested:.2f} years)")
    print(f"First Deposit:                           {by_type('Deposit')['date'].min()}")
    print(f"Number of Deposits:                      {by_type('Deposit').height}")

    print("\n" + "=" * 120)
    print("ASSET ALLOCATION & PERFORMANCE")
    print("=" * 120)
    if as_of:
        print(f"Prices from IBKR statement as of {as_of}; cash from transaction history.")
    print(f"\n{'Symbol':<8} {'Shares':<10} {'Avg$':<10} {'Curr$':<10} {'Cost Basis':<13} {'Market Val':<13} "
          f"{'Port%':<9} {'Asset%':<9} {'Unrealized':<13} {'Net Divs':<10} {'Ret%':<8}")
    print("-" * 120)

    for row in summary.sort('portfolio_weight_pct', descending=True).iter_rows(named=True):
        print(f"{row['symbol']:<8} "
              f"{fmt_num(row['shares'], 2, ''):<10} "
              f"{fmt_num(row['avg_price'], 2, '$'):<10} "
              f"{fmt_num(row['current_price'], 2, '$'):<10} "
              f"{fmt_num(row['cost_basis'], 2, '$'):<13} "
              f"{fmt_num(row['market_value'], 2, '$'):<13} "
              f"{fmt_pct(row['portfolio_weight_pct'], 2):<9} "
              f"{fmt_pct(row['asset_weight_pct'], 2):<9} "
              f"{fmt_num(row['unrealized_pnl'], 2, '$'):<13} "
              f"{fmt_num(row['net_dividends'], 2, '$'):<10} "
              f"{fmt_pct(row['total_return_pct'], 2):<8}")

    # Add cash row
    print(f"{'Cash':<8} {'-':<10} {'-':<10} {'-':<10} {'-':<13} {fmt_num(final['cash'], 2, '$'):<13} "
          f"{fmt_pct(final['cash'] / table_total * 100):<9} {'-':<9} {'-':<13} {'-':<10} {'-':<8}")

    print("-" * 120)
    cost_basis_sum = summary['cost_basis'].sum() or 0
    table_net_dividends = summary['net_dividends'].sum()
    total_asset_return = ((summary['unrealized_pnl'].sum() + table_net_dividends) / cost_basis_sum * 100) if cost_basis_sum else 0
    print(f"{'TOTAL':<8} {'':<10} {'':<10} {'':<10} {fmt_num(cost_basis_sum, 2, '$'):<13} "
          f"{fmt_num(table_total, 2, '$'):<13} {fmt_pct(100):<9} {fmt_pct(100):<9} "
          f"{fmt_num(summary['unrealized_pnl'].sum(), 2, '$'):<13} {fmt_num(table_net_dividends, 2, '$'):<10} "
          f"{fmt_pct(total_asset_return):<8}")

    print(f"\nNote: 'Ret%' = (Unrealized P&L + Net Dividends) / Cost Basis × 100")
    print(f"      'Port%' = share of holdings + cash, 'Asset%' = share of holdings only.")
    print(f"      Returns on current holdings only, excluding realized gains from closed positions.")

    print("\n" + "=" * 120)
    print("PORTFOLIO CAPITAL STRUCTURE")
    print("=" * 120)

    # Deposits are valued on their deposit day; what the yen actually became in USD differs by FX P&L
    total_jpy_deposited = by_type('Deposit').filter(pl.col('currency') == 'JPY')['amount'].sum()
    if total_jpy_deposited > 0:
        usd_obtained = final['contributed'] + fx_pnl
        jpy_value_today = total_jpy_deposited / fx_rate
        vs_today = usd_obtained - jpy_value_today

        print(f"JPY Deposits Total:                      ¥{total_jpy_deposited:>12,.0f}")
        print(f"  Value on deposit days:                 ${final['contributed']:>12,.2f}  = Σ(Deposit / USDJPY on deposit day)")
        print(f"  USD obtained (converted + yen left):   ${usd_obtained:>12,.2f}  = Actual conversions + unconverted yen today")
        print(f"  FX Gain/Loss (FX):                     ${fx_pnl:>12,.2f}  = USD obtained - Value on deposit days")
        print(f"  Value if converted today (¥{fx_rate:.2f}): ${jpy_value_today:>12,.2f}  [{'gained' if vs_today > 0 else 'lost'} "
              f"${abs(vs_today):,.2f} vs. converting today]")
    print(f"{'-' * 120}")
    print(f"Contributed Capital (C):                 ${final['contributed']:>12,.2f}  = Σ(Deposits valued on deposit day)")
    print(f"Invested Capital (I):                    ${final['invested_capital']:>12,.2f}  = Σ(Cost Basis inc. buy commissions)")
    print(f"Cash (K):                                ${final['cash']:>12,.2f}  = C - I + R_cap + D_gross - T_div - T_fees + FX")
    print(f"Holdings Market Value (H):               ${final['holdings_value']:>12,.2f}  = Σ(Shares × Current Price)")
    print(f"{'-' * 120}")
    print(f"Total Equity (E):                        ${final['equity']:>12,.2f}  = K + H")
    print(f"Portfolio P&L:                           ${final['portfolio_pnl']:>12,.2f}  = E - C")

    print("\n" + "=" * 120)
    print("REALIZED INCOME")
    print("=" * 120)
    print(f"Realized Capital Gains (R_cap):          ${final['realized_pnl']:>12,.2f}  = Σ[(Sale Price - Sell Commission) - Cost Basis]")
    print(f"Gross Dividends (D_gross):               ${gross_dividends:>12,.2f}  = Σ(Dividend Payments)")
    print(f"{'-' * 120}")
    print(f"Total Realized Income (before costs):   ${final['realized_pnl'] + gross_dividends:>12,.2f}  = R_cap + D_gross")

    print("\n" + "=" * 120)
    print("UNREALIZED GAINS (Open Positions)")
    print("=" * 120)
    print(f"Unrealized P&L (U):                      ${final['unrealized_pnl']:>12,.2f}  = H - I")

    print("\n" + "=" * 120)
    print("COSTS & TAXES")
    print("=" * 120)
    print(f"Trading Commissions:                     ${total_commissions:>12,.2f}  [Already included in Invested Capital & Realized Gains]")
    print(f"Dividend Withholding Tax (T_div):        ${total_withholding_tax:>12,.2f}  = Σ(Tax Withheld on Dividends)")
    print(f"Account Fees (T_fees):                   ${total_fees:>12,.2f}  = Σ(Other Account Fees)")
    print(f"{'-' * 120}")
    print(f"Costs to subtract (T_div + T_fees):      ${total_withholding_tax + total_fees:>12,.2f}  = T_div + T_fees")

    print("\n" + "=" * 120)
    print("TOTAL INVESTMENT PERFORMANCE")
    print("=" * 120)

    net_investment_pnl = (final['unrealized_pnl'] + final['realized_pnl'] + gross_dividends
                          - total_withholding_tax - total_fees + fx_pnl)
    reconciliation_gap = net_investment_pnl - final['portfolio_pnl']

    print(f"Unrealized Gains:                        ${final['unrealized_pnl']:>12,.2f}")
    print(f"Realized Gains (net of commissions):     ${final['realized_pnl']:>12,.2f}")
    print(f"Gross Dividends:                         ${gross_dividends:>12,.2f}")
    print(f"Less: Withholding Tax & Fees:            ${total_withholding_tax + total_fees:>12,.2f}")
    print(f"FX Gain/Loss:                            ${fx_pnl:>12,.2f}")
    print(f"{'-' * 120}")
    print(f"Net Investment P&L:                      ${net_investment_pnl:>12,.2f}  = U + R_cap + D_gross - T_div - T_fees + FX")
    status = "OK" if abs(reconciliation_gap) < 0.01 else "MISMATCH - check transactions"
    print(f"Reconciliation vs Portfolio P&L (E - C): ${reconciliation_gap:>12,.2f}  [{status}]")
    print(f"{'-' * 120}")
    print(f"Simple Return on Contributed:            {final['simple_return_pct']:>11,.2f}%  = (E / C - 1) × 100")
    print(f"Time-Weighted Return (TWR):              {final['twr_pct']:>11,.2f}%  = Π(1 + daily return) - 1, deposits removed")
    print(f"Annualized TWR:                          {twr_annualized:>11,.2f}%  = [(1 + TWR)^(1/years) - 1] × 100")
    print(f"Money-Weighted Return (XIRR, USD):       {pct_or_na(usd_xirr):>12}  = annual rate on your actual deposit timing")

    print("\n" + "=" * 120)
    print("JPY PERSPECTIVE")
    print("=" * 120)
    print(f"USD/JPY (latest market close):           ¥{fx_rate:>12,.2f}")
    print(f"Total Deposited (JPY):                   ¥{final['deposited_jpy']:>12,.0f}")
    print(f"Portfolio Value (JPY):                   ¥{final['equity_jpy']:>12,.0f}  = E × USD/JPY")
    print(f"Portfolio P&L (JPY):                     ¥{final['equity_jpy'] - final['deposited_jpy']:>12,.0f}")
    jpy_return = (final['equity_jpy'] / final['deposited_jpy'] - 1) * 100 if final['deposited_jpy'] else None
    print(f"Simple Return (JPY):                     {fmt_pct(jpy_return):>12}  = (E_jpy / Deposits_jpy - 1) × 100")
    print(f"Money-Weighted Return (XIRR, JPY):       {pct_or_na(jpy_xirr):>12}")

    print("\n" + "=" * 120)
    print("SUMMARY")
    print("=" * 120)
    print(f"Total Capital Contributed:               ${final['contributed']:>12,.2f}  (Deposits valued on deposit day)")
    if total_jpy_deposited > 0:
        print(f"  Note: Deposited ¥{total_jpy_deposited:,.0f}, {'gained' if vs_today > 0 else 'lost'} "
              f"${abs(vs_today):,.2f} vs. converting today")
    print(f"Current Portfolio Value:                 ${final['equity']:>12,.2f}  (¥{final['equity_jpy']:,.0f})")
    print(f"Net Investment Gain:                     ${net_investment_pnl:>12,.2f}  ({final['simple_return_pct']:.2f}% on contributed capital)")
    print(f"  - Realized (after commissions):        ${final['realized_pnl'] + gross_dividends - total_withholding_tax - total_fees:>12,.2f}")
    print(f"  - Unrealized (Paper Gains):            ${final['unrealized_pnl']:>12,.2f}")
    print(f"  - FX Gain/Loss:                        ${fx_pnl:>12,.2f}")
    print(f"Total External Costs (taxes + fees):     ${total_withholding_tax + total_fees:>12,.2f}")
    print(f"Total Commissions (embedded in above):   ${total_commissions:>12,.2f}")
    print(f"Annualized Return:                       {twr_annualized:.2f}% TWR, {pct_or_na(usd_xirr)} XIRR (USD), "
          f"{pct_or_na(jpy_xirr)} XIRR (JPY) over {years_invested:.2f} years")
    print(f"Current Allocation:                      {(final['holdings_value']/final['equity']*100):.1f}% equities, {cash_weight:.1f}% cash")
    print("=" * 120)


def _print_annual_summary(transactions, portfolio_history):
    """
    Print the portfolio summary broken down by calendar year (Jan 1 - Dec 31).

    Each year is measured from the previous year-end valuation, so years add up to the 'Total'
    (since inception) column. Years not fully covered by the history are marked with '*'.
    """
    history = portfolio_history.with_columns(pl.col('date').dt.year().alias('year'))
    symbols = [c[:-len('_value')] for c in history.columns if c.endswith('_value') and c != 'holdings_value']

    def period_stats(rows):
        """Metrics for `rows`, measured from the valuation on the day before it starts."""
        first_date = rows['date'][0]
        prior = history.filter(pl.col('date') < first_date).tail(1)
        start = prior.row(0, named=True) if prior.height else None
        end = rows.row(-1, named=True)
        change = lambda key: end[key] - (start[key] if start else 0.0)
        days = (end['date'] - (start['date'] if start else first_date)).days

        # Money-weighted return: the starting value is treated as the first contribution
        before = lambda key: start[key] if start else 0.0
        window = prior.vstack(rows).with_columns(
            (pl.col('contributed') - before('contributed') + before('equity')).alias('usd_basis'),
            (pl.col('deposited_jpy') - before('deposited_jpy') + before('equity_jpy')).alias('jpy_basis'))

        def period_mwr(value_col, basis_col):
            rate = money_weighted_return(window, value_col, basis_col)
            return ((1 + rate) ** (days / 365.25) - 1) * 100 if rate is not None else None

        trades = transactions.filter((pl.col('type') == 'Trade') &
                                     (pl.col('date') >= first_date) & (pl.col('date') <= end['date']))
        stats = {
            'from': first_date, 'to': end['date'], 'days': days,
            'start_equity': before('equity'), 'deposits': change('contributed'),
            'pnl': change('portfolio_pnl'), 'end_equity': end['equity'],
            'unrealized': change('unrealized_pnl'), 'realized': change('realized_pnl'),
            'dividends': change('dividends'), 'tax': -change('withholding_tax'), 'fees': -change('fees'),
            'fx': change('fx_pnl'), 'commissions': trades['commission'].sum(),
            'twr': ((1 + end['twr_pct'] / 100) / (1 + before('twr_pct') / 100) - 1) * 100,
            'mwr': period_mwr('equity', 'usd_basis'),
            'fx_rate': end['fx_rate'],
            'start_equity_jpy': before('equity_jpy'), 'deposits_jpy': change('deposited_jpy'),
            'pnl_jpy': change('equity_jpy') - change('deposited_jpy'), 'end_equity_jpy': end['equity_jpy'],
            'mwr_jpy': period_mwr('equity_jpy', 'jpy_basis'),
            'cash_pct': end['cash'] / end['equity'] * 100 if end['equity'] else None,
        }
        stats['gap'] = stats['pnl'] - sum(stats[k] for k in ['unrealized', 'realized', 'dividends', 'tax', 'fees', 'fx'])
        for sym in symbols:
            stats[f'{sym}_pct'] = end[f'{sym}_value'] / end['equity'] * 100 if end['equity'] else None
        return stats

    columns = []
    for year in history['year'].unique().sort().to_list():
        rows = history.filter(pl.col('year') == year)
        first, last = rows['date'][0], rows['date'][-1]
        full_year = (first.month, first.day, last.month, last.day) == (1, 1, 12, 31)
        columns.append((str(year) if full_year else f"{year}*", period_stats(rows)))
    columns.append(('Total', period_stats(history)))

    def signed(symbol, decimals):
        def fmt(v):
            v = round(v, decimals) + 0.0  # + 0.0 turns -0.00 into 0.00
            return f"{'-' if v < 0 else ''}{symbol}{abs(v):,.{decimals}f}"
        return fmt
    usd, jpy = signed('$', 2), signed('¥', 0)
    pct = lambda v: f"{v:.2f}%" if v is not None else 'N/A'

    label_width, col_width = 38, 14
    width = label_width + col_width * len(columns)

    def line(label, key=None, fmt=str, values=None):
        cells = values if values is not None else [fmt(stats[key]) for _, stats in columns]
        print(f"{label:<{label_width}}" + "".join(f"{c:>{col_width}}" for c in cells))

    print("=" * width)
    line("ANNUAL PORTFOLIO SUMMARY", values=[label for label, _ in columns])
    line("  From", 'from')
    line("  To", 'to')
    line("  Days", 'days')
    print("=" * width)

    print("CAPITAL (USD)")
    line("  Starting value", 'start_equity', usd)
    line("  + Deposits", 'deposits', usd)
    line("  + Portfolio P&L", 'pnl', usd)
    line("  = Ending value", 'end_equity', usd)

    print("\nP&L BREAKDOWN (USD)")
    line("  Change in unrealized gains", 'unrealized', usd)
    line("  Realized gains", 'realized', usd)
    line("  Dividends (gross)", 'dividends', usd)
    line("  Withholding tax", 'tax', usd)
    line("  Account fees", 'fees', usd)
    line("  FX gain/loss on deposited yen", 'fx', usd)
    line("  = Portfolio P&L", 'pnl', usd)
    line("  Commissions (already included above)", 'commissions', usd)

    print("\nRETURNS (for the period, not annualized)")
    line("  Time-weighted (TWR)", 'twr', pct)
    line("  Money-weighted (MWR, USD)", 'mwr', pct)
    line("  Money-weighted (MWR, JPY)", 'mwr_jpy', pct)

    print("\nCAPITAL (JPY)")
    line("  USD/JPY at period end", 'fx_rate', lambda v: f"¥{v:,.2f}")
    line("  Starting value", 'start_equity_jpy', jpy)
    line("  + Deposits", 'deposits_jpy', jpy)
    line("  + Portfolio P&L", 'pnl_jpy', jpy)
    line("  = Ending value", 'end_equity_jpy', jpy)

    print("\nALLOCATION AT PERIOD END (% of total value)")
    held = [sym for sym in symbols if any(stats[f'{sym}_pct'] for _, stats in columns)]
    for sym in sorted(held, key=lambda s: columns[-1][1][f'{s}_pct'] or 0, reverse=True):
        line(f"  {sym}", f'{sym}_pct', pct)
    line("  Cash", 'cash_pct', pct)

    print("=" * width)
    print("* Partial year. Each year starts from the previous year-end value, so the years add up to Total.")
    print("  Deposits are valued in USD on their deposit day; later exchange-rate moves are FX gain/loss.")
    mismatched = [label for label, stats in columns if abs(stats['gap']) >= 0.01]
    if mismatched:
        print(f"  WARNING: P&L breakdown does not add up for {', '.join(mismatched)} - check transactions.")

# mom_backtest.py
# Cross-sectional momentum (12-1) on your Europe universe
# Requires: Investment_universe/europe_prices.csv (or europe_returns.csv)

import pandas as pd
import numpy as np
from pathlib import Path

# ------------------------
# Params (tweak as you like)
# ------------------------
DATA_DIR            = Path("Investment_universe")
PRICES_CSV          = DATA_DIR / "europe_prices.csv"     # uses Close (auto-adjusted)
# Alternatively use returns:
# RETURNS_CSV       = DATA_DIR / "europe_returns.csv"

LOOKBACK_MONTHS     = 12          # formation window length
SKIP_MONTHS         = 1           # skip most-recent month (12-1 momentum)
TOP_Q               = 0.20        # top quantile (e.g., 20%)
BOT_Q               = 0.20        # bottom quantile (for short leg)
LONG_SHORT          = True        # True = long-short; False = long-only (top quantile)
WEIGHTING           = "equal"     # only equal-weight implemented here
TRANS_COST_BPS      = 5           # round-trip cost per turnover (e.g., 5 bps = 0.0005)
MIN_ASSETS_PER_SIDE = 10          # require at least this many names on each side to trade
START                = "2014-01-01"  # start date (after burn-in)
END                  = None          # to latest

# ------------------------
# Helper functions
# ------------------------
def to_monthly_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """End-of-month prices."""
    monthly = prices.resample("M").last()
    return monthly

def monthly_returns_from_prices(prices_m: pd.DataFrame) -> pd.DataFrame:
    rets_m = prices_m.pct_change()
    return rets_m

def make_momentum_signal(prices_m: pd.DataFrame,
                         lookback_m=12, skip_m=1) -> pd.DataFrame:
    """
    12-1 signal: past 12 months cumulative return excluding last 1 month.
    Signal at month t uses prices up to t-1-skip.
    """
    # total return over lookback window (ending skip months before rebalance date)
    # Compute rolling cumulative returns
    ret_m = prices_m.pct_change()
    cum = (1 + ret_m).rolling(window=lookback_m + skip_m, min_periods=lookback_m + skip_m).apply(np.prod, raw=True)
    # Remove the last SKIP months by dividing out last SKIP months product
    if skip_m > 0:
        last_skip_prod = (1 + ret_m).rolling(window=skip_m, min_periods=skip_m).apply(np.prod, raw=True)
        signal = cum / last_skip_prod - 1.0
    else:
        signal = cum - 1.0
    # shift forward 1 step so the signal is known at formation month end
    return signal.shift(1)

def quantile_buckets(signal_row: pd.Series, top_q=0.2, bot_q=0.2):
    """Return index lists for top and bottom buckets."""
    s = signal_row.dropna()
    if s.empty:
        return [], []
    n = len(s)
    top_n = max(int(np.floor(top_q * n)), 0)
    bot_n = max(int(np.floor(bot_q * n)), 0)
    if top_n == 0 or bot_n == 0:
        return [], []
    ranked = s.sort_values(ascending=True)
    bottom = ranked.index[:bot_n].tolist()
    top = ranked.index[-top_n:].tolist()
    return top, bottom

def portfolio_return(next_month_rets: pd.Series, long_names, short_names=None):
    if not long_names:
        return np.nan
    w_long = 1.0 / len(long_names)
    long_leg = next_month_rets.reindex(long_names).mean() if len(long_names) else 0.0
    if short_names is None or len(short_names) == 0:
        return long_leg
    short_leg = next_month_rets.reindex(short_names).mean()
    return long_leg - short_leg

def max_drawdown(series: pd.Series):
    cum = (1 + series.fillna(0)).cumprod()
    peak = cum.cummax()
    dd = cum / peak - 1.0
    return dd.min()

def annualize_return(monthly_series: pd.Series):
    avg_m = monthly_series.mean()
    return (1 + avg_m) ** 12 - 1

def annualize_vol(monthly_series: pd.Series):
    return monthly_series.std(ddof=1) * np.sqrt(12)

def compute_turnover(prev_long, prev_short, long_now, short_now):
    """
    Turnover of long & short books (equal weight).
    We’ll approximate turnover as % names that changed (since equal weights).
    """
    if prev_long is None:
        long_to = 0.0
    else:
        long_to = 1 - len(set(prev_long).intersection(long_now)) / max(len(set(prev_long).union(long_now)), 1)

    if prev_short is None:
        short_to = 0.0
    else:
        short_to = 1 - len(set(prev_short).intersection(short_now)) / max(len(set(prev_short).union(short_now)), 1)

    if short_now is None:  # long-only
        return long_to
    else:
        return 0.5 * (long_to + short_to)

# ------------------------
# Backtest
# ------------------------
def run_backtest(prices: pd.DataFrame):
    prices = prices.loc[START:END].copy()
    prices_m = to_monthly_prices(prices)
    rets_m = monthly_returns_from_prices(prices_m)
    signal = make_momentum_signal(prices_m, LOOKBACK_MONTHS, SKIP_MONTHS)

    portfolio_rets = []
    long_rets = []
    short_rets = []
    turnovers = []
    dates = []

    prev_long = None
    prev_short = None

    for t in signal.index:
        s_row = signal.loc[t]
        long_names, short_names = quantile_buckets(s_row, TOP_Q, BOT_Q if LONG_SHORT else 0.0)

        # enforce a minimum breadth
        if LONG_SHORT:
            if len(long_names) < MIN_ASSETS_PER_SIDE or len(short_names) < MIN_ASSETS_PER_SIDE:
                # skip if too few names
                continue
        else:
            if len(long_names) < MIN_ASSETS_PER_SIDE:
                continue

        # next month's realized return (what we earn after forming at t)
        try:
            next_ret = rets_m.loc[rets_m.index[rets_m.index.get_loc(t) + 1]]
        except Exception:
            break  # last month; no next return

        # turnover & transaction costs
        to = compute_turnover(prev_long, prev_short, long_names, short_names if LONG_SHORT else None)
        turnovers.append(to)

        gross = portfolio_return(next_ret, long_names, short_names if LONG_SHORT else None)
        # cost: round-trip bps * turnover
        cost = (TRANS_COST_BPS / 1e4) * to
        net = gross - cost

        portfolio_rets.append(net)
        if LONG_SHORT:
            long_rets.append(next_ret.reindex(long_names).mean())
            short_rets.append(next_ret.reindex(short_names).mean())
        dates.append(next_ret.name)

        prev_long, prev_short = long_names, (short_names if LONG_SHORT else None)

    port = pd.Series(portfolio_rets, index=pd.to_datetime(dates), name="Mom_Portfolio")
    turn = pd.Series(turnovers, index=pd.to_datetime(dates), name="Turnover")

    # performance summary
    perf = {
        "CAGR": annualize_return(port),
        "AnnVol": annualize_vol(port),
        "Sharpe(ann)": (annualize_return(port) / annualize_vol(port)) if annualize_vol(port) > 0 else np.nan,
        "MaxDD": max_drawdown(port),
        "AvgMonthlyTurnover": turn.mean(),
        "N_Months": len(port)
    }
    perf = pd.Series(perf)

    return port, turn, perf

# ------------------------
# Run
# ------------------------
if __name__ == "__main__":
    prices = pd.read_csv(PRICES_CSV, index_col=0, parse_dates=True)
    # optional: drop very sparse columns
    prices = prices.dropna(axis=1, how="all").ffill().bfill()

    port, turnover, perf = run_backtest(prices)

    # Save results
    out_dir = DATA_DIR
    port.to_csv(out_dir / "mom_portfolio_monthly_returns.csv")
    turnover.to_csv(out_dir / "mom_turnover.csv")
    perf.to_csv(out_dir / "mom_performance_summary.csv")
    print("Performance summary:\n", perf.round(4))

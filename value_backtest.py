# value_backtest.py  (robust)
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

DATA_DIR             = Path("Investment_universe")
PRICES_CSV           = DATA_DIR / "europe_prices.csv"

USE_LOCAL_FUND_CSV   = False
LOCAL_FUND_CSV       = DATA_DIR / "fundamentals.csv"
CACHE_PARQUET        = DATA_DIR / "fundamentals_cache.parquet"

TOP_Q, BOT_Q         = 0.30, 0.30
LONG_SHORT           = True
MIN_ASSETS_PER_SIDE  = 8            # eased a bit so you see results first

FORMATION_MONTH      = 6            # June
HOLDING_MONTHS       = 12
TRANS_COST_BPS       = 10
START, END           = "2013-01-01", None

def load_prices(path: Path):
    px = pd.read_csv(path, index_col=0, parse_dates=True)
    px = px.loc[START:END].dropna(how="all", axis=1).ffill().bfill()
    return px

def to_monthly_prices(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.resample("M").last()

def monthly_returns_from_prices(prices_m: pd.DataFrame) -> pd.DataFrame:
    return prices_m.pct_change()

# -------- Fundamentals v1: book equity + shares (original) --------
def fetch_fundamentals_yf_full(tickers):
    import yfinance as yf
    rows = []
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            bs = tk.balance_sheet
            if bs is None or bs.empty:
                continue
            fiscal_date = bs.columns[0]
            if "Total Stockholder Equity" in bs.index:
                be = bs.loc["Total Stockholder Equity"].iloc[0]
            else:
                continue
            # shares
            sh = None
            try:
                sh = tk.fast_info.get("sharesOutstanding", None)
            except Exception:
                pass
            if sh in [None, 0]:
                try:
                    sh = tk.info.get("sharesOutstanding", None)
                except Exception:
                    sh = None
            rows.append({"ticker": t, "fiscal_date": pd.to_datetime(fiscal_date),
                         "book_equity": be, "shares_out": sh})
        except Exception:
            continue
    return pd.DataFrame(rows)

def load_or_build_fundamentals_full(tickers, cache=CACHE_PARQUET):
    if USE_LOCAL_FUND_CSV and LOCAL_FUND_CSV.exists():
        f = pd.read_csv(LOCAL_FUND_CSV, parse_dates=["fiscal_date"])
        f["ticker"] = f["ticker"].astype(str)
        return f
    if cache.exists():
        try:
            return pd.read_parquet(cache)
        except Exception:
            pass
    print("Fetching fundamentals (book equity + shares) via yfinance …")
    f = fetch_fundamentals_yf_full(tickers)
    try:
        f.to_parquet(cache, index=False)
    except Exception:
        pass
    return f

def form_bm_full(prices_m, fundamentals):
    tickers = prices_m.columns.tolist()
    f = fundamentals.copy()
    f = f[f["ticker"].isin(tickers)]
    f = f.dropna(subset=["book_equity", "shares_out"])
    f = f[(f["book_equity"] > 0) & (f["shares_out"] > 0)].copy()

    jun_mask = prices_m.index.month == FORMATION_MONTH
    rebalance_dates = prices_m.index[jun_mask]

    bm_list = []
    for d in rebalance_dates:
        cutoff = d - pd.DateOffset(months=6)
        sel = (
            f[f["fiscal_date"] <= cutoff]
            .sort_values(["ticker", "fiscal_date"])
            .groupby("ticker")
            .tail(1)
        )
        if sel.empty:
            bm_list.append(pd.Series(dtype=float, name=d))
            continue
        px_d = prices_m.loc[d]
        me = sel.set_index("ticker")["shares_out"] * px_d.reindex(sel["ticker"]).values
        bm = sel.set_index("ticker")["book_equity"] / me
        bm.name = d
        bm_list.append(bm)
    return pd.DataFrame(bm_list)

# -------- Fundamentals v2 (fallback): Book Value per Share --------
def fetch_bvps_yf(tickers):
    import yfinance as yf
    rows = []
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            bvps = None
            try:
                bvps = tk.fast_info.get("bookValue", None)   # per share
            except Exception:
                pass
            if bvps in [None, 0]:
                try:
                    bvps = tk.info.get("bookValue", None)
                except Exception:
                    bvps = None
            if bvps not in [None, 0]:
                rows.append({"ticker": t, "bvps": float(bvps)})
        except Exception:
            continue
    return pd.DataFrame(rows)

def form_bm_bvps(prices_m, bvps_df):
    """B/M ≈ BVPS / Price at June (lag is implicitly handled by using last annual BVPS)."""
    jun_mask = prices_m.index.month == FORMATION_MONTH
    rebalance_dates = prices_m.index[jun_mask]
    bvps = bvps_df.set_index("ticker")["bvps"]
    bm_list = []
    for d in rebalance_dates:
        px_d = prices_m.loc[d]
        common = bvps.index.intersection(px_d.index)
        bm = (bvps.loc[common] / px_d.loc[common]).dropna()
        bm.name = d
        bm_list.append(bm)
    return pd.DataFrame(bm_list)

# -------- Portfolio helpers --------
def quantile_buckets(signal_row, top_q=0.3, bot_q=0.3):
    s = signal_row.dropna()
    if s.empty:
        return [], []
    n = len(s)
    top_n = max(int(np.floor(top_q * n)), 0)
    bot_n = max(int(np.floor(bot_q * n)), 0)
    ranked = s.sort_values()
    growth = ranked.index[:bot_n].tolist()
    value  = ranked.index[-top_n:].tolist()
    return value, growth

def portfolio_return(next_month_rets, long_names, short_names=None):
    if not long_names:
        return np.nan
    long_leg = next_month_rets.reindex(long_names).mean()
    if short_names:
        short_leg = next_month_rets.reindex(short_names).mean()
        return long_leg - short_leg
    return long_leg

def compute_turnover(prev_long, prev_short, long_now, short_now):
    if prev_long is None:
        long_to = 0.0
    else:
        long_to = 1 - len(set(prev_long).intersection(long_now)) / max(len(set(prev_long).union(long_now)), 1)
    if prev_short is None:
        short_to = 0.0
    else:
        short_to = 1 - len(set(prev_short).intersection(short_now)) / max(len(set(prev_short).union(short_now)), 1)
    return long_to if short_now is None else 0.5*(long_to+short_to)

def max_drawdown(series):
    cum = (1 + series.fillna(0)).cumprod()
    peak = cum.cummax()
    return (cum/peak - 1).min()

def ann_ret(x):  return (1 + x.mean())**12 - 1
def ann_vol(x):  return x.std(ddof=1) * np.sqrt(12)

# -------- Backtest --------
def run_value_backtest(prices):
    prices_m = to_monthly_prices(prices)
    rets_m   = monthly_returns_from_prices(prices_m)

    tickers = prices.columns.tolist()
    # Try full fundamentals
    fund_full = load_or_build_fundamentals_full(tickers)
    print(f"Full fundamentals rows: {len(fund_full)} (unique tickers: {fund_full['ticker'].nunique() if not fund_full.empty else 0})")
    bm_full = form_bm_full(prices_m, fund_full) if not fund_full.empty else pd.DataFrame()

    coverage_full = bm_full.notna().sum(axis=1).mean() if not bm_full.empty else 0
    print(f"Avg usable names per June (full method): {coverage_full:.1f}")

    # Fallback BVPS if coverage is thin (< 30 names on average)
    if (bm_full.empty) or (coverage_full < 30):
        print("Using fallback: BookValuePerShare / Price …")
        bvps_df = fetch_bvps_yf(tickers)
        print(f"BVPS fetched for {len(bvps_df)} tickers")
        bm = form_bm_bvps(prices_m, bvps_df)
    else:
        bm = bm_full

    print(f"Avg usable names per June (final B/M): {bm.notna().sum(axis=1).mean():.1f}")
    if bm.empty or bm.notna().sum(axis=1).max() < max(2*MIN_ASSETS_PER_SIDE, 20):
        print("Not enough coverage to form portfolios. Try lowering MIN_ASSETS_PER_SIDE or widen TOP/BOT quantiles.")
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)

    dates, port_rets, turnovers = [], [], []
    prev_long, prev_short = None, None

    for d in bm.index:
        value_names, growth_names = quantile_buckets(bm.loc[d], TOP_Q, BOT_Q if LONG_SHORT else 0.0)
        if LONG_SHORT:
            if len(value_names) < MIN_ASSETS_PER_SIDE or len(growth_names) < MIN_ASSETS_PER_SIDE:
                continue
        else:
            if len(value_names) < MIN_ASSETS_PER_SIDE:
                continue
        if d not in rets_m.index: 
            continue
        pos = rets_m.index.get_loc(d)
        hold_slice = rets_m.iloc[pos+1: pos+1+HOLDING_MONTHS]

        for t, next_ret in hold_slice.iterrows():
            to = compute_turnover(prev_long, prev_short, value_names, growth_names if LONG_SHORT else None)
            gross = portfolio_return(next_ret, value_names, growth_names if LONG_SHORT else None)
            cost  = (TRANS_COST_BPS/1e4) * to
            net   = gross - cost
            dates.append(t); port_rets.append(net); turnovers.append(to)
            prev_long, prev_short = value_names, (growth_names if LONG_SHORT else None)

    port = pd.Series(port_rets, index=pd.to_datetime(dates), name="Value_Portfolio").sort_index()
    turn = pd.Series(turnovers, index=port.index, name="Turnover")

    if port.empty:
        return port, turn, pd.Series(dtype=float)

    perf = pd.Series({
        "CAGR": ann_ret(port),
        "AnnVol": ann_vol(port),
        "Sharpe(ann)": (ann_ret(port)/ann_vol(port)) if ann_vol(port) > 0 else np.nan,
        "MaxDD": max_drawdown(port),
        "AvgMonthlyTurnover": turn.mean(),
        "N_Months": len(port)
    })
    return port, turn, perf

if __name__ == "__main__":
    prices = load_prices(PRICES_CSV)
    port, turnover, perf = run_value_backtest(prices)

    (DATA_DIR / "value_portfolio_monthly_returns.csv").write_text(port.to_csv())
    (DATA_DIR / "value_turnover.csv").write_text(turnover.to_csv())
    (DATA_DIR / "value_performance_summary.csv").write_text(perf.to_csv())

    print("\nPerformance summary:")
    print(perf.round(4))

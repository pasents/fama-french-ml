# viz_strategy_vs_msci.py
# -------------------------------------------------------------
# Visualizes your momentum allocation strategy vs an MSCI benchmark (IWDA.AS by default).
# Accepts either:
#   - Yahoo Finance pull (if internet available)
#   - Local Investment_universe/MSCI_BENCHMARK.csv
#       • Price format: Date, Adj Close or Close
#       • Return format: Date, Benchmark (any freq; will be compounded to monthly)
# -------------------------------------------------------------

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

TIMESERIES = "reports/momentum_allocation_timeseries.csv"
BENCH_FALLBACK = "Investment_universe/MSCI_BENCHMARK.csv"


# =======================
#  Helpers
# =======================
def to_month_end_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return a month-end DatetimeIndex (no ambiguity)."""
    return (idx.to_period("M").to_timestamp("M")).tz_localize(None)

def compound_to_monthly_from_returns(df: pd.DataFrame) -> pd.Series:
    """
    Accepts returns at any frequency with a DatetimeIndex and ONE numeric column.
    Compounds to monthly simple returns: (1+r).prod() - 1 per month.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index must be datetime for returns format.")
    # group by Year-Month and compound
    g = (1.0 + df.iloc[:, 0]).groupby([df.index.year, df.index.month]).prod() - 1.0
    month_end = pd.to_datetime([f"{y}-{m:02d}-01" for (y, m) in g.index]) + pd.offsets.MonthEnd(0)
    return pd.Series(g.values, index=month_end, name="Benchmark").sort_index()

def monthly_returns_from_prices(px: pd.DataFrame) -> pd.Series:
    """Compute monthly simple returns from a price series with DatetimeIndex."""
    if getattr(px.index, "tz", None) is not None:
        px.index = px.index.tz_localize(None)
    px = px.sort_index()
    mpx = px.resample("ME").last()
    return mpx.pct_change().dropna().iloc[:, 0].rename("Benchmark")

def perf_stats(r: pd.Series):
    r = r.dropna()
    n = len(r)
    if n == 0:
        return np.nan, np.nan, np.nan
    ann_ret = (1 + r).prod() ** (12 / n) - 1
    ann_vol = r.std(ddof=0) * np.sqrt(12)
    sharpe = (r.mean() * 12) / (ann_vol if ann_vol != 0 else np.nan)
    return ann_ret, ann_vol, sharpe


# =======================
#  Load Strategy Returns
# =======================
def load_strategy(ts_path=TIMESERIES):
    if not os.path.exists(ts_path):
        raise FileNotFoundError(f"{ts_path} not found. Run momentum_allocation_backbone.py first.")

    ts = pd.read_csv(ts_path, parse_dates=[0], index_col=0).sort_index()
    port_col = "Port" if "Port" in ts.columns else ("Const70_30" if "Const70_30" in ts.columns else None)
    if port_col is None:
        raise RuntimeError("Neither 'Port' nor 'Const70_30' found in the timeseries CSV.")
    strat = ts[port_col].dropna().rename("Strategy")
    strat.index = to_month_end_index(strat.index)  # enforce month-end
    return strat, port_col


# =========================
#  Fetch or Load Benchmark
# =========================
def fetch_benchmark():
    # --- Try Yahoo Finance first ---
    try:
        import yfinance as yf
        candidates = [
            ("IWDA.AS", "MSCI World (IWDA.AS, EUR)"),
            ("EUNL.DE", "MSCI World (EUNL.DE, EUR)"),
            ("IMEU.L",  "MSCI Europe (IMEU.L, GBP)"),
            ("IEUR",    "MSCI Europe (IEUR, USD)"),
        ]
        for tkr, label in candidates:
            try:
                df = yf.download(tkr, period="max", interval="1d", auto_adjust=True, progress=False)
                if df is None or df.empty:
                    continue
                col = "Close" if "Close" in df.columns else ("Adj Close" if "Adj Close" in df.columns else None)
                if col is None:
                    continue
                px = df[[col]].rename(columns={col: "PX"}).dropna()
                mret = monthly_returns_from_prices(px)
                mret.index = to_month_end_index(mret.index)
                return mret, f"{label} [{tkr}]"
            except Exception:
                continue
    except Exception:
        pass

    # --- Manual CSV fallback ---
    if not os.path.exists(BENCH_FALLBACK):
        raise RuntimeError("Benchmark unavailable. Provide Investment_universe/MSCI_BENCHMARK.csv or enable internet/yfinance.")

    df = pd.read_csv(BENCH_FALLBACK)
    # normalize headers
    lower = {c.lower().strip().replace(" ", ""): c for c in df.columns}

    # PRICE CSV case
    date_col = next((lower[k] for k in lower if k == "date"), None)
    price_col = next((lower[k] for k in lower if k in ("adjclose", "adjustedclose", "close", "px", "price")), None)
    if date_col and price_col:
        px = df[[date_col, price_col]].dropna()
        px[date_col] = pd.to_datetime(px[date_col])
        px = px.set_index(date_col).sort_index()
        mret = monthly_returns_from_prices(px)
        mret.index = to_month_end_index(mret.index)
        return mret, "Custom MSCI (manual PRICE CSV)"

    # RETURNS CSV case
    date_guess = next((c for c in df.columns if c.lower() in ("date", "month", "time_period")), None)
    if date_guess is None:
        raise ValueError("MSCI_BENCHMARK.csv must have a Date column (for return format).")

    df[date_guess] = pd.to_datetime(df[date_guess])
    df = df.set_index(date_guess).sort_index()

    # find a numeric returns column
    if "Benchmark" in df.columns:
        ret_col = "Benchmark"
    else:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            raise ValueError("MSCI_BENCHMARK.csv must contain a numeric return column.")
        ret_col = num_cols[0]

    ret = pd.DataFrame(df[ret_col].astype(float))
    # If frequency > monthly, compound to monthly; otherwise normalize to month-end.
    mret = compound_to_monthly_from_returns(ret)
    mret.index = to_month_end_index(mret.index)
    return mret, "Custom MSCI (manual RETURN CSV)"


# =======================
#  Main Script
# =======================
if __name__ == "__main__":
    strat, strat_label = load_strategy()
    bench, bench_label = fetch_benchmark()

    both = pd.concat([strat, bench], axis=1).dropna()
    both.index = to_month_end_index(both.index)  # unify index
    cum = (1.0 + both).cumprod()

    # ---- Plot
    plt.figure()
    cum.plot()
    plt.title(f"Cumulative Returns: {strat_label} vs {bench_label}")
    plt.xlabel("Date")
    plt.ylabel("Growth of 1")
    plt.tight_layout()
    os.makedirs("reports", exist_ok=True)
    out_png = "reports/strategy_vs_msci.png"
    plt.savefig(out_png, dpi=150)

    # ---- Metrics
    s_ret, s_vol, s_sh = perf_stats(both["Strategy"])
    b_ret, b_vol, b_sh = perf_stats(both["Benchmark"])
    corr = both["Strategy"].corr(both["Benchmark"])

    # ---- Save aligned monthly data
    out_csv = "reports/strategy_vs_msci_monthly.csv"
    both.to_csv(out_csv, float_format="%.8f")

    print(f"Saved plot: {out_png}")
    print(f"Saved data: {out_csv}")
    print("\n== Metrics (annualized) ==")
    print(f"Strategy  : Return {s_ret:.2%}, Vol {s_vol:.2%}, Sharpe {s_sh:.2f}")
    print(f"Benchmark : Return {b_ret:.2%}, Vol {b_vol:.2%}, Sharpe {b_sh:.2f}")
    print(f"Correlation (monthly): {corr:.2f}")

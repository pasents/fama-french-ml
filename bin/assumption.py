#!/usr/bin/env python3
"""
assumption.py — Multi-asset time-series assumption checks

Usage:
  python assumption.py --file PATH [--out DIR] [--max-plots N]

Input formats (auto-detected):
1) Long: columns include (date, ticker, price/close/adj_close) or (date, ticker, ret)
2) Wide: first column is date (or date index), remaining columns are tickers with prices/returns

What it does:
- Detects schema (prices vs returns) and builds a panel (Date x Tickers)
- If prices supplied, computes simple returns
- Cleans data (drops all-empty series, requires >= 12 obs)
- Assumption tests per asset (on returns):
  * Stationarity: ADF and KPSS
  * Serial correlation: Ljung–Box (several lags)
  * Normality: Jarque–Bera (plus skew & kurtosis)
  * Conditional heteroskedasticity: Engle’s ARCH LM test
  * Breaks (mean stability): CUSUM test on constant-only OLS residuals
- Descriptives: ann. return/vol, Sharpe(0%), max drawdown, lag-1 autocorr
- Cross-section: Pearson & Spearman correlations, eigen spectrum, effective rank
- Outliers: counts of |z| > 6
- Plots: hist of ann returns, hist of Sharpes, eigenvalue scree,
         per-asset (top-N): return histogram + QQ, ACF stem plot

Outputs:
  - CSVs and PNG charts saved in output directory
  - SUMMARY.json with meta and pointers to outputs
"""

import argparse
import math
import os
import json
import warnings
from typing import Dict, Optional, Tuple, List

# --- headless-friendly plotting (VS Code / Windows) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

# Optional stats imports with safe fallbacks
try:
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch, breaks_cusumolsresid
except Exception:
    sm = None
    adfuller = None
    kpss = None
    acorr_ljungbox = None
    het_arch = None
    breaks_cusumolsresid = None

try:
    from scipy import stats as sstats
except Exception:
    sstats = None

ASSUMED_TRADING_DAYS = 252

DATE_CANDS = ["date", "Date", "DATE", "timestamp", "Timestamp"]
TICKER_CANDS = ["ticker", "Ticker", "TICKER", "symbol", "Symbol"]
PRICE_CANDS = ["adj_close", "Adj Close", "adjclose", "close", "Close", "price", "Price"]
RET_CANDS = ["ret", "return", "returns", "Ret", "Return"]


def detect_schema(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    schema = {"format": None, "date": None, "ticker": None, "price": None, "ret": None}
    cols = list(df.columns)

    # Index-as-date wide
    if not isinstance(df.index, pd.RangeIndex):
        try:
            pd.to_datetime(df.index)
            schema["format"] = "wide_index_date"
            schema["date"] = "__index__"
            return schema
        except Exception:
            pass

    # Date col
    for c in DATE_CANDS:
        if c in cols:
            schema["date"] = c
            break

    # Ticker col
    for c in TICKER_CANDS:
        if c in cols:
            schema["ticker"] = c
            break

    # Price / Ret
    for c in PRICE_CANDS:
        if c in cols:
            schema["price"] = c
            break
    for c in RET_CANDS:
        if c in cols:
            schema["ret"] = c
            break

    if schema["date"] and schema["ticker"]:
        schema["format"] = "long"
    elif schema["date"]:
        # likely wide with date col and numeric columns
        num_numeric = df.select_dtypes(include=[np.number]).shape[1]
        if num_numeric >= 2:
            schema["format"] = "wide_date_col"
    else:
        # fallback: first col as date for wide data
        first = cols[0]
        try:
            pd.to_datetime(df[first])
            schema["date"] = first
            schema["format"] = "wide_date_col"
        except Exception:
            pass
    return schema


def to_panel(df: pd.DataFrame, schema: Dict[str, Optional[str]]) -> Tuple[pd.DataFrame, str]:
    if schema["format"] == "long":
        d = df.copy()
        d[schema["date"]] = pd.to_datetime(d[schema["date"]], errors="coerce")
        if schema["price"] is not None:
            pvt = d.pivot_table(index=schema["date"], columns=schema["ticker"], values=schema["price"], aggfunc="last")
            kind = "price"
        elif schema["ret"] is not None:
            pvt = d.pivot_table(index=schema["date"], columns=schema["ticker"], values=schema["ret"], aggfunc="last")
            kind = "return"
        else:
            for c in ["Close", "close", "adj_close", "Adj Close", "price", "Price"]:
                if c in d.columns:
                    pvt = d.pivot_table(index=schema["date"], columns=schema["ticker"], values=c, aggfunc="last")
                    kind = "price"
                    break
            else:
                raise ValueError("No price/return column found in long-format data.")
        pvt.sort_index(inplace=True)
        return pvt, kind

    elif schema["format"] in ["wide_date_col", "wide_index_date"]:
        d = df.copy()
        if schema["format"] == "wide_date_col":
            d[schema["date"]] = pd.to_datetime(d[schema["date"]], errors="coerce")
            d.set_index(schema["date"], inplace=True)
        else:
            d.index = pd.to_datetime(d.index, errors="coerce")

        # coerce non-numerics to NaN
        vals = d.apply(pd.to_numeric, errors="coerce")
        median_val = np.nanmedian(vals.to_numpy())
        kind = "price" if median_val > 2 else "return"
        vals.sort_index(inplace=True)
        return vals, kind

    else:
        raise ValueError("Unrecognized format; supply long (date,ticker,price/ret) or wide (date + columns).")


def compute_returns(price_panel: pd.DataFrame) -> pd.DataFrame:
    return price_panel.sort_index().pct_change()


def annualize(mu: pd.Series, sigma: pd.Series) -> Tuple[pd.Series, pd.Series]:
    ann_mu = (1 + mu).pow(ASSUMED_TRADING_DAYS) - 1
    ann_sigma = sigma * math.sqrt(ASSUMED_TRADING_DAYS)
    return ann_mu, ann_sigma


def max_drawdown_from_returns(series: pd.Series) -> float:
    s = series.dropna()
    if s.empty:
        return np.nan
    eq = (1 + s).cumprod()
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    return float(dd.min())


def effective_rank(cov: np.ndarray) -> float:
    w, _ = np.linalg.eig(cov)
    w = np.real(w)
    w = w[w > 0]
    if w.size == 0:
        return np.nan
    p = w / w.sum()
    H = -np.sum(p * np.log(p))
    return float(np.exp(H))


def jarque_bera(x: pd.Series) -> Tuple[float, float]:
    if sstats is None:
        return (np.nan, np.nan)
    x = x.dropna()
    if x.size < 8:
        return (np.nan, np.nan)
    jb_stat, p = sstats.jarque_bera(x)
    return float(jb_stat), float(p)


def adf_test(x: pd.Series) -> Tuple[float, float]:
    if adfuller is None:
        return (np.nan, np.nan)
    x = x.dropna()
    if x.size < 12:
        return (np.nan, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = adfuller(x, autolag="AIC")
    return float(res[0]), float(res[1])


def kpss_test(x: pd.Series) -> Tuple[float, float]:
    if kpss is None:
        return (np.nan, np.nan)
    x = x.dropna()
    if x.size < 20:
        return (np.nan, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p, _, _ = kpss(x, regression="c", nlags="auto")
    return float(stat), float(p)


def ljung_box(x: pd.Series, lags: List[int] = [5, 10, 20]) -> Dict[int, float]:
    if acorr_ljungbox is None:
        return {L: np.nan for L in lags}
    x = x.dropna()
    pvals = {}
    for L in lags:
        try:
            out = acorr_ljungbox(x, lags=[L], return_df=True)
            pvals[L] = float(out["lb_pvalue"].iloc[0])
        except Exception:
            pvals[L] = np.nan
    return pvals


def arch_lm(x: pd.Series, lags: int = 12) -> Tuple[float, float]:
    if het_arch is None:
        return (np.nan, np.nan)
    x = x.dropna()
    if x.size < lags + 5:
        return (np.nan, np.nan)
    try:
        stat, p, _, _ = het_arch(x, nlags=lags)  # fixed deprecation
        return float(stat), float(p)
    except Exception:
        return (np.nan, np.nan)


def cusum_mean_break(x: pd.Series) -> Tuple[float, float]:
    """CUSUM test on residuals of mean-only OLS: x_t = mu + e_t (H0: stable)."""
    if (sm is None) or (breaks_cusumolsresid is None):
        return (np.nan, np.nan)
    x = x.dropna()
    if x.size < 30:
        return (np.nan, np.nan)
    try:
        X = np.ones((x.shape[0], 1))
        ols = sm.OLS(x.values, X).fit()
        stat, p, _ = breaks_cusumolsresid(ols.resid, ddof=1)
        return float(stat), float(p)
    except Exception:
        return (np.nan, np.nan)


def longest_na_run(s: pd.Series) -> int:
    """Length of the longest consecutive NA run in a Series."""
    if s.isna().all():
        return int(s.size)
    x = s.isna().astype(int)
    groups = (x == 0).cumsum()
    runs = (x.groupby(groups).cumsum() * x)
    m = runs.max()
    return int(m) if pd.notna(m) else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="Path to CSV/Parquet with prices or returns")
    ap.add_argument("--out", default="assumption_diagnostics", help="Output directory")
    ap.add_argument("--max-plots", type=int, default=24, help="Max number of per-asset plots (top-N by coverage)")
    args = ap.parse_args()

    inpath = args.file
    outdir = args.out
    os.makedirs(outdir, exist_ok=True)

    # Load (robust CSV/Parquet handling)
    if inpath.lower().endswith(".csv"):
        try:
            df = pd.read_csv(inpath, engine="python")
        except Exception:
            df = pd.read_csv(inpath)
    else:
        try:
            df = pd.read_parquet(inpath)
        except Exception as e:
            raise RuntimeError(
                f"Failed to read Parquet. Install pyarrow or fastparquet. Original error: {e}"
            )

    # Detect schema & build panel
    schema = detect_schema(df)
    panel, kind = to_panel(df, schema)
    panel.index = pd.to_datetime(panel.index, errors="coerce")
    panel = panel.sort_index()

    # coerce all columns to numeric, drop all-empty & short series
    panel = panel.apply(pd.to_numeric, errors="coerce")
    panel = panel.dropna(axis=1, how="all")
    panel = panel.loc[:, panel.count() >= 12]

    # Build returns
    if kind == "price":
        returns = compute_returns(panel)
    else:
        returns = panel.copy()

    # Coverage
    coverage = pd.DataFrame({
        "first_date": returns.apply(lambda s: s.first_valid_index()),
        "last_date":  returns.apply(lambda s: s.last_valid_index()),
        "n_obs":      returns.count(),
        "pct_missing": returns.isna().mean() * 100.0,
        "consecutive_na_max": returns.apply(longest_na_run)
    })
    coverage.to_csv(os.path.join(outdir, "coverage_missingness.csv"))

    # Descriptives
    mu = returns.mean()
    sigma = returns.std()
    skew = returns.skew()
    kurt = returns.kurt()
    ac1 = returns.apply(lambda s: s.autocorr(lag=1))

    ann_mu, ann_sigma = annualize(mu, sigma)
    sharpe = ann_mu / ann_sigma.replace(0, np.nan)
    mdd = returns.apply(max_drawdown_from_returns)

    # Outliers
    z = (returns - mu) / sigma.replace(0, np.nan)
    outliers = (z.abs() > 6.0).sum().sort_values(ascending=False)

    # Assumption tests per asset
    rows = []
    for col in returns.columns:
        s = returns[col]
        jb_stat, jb_p = jarque_bera(s)
        adf_stat, adf_p = adf_test(s)
        kpss_stat, kpss_p = kpss_test(s)
        lb_p = ljung_box(s, lags=[5, 10, 20])
        arch_stat, arch_p = arch_lm(s, lags=12)
        csum_stat, csum_p = cusum_mean_break(s)
        rows.append({
            "asset": col,
            "ann_return": ann_mu.get(col, np.nan),
            "ann_vol": ann_sigma.get(col, np.nan),
            "sharpe": sharpe.get(col, np.nan),
            "skew": skew.get(col, np.nan),
            "kurtosis": kurt.get(col, np.nan),
            "autocorr_lag1": ac1.get(col, np.nan),
            "max_drawdown": mdd.get(col, np.nan),
            "outliers_gt6sigma": outliers.get(col, 0),
            "JB_stat": jb_stat, "JB_p": jb_p,
            "ADF_stat": adf_stat, "ADF_p": adf_p,
            "KPSS_stat": kpss_stat, "KPSS_p": kpss_p,
            "LB_p_lag5": lb_p.get(5, np.nan),
            "LB_p_lag10": lb_p.get(10, np.nan),
            "LB_p_lag20": lb_p.get(20, np.nan),
            "ARCH_LM_stat": arch_stat, "ARCH_LM_p": arch_p,
            "CUSUM_stat": csum_stat, "CUSUM_p": csum_p
        })
    asset_tests = pd.DataFrame(rows).set_index("asset").sort_values("sharpe", ascending=False)
    asset_tests.to_csv(os.path.join(outdir, "asset_assumption_tests.csv"))

    # Correlations
    valid = returns.dropna(how="all", axis=1)
    corr_pearson = valid.corr(method="pearson", min_periods=30)
    corr_spearman = valid.corr(method="spearman", min_periods=30)
    corr_pearson.to_csv(os.path.join(outdir, "correlations_pearson.csv"))
    corr_spearman.to_csv(os.path.join(outdir, "correlations_spearman.csv"))

    # Eigen spectrum & effective rank (mean-impute for NaNs so eig works)
    X = valid.copy()
    col_means = X.mean(skipna=True)
    X = X.fillna(col_means)
    cov = np.cov(X.to_numpy().T)
    w, _ = np.linalg.eig(cov)
    w = np.real(w)
    w_sorted = np.sort(w)[::-1]
    eig_df = pd.DataFrame({"eigenvalue": w_sorted})
    eig_df.to_csv(os.path.join(outdir, "eigen_spectrum.csv"), index=False)

    try:
        erank = effective_rank(cov)
    except Exception:
        erank = np.nan

    # --------- Plots (matplotlib, one plot per figure, default colors) ---------
    fig = plt.figure(figsize=(6, 4))
    plt.hist(asset_tests["ann_return"].dropna(), bins=30)
    plt.title("Histogram of Annualized Returns")
    plt.xlabel("Annualized Return"); plt.ylabel("Count")
    fig.savefig(os.path.join(outdir, "plot_ann_return_hist.png"), bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(6, 4))
    plt.hist(asset_tests["sharpe"].replace([np.inf, -np.inf], np.nan).dropna(), bins=30)
    plt.title("Histogram of Sharpe Ratios")
    plt.xlabel("Sharpe (ann.)"); plt.ylabel("Count")
    fig.savefig(os.path.join(outdir, "plot_sharpe_hist.png"), bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, len(w_sorted) + 1), w_sorted, marker="o")
    title = f"Scree Plot (Eff. Rank ≈ {erank:.1f})" if not np.isnan(erank) else "Scree Plot"
    plt.title(title); plt.xlabel("Component"); plt.ylabel("Eigenvalue")
    fig.savefig(os.path.join(outdir, "plot_scree.png"), bbox_inches="tight")
    plt.close(fig)

    # Per-asset plots (top-N by coverage)
    avail = returns.count().sort_values(ascending=False)
    topN = list(avail.index[:max(0, int(args.max_plots))])
    for col in topN:
        s = returns[col].dropna()
        if s.empty:
            continue

        # Histogram
        fig = plt.figure(figsize=(6, 4))
        plt.hist(s, bins=40)
        plt.title(f"{col} — Return Histogram")
        plt.xlabel("Return"); plt.ylabel("Count")
        fig.savefig(os.path.join(outdir, f"{col}_hist.png"), bbox_inches="tight")
        plt.close(fig)

        # QQ plot — save the actual statsmodels figure
        try:
            if sm is not None:
                fig = sm.qqplot(s, line="s")
                fig.suptitle(f"{col} — QQ Plot")
                fig.savefig(os.path.join(outdir, f"{col}_qq.png"), bbox_inches="tight")
                plt.close(fig)
            else:
                fig = plt.figure(figsize=(6, 4))
                x = np.sort(s.values)
                q = np.sort(np.random.normal(loc=0, scale=np.std(s), size=len(x)))
                plt.scatter(q, x, s=8)
                plt.title(f"{col} — QQ (approx)")
                plt.xlabel("Theoretical Quantiles"); plt.ylabel("Sample Quantiles")
                fig.savefig(os.path.join(outdir, f"{col}_qq.png"), bbox_inches="tight")
                plt.close(fig)
        except Exception:
            pass

        # ACF — draw on a figure we control, then save
        try:
            if sm is not None:
                from statsmodels.graphics.tsaplots import plot_acf
                fig = plt.figure(figsize=(6, 4))
                ax = fig.add_subplot(111)
                plot_acf(s, lags=min(40, max(5, int(len(s) / 10))), alpha=0.05, ax=ax)
                ax.set_title(f"{col} — ACF")
                fig.savefig(os.path.join(outdir, f"{col}_acf.png"), bbox_inches="tight")
                plt.close(fig)
            else:
                fig = plt.figure(figsize=(6, 4))
                lmax = min(40, max(5, int(len(s) / 10)))
                acfs = [s.autocorr(lag=k) for k in range(1, lmax + 1)]
                plt.stem(range(1, lmax + 1), acfs, use_line_collection=True)
                plt.title(f"{col} — ACF (approx)")
                plt.xlabel("Lag"); plt.ylabel("ACF")
                fig.savefig(os.path.join(outdir, f"{col}_acf.png"), bbox_inches="tight")
                plt.close(fig)
        except Exception:
            pass

    meta = {
        "input_file": inpath,
        "schema": schema,
        "interpreted_kind": kind,
        "n_assets": int(returns.shape[1]),
        "date_min": str(returns.index.min()) if len(returns.index) else None,
        "date_max": str(returns.index.max()) if len(returns.index) else None,
        "outputs": {
            "coverage_missingness": "coverage_missingness.csv",
            "asset_assumption_tests": "asset_assumption_tests.csv",
            "outliers_note": "included in asset_assumption_tests (outliers_gt6sigma)",
            "correlations_pearson": "correlations_pearson.csv",
            "correlations_spearman": "correlations_spearman.csv",
            "eigen_spectrum": "eigen_spectrum.csv",
            "plots": [
                "plot_ann_return_hist.png",
                "plot_sharpe_hist.png",
                "plot_scree.png",
                "* per-asset: <ASSET>_{hist,qq,acf}.png"
            ]
        }
    }
    with open(os.path.join(outdir, "SUMMARY.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print("Done. Outputs in:", outdir)
    print(json.dumps(meta, indent=2, default=str))


if __name__ == "__main__":
    main()

"""
run_factor_regressions.py
Runs time-series factor regressions (OLS + Newey–West) using available factors.
Outputs:
  - reports/factor_regression_summary.csv
  - reports/fama_macbeth_results.csv (placeholder)
  - reports/beta_scatter_MKT_vs_MOM.png
  - reports/rolling_*.csv, reports/rolling_beta_MKT_avg.png
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import statsmodels.api as sm

DATA_DIR = Path("Investment_universe")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

FACTORS_CSV = DATA_DIR / "factors_europe.csv"
RETURNS_CSV_CANDIDATES = [
    DATA_DIR / "modeling_returns.csv",
    DATA_DIR / "europe_returns_cleaned.csv",
    DATA_DIR / "europe_returns.csv",
    DATA_DIR / "returns.csv",
]

MIN_MONTHS_FOR_FACTOR = 24
MIN_MONTHS_FOR_REGRESSION = 24

# ------------------ helpers ------------------

def _read_first_existing(paths):
    for p in paths:
        if p.exists():
            df = pd.read_csv(p, index_col=0, parse_dates=True)
            return df
    raise FileNotFoundError(f"No returns file found. Looked for: {paths}")

def _to_month_end_unique_df(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.sort_index()
    df.index = df.index.to_period("M").to_timestamp("M")
    if df.index.has_duplicates:
        df = df.groupby(df.index).last()
    df = df.loc[:, ~df.columns.duplicated()]
    return df

def _to_month_end_unique_series(s: pd.Series) -> pd.Series:
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index, errors="coerce")
    s = s.sort_index()
    s.index = s.index.to_period("M").to_timestamp("M")
    if s.index.has_duplicates:
        s = s.groupby(s.index).last()
    return s

def load_data():
    # Returns
    rets = _read_first_existing(RETURNS_CSV_CANDIDATES)
    rets = _to_month_end_unique_df(rets)

    # Factors
    fac = pd.read_csv(FACTORS_CSV, index_col=0, parse_dates=True)
    fac = _to_month_end_unique_df(fac)

    # Keep only known factor names that exist
    avail = [c for c in ["MKT","SMB","HML","RMW","CMA","MOM","RF"] if c in fac.columns]
    if not avail:
        raise ValueError("No factor columns found in factors_europe.csv.")
    fac = fac[avail]

    # Compute excess returns if RF exists
    if "RF" in fac.columns:
        rf = _to_month_end_unique_series(fac["RF"])
        rf = rf.reindex(rets.index).ffill()  # align to returns calendar
        exrets = rets.sub(rf, axis=0)
    else:
        exrets = rets.copy()

    # Align dates
    idx = exrets.index.intersection(fac.index)
    exrets = exrets.loc[idx].sort_index()
    fac = fac.loc[idx].sort_index()

    # Drop factor columns with insufficient data (except RF)
    fac_no_rf = fac.drop(columns=[c for c in ["RF"] if c in fac.columns], errors="ignore").copy()
    sufficient = [c for c in fac_no_rf.columns if fac_no_rf[c].notna().sum() >= MIN_MONTHS_FOR_FACTOR]
    if not sufficient:
        raise ValueError("All factor columns are too sparse. Ensure at least one factor has >= "
                         f"{MIN_MONTHS_FOR_FACTOR} non-NaN months.")
    fac_use = fac_no_rf[sufficient]

    # Final regressors with constant
    X = sm.add_constant(fac_use)

    return exrets, X, sufficient

def nw_ts_regression(y, X, lags=6):
    model = sm.OLS(y, X, missing="drop")
    res = model.fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return res

# ------------------ rolling betas ------------------

def run_rolling_betas(window=36):
    """Compute rolling betas for each ticker using the currently used factors."""
    exrets, X, used_factors = load_data()
    dfX = X.drop(columns=["const"])  # only factor columns
    tickers = exrets.columns
    idx = exrets.index

    # Prepare containers
    store = {f"beta_{f}": pd.DataFrame(index=idx) for f in dfX.columns}
    store["alpha"] = pd.DataFrame(index=idx)
    store["n_obs"] = pd.DataFrame(index=idx)

    # prepack factors as numpy for speed
    F = dfX.values
    F = np.c_[np.ones(len(F)), F]  # add const

    for tkr in tickers:
        y = exrets[tkr].values
        betas = np.full((len(idx), F.shape[1]), np.nan)
        nobs = np.zeros(len(idx))

        for i in range(window - 1, len(idx)):
            y_win = y[i-window+1:i+1]
            X_win = F[i-window+1:i+1, :]
            if np.isnan(y_win).any() or np.isnan(X_win).any():
                continue
            XtX = X_win.T @ X_win
            if np.linalg.cond(XtX) > 1e8:
                continue
            b = np.linalg.inv(XtX) @ (X_win.T @ y_win)
            betas[i, :] = b
            nobs[i] = window

        store["alpha"][tkr] = betas[:, 0]
        for j, f in enumerate(dfX.columns, start=1):
            store[f"beta_{f}"][tkr] = betas[:, j]
        store["n_obs"][tkr] = nobs

    # Save CSVs
    for key, df in store.items():
        (REPORTS_DIR / f"rolling_{key}.csv").write_text("") if df.empty else df.to_csv(REPORTS_DIR / f"rolling_{key}.csv")
    print(f"[OK] Saved rolling betas to reports/rolling_*.csv")

    # Quick plot: average rolling market beta
    if "beta_MKT" in store:
        avg_beta = store["beta_MKT"].mean(axis=1)
        plt.figure(figsize=(7, 3))
        plt.plot(avg_beta.index, avg_beta.values)
        plt.title("Average Rolling Market Beta (36M)")
        plt.tight_layout()
        outpng = REPORTS_DIR / "rolling_beta_MKT_avg.png"
        plt.savefig(outpng, dpi=150)
        plt.close()
        print(f"[OK] Saved {outpng}")

# ------------------ main steps ------------------

def run_time_series_regressions():
    exrets, X, used_factors = load_data()

    print(f"[INFO] Sample span: {exrets.index.min().date()} → {exrets.index.max().date()}")
    print(f"[INFO] Months: {len(exrets.index):d} | Tickers: {exrets.shape[1]:d} | Factors used: {used_factors}")

    results = []
    for ticker in exrets.columns:
        y = exrets[ticker]
        df = pd.concat([y, X], axis=1, join="inner").dropna()
        if df.shape[0] < MIN_MONTHS_FOR_REGRESSION:
            continue
        y_i = df.iloc[:, 0]
        X_i = df.iloc[:, 1:]
        res = nw_ts_regression(y_i, X_i, lags=6)

        row = {
            "ticker": ticker,
            "n_obs": int(res.nobs),
            "alpha": res.params.get("const", np.nan),
            "alpha_t": res.tvalues.get("const", np.nan),
            "adj_R2": res.rsquared_adj,
        }
        for name in X_i.columns:
            if name == "const":
                continue
            row[f"beta_{name}"] = res.params.get(name, np.nan)
            row[f"t_{name}"] = res.tvalues.get(name, np.nan)
        results.append(row)

    ts_df = pd.DataFrame(results)
    out_csv = REPORTS_DIR / "factor_regression_summary.csv"
    if ts_df.empty:
        print("[WARN] No tickers had enough overlapping data for regression.")
        ts_df = pd.DataFrame(columns=["ticker","n_obs","alpha","alpha_t","adj_R2"])
        ts_df.to_csv(out_csv, index=False)
        return ts_df

    ts_df = ts_df.sort_values("adj_R2", ascending=False)
    ts_df.to_csv(out_csv, index=False)
    print(f"[OK] Saved {out_csv} (rows={len(ts_df)})")
    return ts_df

def run_fama_macbeth():
    # Needs firm-level characteristics (B/M, profitability, investment). Skipped for now.
    (REPORTS_DIR / "fama_macbeth_results.csv").write_text("")
    print("[INFO] Skipping Fama–MacBeth (needs firm-level characteristics).")

def quick_plot(ts_df: pd.DataFrame):
    if ts_df.empty:
        return
    need = [c for c in ["beta_MKT","beta_MOM"] if c in ts_df.columns]
    if len(need) == 2:
        plt.figure(figsize=(6,5))
        plt.scatter(ts_df["beta_MKT"], ts_df["beta_MOM"])
        plt.xlabel("Beta MKT")
        plt.ylabel("Beta MOM")
        plt.title("Ticker Betas: Market vs Momentum")
        plt.grid(True, alpha=0.3)
        out = REPORTS_DIR / "beta_scatter_MKT_vs_MOM.png"
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"[OK] Saved {out}")

def main():
    ts_df = run_time_series_regressions()
    run_fama_macbeth()
    quick_plot(ts_df)
    run_rolling_betas(window=36)

if __name__ == "__main__":
    main()

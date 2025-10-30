#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.stats.sandwich_covariance import cov_hac
from scipy.stats import spearmanr

BASE = Path("Investment_universe")
RET_PATH = BASE / "modeling_returns.csv"
PRC_PATH = BASE / "modeling_prices.csv"

# ---------- helpers ----------
def _cov_hac_compat(res, lags: int):
    try:
        return cov_hac(res, maxlags=lags)   # newer statsmodels
    except TypeError:
        return cov_hac(res, nlags=lags)     # older statsmodels

def nw_tstat(y: pd.Series, x: pd.Series | None = None, lags: int = 5):
    """HAC/Newey–West t-stat for the intercept."""
    y = pd.Series(y).astype(float).dropna()
    if len(y) == 0:
        return np.nan, np.nan, np.nan

    if x is None:
        X = np.ones((len(y), 1), dtype=float)  # intercept only
        X = add_constant(X, has_constant="add")
    else:
        x = pd.Series(x).astype(float)
        aligned = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
        if len(aligned) == 0 or aligned["x"].nunique() == 1:
            return np.nan, np.nan, np.nan  # no variation → undefined
        y = aligned["y"]
        X = add_constant(aligned[["x"]].to_numpy(), has_constant="add")

    res = OLS(y.to_numpy(), X, missing="drop").fit()
    S = _cov_hac_compat(res, lags)
    se = np.sqrt(np.diag(S))
    if se[0] == 0 or np.isnan(se[0]):
        return np.nan, float(res.params[0]), np.nan
    t = res.params[0] / se[0]
    return float(t), float(res.params[0]), float(se[0])

def add_row(rows, *, test, estimate=None, se=None, tstat=None, N=None, note=None):
    rows.append({
        "test": test,
        "estimate": estimate,
        "se": se,
        "tstat": tstat,
        "N": N,
        "note": note
    })

def long_short_monthly(score: pd.DataFrame, returns: pd.DataFrame, q=0.1):
    """Equal-weight L–S by month on a score DataFrame (same shape as returns)."""
    ls = []
    for dt in returns.index[12:]:
        s = score.loc[dt].dropna()
        if s.size < 20 or s.nunique() < 3:
            continue
        lo = s.quantile(q)
        hi = s.quantile(1 - q)
        long = s.index[s >= hi]
        short = s.index[s <= lo]
        if len(long) == 0 or len(short) == 0:
            continue
        r = returns.loc[dt]
        lsr = r.reindex(long).mean() - r.reindex(short).mean()
        if pd.notna(lsr):
            ls.append((dt, lsr))
    return pd.Series(dict(ls)).sort_index()

# ---------- load ----------
rets = pd.read_csv(RET_PATH, index_col=0, parse_dates=True).sort_index()
prices = pd.read_csv(PRC_PATH, index_col=0, parse_dates=True).sort_index()
tickers = list(rets.columns)

summary_rows = []

# 1) Zero-mean per-asset (daily)
lags = 5
tstats = []
for c in tickers:
    y = rets[c].dropna()
    if len(y) < 250:
        continue
    t, est, se = nw_tstat(y, lags=lags)
    if not np.isnan(t):
        tstats.append(t)

add_row(summary_rows,
        test="Zero-mean across assets (daily)",
        estimate=np.nan,
        se=np.nan,
        tstat=np.nan if len(tstats) == 0 else float(np.nanmean(tstats)),
        N=len(tstats),
        note=f"avg HAC t-stat across assets, lags={lags}")

# 2) Momentum L–S (12m, skip 1m) at monthly frequency
mret = rets.resample("ME").apply(lambda x: (1 + x).prod() - 1)  # month-end
mom12 = (1 + mret).rolling(12).apply(lambda x: np.prod(x) - 1, raw=False).shift(1)

ls_mom = long_short_monthly(mom12, mret)
if ls_mom.notna().sum() >= 12:
    t, est, se = nw_tstat(ls_mom.dropna(), lags=6)  # ~6 for monthly HAC
    add_row(summary_rows,
            test="Momentum L–S (12m, skip1m), monthly",
            estimate=est, se=se, tstat=t, N=int(ls_mom.notna().sum()),
            note="equal-weight deciles; EU financials")
else:
    add_row(summary_rows, test="Momentum L–S (12m, skip1m), monthly",
            note="Insufficient non-NaN months to test")

# 3) Fama–MacBeth daily: next-day return on 20d momentum & 20d vol
mom20d = rets.rolling(20).sum().shift(1)
vol20d = rets.rolling(20).std().shift(1)

betas = []
idx = rets.index
for i in range(60, len(idx) - 1):
    dt = idx[i]
    dt_next = idx[i + 1]
    X = pd.concat({"mom20d": mom20d.loc[dt], "vol20d": vol20d.loc[dt]}, axis=1).dropna()
    y_next = rets.loc[dt_next].reindex(X.index).dropna()
    X = X.reindex(y_next.index)
    if len(X) < 20 or X.nunique().min() <= 1:
        continue
    res = OLS(y_next.values, add_constant(X.values)).fit()
    betas.append(res.params)  # [alpha, beta_mom, beta_vol]

if betas:
    B = np.vstack(betas)
    names = ["alpha", "beta_mom20d", "beta_vol20d"]
    for j, nm in enumerate(names):
        series = pd.Series(B[:, j]).dropna()
        if len(series) >= 30 and series.nunique() > 1:
            t, est, se = nw_tstat(series, lags=10)
            add_row(summary_rows, test=f"Fama–MacBeth daily {nm}",
                    estimate=est, se=se, tstat=t, N=len(series),
                    note="HAC lags=10")
        else:
            add_row(summary_rows, test=f"Fama–MacBeth daily {nm}",
                    note="Insufficient variation/time points")

# 4) Daily Spearman IC: mom20d vs next-day return
y_next_df = rets.shift(-1)  # next trading day
ic_vals = []
for dt in rets.index[60:-1]:
    s = mom20d.loc[dt].dropna()
    if s.size < 20 or s.nunique() <= 1:
        continue
    y_next = y_next_df.loc[dt].reindex(s.index).dropna()
    s = s.reindex(y_next.index).dropna()
    if len(y_next) < 20 or s.nunique() <= 1 or y_next.nunique() <= 1:
        continue
    rho, _ = spearmanr(s, y_next)
    if pd.notna(rho):
        ic_vals.append(rho)

ic_series = pd.Series(ic_vals).dropna()
if len(ic_series) >= 30 and ic_series.nunique() > 1:
    t, est, se = nw_tstat(ic_series, lags=10)
    add_row(summary_rows, test="Spearman IC (mom20d → next-day)",
            estimate=est, se=se, tstat=t, N=len(ic_series),
            note="daily IC; HAC lags=10")
else:
    add_row(summary_rows, test="Spearman IC (mom20d → next-day)",
            note="Insufficient valid IC observations")

# ---------- output ----------
out = pd.DataFrame(summary_rows)[["test", "estimate", "se", "tstat", "N", "note"]]
print("\n=== Hypothesis Testing Summary ===")
print(out.to_string(index=False))

Path("reports").mkdir(exist_ok=True)
out_path = Path("reports/hypothesis_tests_summary.csv")
out.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")

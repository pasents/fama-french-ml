# =============================================================================
# Project note — Fama–French (EU) + Momentum  |  Script: stationarity_tests.py
# Purpose: Run ADF (null: unit root) and KPSS (null: stationary) on ALL numeric
# columns in the dataset; auto-handles date parsing and %→decimal conversion.
#
# Data files:
# - daily:   datasets/europe_ff5_plus_mom_daily.csv
# - monthly: datasets/europe_ff5_plus_mom_monthly.csv
#
# Summary of findings (daily, 1990–2025):
# - Returns MKT_RF, SMB, HML, CMA, MOM: ADF rejects; KPSS mostly not reject ⇒
#   stationary (good for regressions/backtests in returns).
# - RMW: ADF says stationary; KPSS[c] rejects but KPSS[ct] does not ⇒
#   likely trend-stationary / structural breaks.
# - RF: non-stationary (rates trend). Use excess returns or ΔRF when modeling.
#
# KPSS warnings:
# - "statistic outside look-up table" means p-value is beyond tabulated range.
#   It strengthens the decision (p is even smaller/larger than reported).
#
# Usage:
#   python stationarity_tests.py datasets/europe_ff5_plus_mom_daily.csv \
#     --out artifacts/stationarity_daily.csv
#
# Interpretation guide:
# - ADF: p < 0.05 ⇒ reject unit root ⇒ stationary.
# - KPSS: p < 0.05 ⇒ reject stationarity ⇒ non-stationary.
#
# Next steps:
# [ ] Add option to resample daily→monthly inside the script and re-test.
# [ ] Add HAC/Newey–West regressions after tests to report factor betas.
# =============================================================================

# stationarity_tests.py
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import glob, sys
from statsmodels.tsa.stattools import adfuller, kpss

# ---------- loaders ----------
def _looks_like_yyyymm(col: pd.Series) -> bool:
    s = pd.to_numeric(col, errors="coerce")
    ok = s.dropna().astype(int).astype(str).str.len().between(5, 6)
    return ok.mean() > 0.8

def load_df_smart(path_like, guess_date_cols=("date","Date","DATE","YYYYMM","yyyymm","yearmonth")):
    p = Path(path_like)
    if not p.exists():
        base = p.name
        candidates = (glob.glob(f"datasets/{base}") or
                      glob.glob(f"datasets/*{base}*") or
                      glob.glob("datasets/*ff*eu*mom*.*") or
                      glob.glob("datasets/*ff5*mom*.*"))
        if not candidates:
            raise FileNotFoundError(f"File not found: {path_like}")
        p = Path(sorted(candidates)[0])

    if p.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(p)
    else:
        df = pd.read_csv(p)

    df.columns = [c.strip() for c in df.columns]

    # parse date
    for c in guess_date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
            df = df.dropna(subset=[c]).sort_values(c).set_index(c)
            break
    else:
        yyyymm_cols = [c for c in df.columns if _looks_like_yyyymm(df[c])]
        if yyyymm_cols:
            c = yyyymm_cols[0]
            idx = pd.to_numeric(df[c], errors="coerce").astype("Int64").astype(str) + "01"
            df["__date__"] = pd.to_datetime(idx, format="%Y%m%d", errors="coerce")
            df = df.dropna(subset=["__date__"]).sort_values("__date__").set_index("__date__")
            df.drop(columns=[c], inplace=True, errors="ignore")
        else:
            try:
                df.index = pd.to_datetime(df.index, errors="coerce")
                df = df.dropna(how="all").sort_index()
            except Exception:
                pass

    # keep only numeric columns
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(how="all")
    print(f"Loaded: {p.resolve()}  | rows={len(df)}  cols={list(df.columns)}")
    return df

def maybe_percent_to_decimal(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return s/100.0 if s.dropna().abs().median() > 0.2 else s

# ---------- tests ----------
def adf_test(x, regression="c"):
    x = x.dropna().astype(float)
    stat, pval, lags, nobs, crit, _ = adfuller(x, autolag="AIC", regression=regression)
    return {"stat": stat, "pval": pval, "lags": lags, "nobs": nobs, "crit": crit, "reg": regression}

def kpss_test(x, regression="c"):
    x = x.dropna().astype(float)
    stat, pval, lags, crit = kpss(x, regression=regression, nlags="auto")
    return {"stat": stat, "pval": pval, "lags": lags, "crit": crit, "reg": regression}

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Run ADF & KPSS stationarity tests on all numeric columns.")
    ap.add_argument("csv", type=str, help="Path to CSV/XLSX (will also search ./datasets)")
    ap.add_argument("--out", default="", help="Optional path to save results CSV (e.g., artifacts/stationarity_daily.csv)")
    args = ap.parse_args()

    df = load_df_smart(args.csv)

    # convert percent-looking series to decimals
    for c in df.columns:
        df[c] = maybe_percent_to_decimal(df[c])

    results = []
    alpha = 0.05

    for col in df.columns:
        s = df[col].dropna()
        if s.empty:
            continue

        # ADF (null: non-stationary)
        for reg in ("c", "ct"):
            a = adf_test(s, regression=reg)
            verdict = "stationary" if a["pval"] < alpha else "non-stationary"
            results.append({
                "series": col, "test": f"ADF[{reg}]",
                "stat": a["stat"], "pval": a["pval"], "lags": a["lags"],
                "nobs": a["nobs"], "null": "unit root", "verdict_at_5%": verdict
            })

        # KPSS (null: stationary)
        for reg in ("c", "ct"):
            try:
                k = kpss_test(s, regression=reg)
                verdict = "non-stationary" if k["pval"] < alpha else "stationary"
                results.append({
                    "series": col, "test": f"KPSS[{reg}]",
                    "stat": k["stat"], "pval": k["pval"], "lags": k["lags"],
                    "nobs": len(s), "null": "stationary", "verdict_at_5%": verdict
                })
            except Exception as e:
                # KPSS can fail if the series is too short or near white noise with tiny variance
                results.append({
                    "series": col, "test": f"KPSS[{reg}]", "stat": np.nan, "pval": np.nan,
                    "lags": np.nan, "nobs": len(s), "null": "stationary",
                    "verdict_at_5%": f"error: {e}"
                })

    res_df = pd.DataFrame(results)
    with pd.option_context('display.float_format', '{:0.4f}'.format):
        print("\n=== Stationarity summary (α = 0.05) ===")
        print(res_df)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        res_df.to_csv(out, index=False)
        print(f"\nSaved results to: {out}")

if __name__ == "__main__":
    main()

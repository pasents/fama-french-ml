# =============================================================================
# Project note — Fama–French (EU) + Momentum  |  Script: factor_stats.py
# Purpose: Compute CAGR, Ann.Vol, Sharpe, MaxDD for MOM vs Market (daily/monthly).
#
# Data files used (examples)
# - daily:   datasets/europe_ff5_plus_mom_daily.csv
# - monthly: datasets/europe_ff5_plus_mom_monthly.csv
# Columns: ['MKT_RF','SMB','HML','RMW','CMA','RF','MOM']
#
# Key in-sample result (daily, 1990–2025, 252/yr):
# - MOM:    CAGR ~ 9.42%, Vol ~ 11.75%, Sharpe ~ 0.825, MaxDD ~ −47.5%
# - MKT_RF: CAGR ~ 4.51%, Vol ~ 17.88%, Sharpe ~ 0.336, MaxDD ~ −63.1%
# Takeaway: MOM outperforms the market (excess) in this sample.
#
# Usage (daily, market excess):
#   python factor_stats.py datasets/europe_ff5_plus_mom_daily.csv \
#     --mom-col MOM --mkt-col MKT_RF --rf-col RF --freq daily \
#     --out artifacts/mom_market_daily.csv
#
# Notes:
# - MOM is a zero-cost factor (already excess); MKT_RF is market excess.
# - If you have total market 'MARKET', subtract RF inside the script
#   (pass --mkt-col MARKET). Annualization uses 252 for daily or 12 for monthly.
#
# TODO:
# [ ] Option to also print total market (MARKET = MKT_RF + RF).
# [ ] Add comparison vs investable proxies (UCITS ETFs) with costs.
# =============================================================================

# factor_stats.py
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import glob
import sys

# ---------- metrics ----------
def cagr(returns, periods_per_year):
    if len(returns) == 0:
        return np.nan
    g = float((1.0 + returns).prod())
    years = len(returns) / periods_per_year
    return g**(1/years) - 1 if years > 0 else np.nan

def ann_vol(returns, periods_per_year):
    return returns.std(ddof=1) * np.sqrt(periods_per_year)

def sharpe(excess_returns, periods_per_year):
    mu = excess_returns.mean() * periods_per_year
    sd = excess_returns.std(ddof=1) * np.sqrt(periods_per_year)
    return mu / sd if sd > 0 else np.nan

def max_drawdown(returns):
    eq = (1 + returns).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1
    return dd.min()

# ---------- helpers ----------
def maybe_percent_to_decimal(series):
    s = pd.to_numeric(series, errors="coerce")
    # If magnitudes look like percentages, convert to decimals
    if s.dropna().abs().median() > 0.2:
        return s / 100.0
    return s

def _looks_like_yyyymm(col: pd.Series) -> bool:
    s = pd.to_numeric(col, errors="coerce")
    ok = s.dropna().astype(int).astype(str).str.len().between(5, 6)
    return ok.mean() > 0.8

def infer_freq(index: pd.DatetimeIndex) -> str:
    if len(index) < 3:
        return "auto"
    d = index.to_series().diff().dt.days.dropna()
    med = d.median()
    if med <= 2:
        return "daily"
    if 25 <= med <= 35:
        return "monthly"
    return "auto"

def periods_per_year(freq: str, index: pd.DatetimeIndex) -> int:
    if freq == "daily":
        return 252
    if freq == "monthly":
        return 12
    return 252 if infer_freq(index) == "daily" else 12

def load_data_smart(path_like: str,
                    guess_date_cols=("date","Date","DATE","YYYYMM","yyyymm","yearmonth")) -> pd.DataFrame:
    cli_path = Path(path_like)
    path = cli_path if cli_path.exists() else None

    if path is None:
        base = cli_path.name
        candidates = []
        if Path("datasets").exists():
            candidates = (
                glob.glob(f"datasets/{base}") or
                glob.glob(f"datasets/*{base}*") or
                glob.glob("datasets/*ff*eu*mom*.*") or
                glob.glob("datasets/*ff5*mom*.*")
            )
        if not candidates:
            raise FileNotFoundError(
                f"File not found: {path_like}\n"
                f"Hint: run `dir datasets` and pass the exact filename."
            )
        path = Path(sorted(candidates)[0])

    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    df.columns = [c.strip() for c in df.columns]

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
                df = df.dropna(axis=0, how="any").sort_index()
            except Exception:
                pass

    print(f"\nLoaded file: {path.resolve()}")
    print(f"Rows: {len(df)}, Columns: {list(df.columns)}")
    return df

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Stats for MOM vs market (daily or monthly FF data).")
    ap.add_argument("csv", type=str, help="Path to CSV/XLSX (script will also search ./datasets)")
    ap.add_argument("--mom-col", default="MOM", help="Momentum factor column")
    ap.add_argument("--mkt-col", default="MARKET",
                    help="Market column. Use 'MKT_RF' for market EXCESS (no RF subtraction).")
    ap.add_argument("--rf-col", default="RF", help="Risk-free column (optional)")
    ap.add_argument("--freq", default="auto", choices=["auto","daily","monthly"],
                    help="Data frequency for annualization (default: auto)")
    ap.add_argument("--out", default="", help="Optional path to save series and .stats.csv")
    args = ap.parse_args()

    df = load_data_smart(args.csv)

    # Check columns
    required = [args.mom_col, args.mkt_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print("\nERROR: Missing required columns:", missing, file=sys.stderr)
        print("Available columns:", list(df.columns), file=sys.stderr)
        if "MKT_RF" in df.columns and args.mkt_col.upper() == "MARKET":
            print("Hint: Use --mkt-col MKT_RF", file=sys.stderr)
        sys.exit(1)

    # numeric + RF handling
    rf = pd.Series(0.0, index=df.index)
    if args.rf_col in df.columns:
        rf = maybe_percent_to_decimal(df[args.rf_col]).fillna(0.0)

    mom = maybe_percent_to_decimal(df[args.mom_col]).astype(float)
    mkt = maybe_percent_to_decimal(df[args.mkt_col]).astype(float)

    # align
    idx = mom.index.intersection(mkt.index).intersection(rf.index)
    mom, mkt, rf = mom.loc[idx], mkt.loc[idx], rf.loc[idx]

    # Frequency & annualization
    freq_detected = args.freq if args.freq != "auto" else infer_freq(idx)
    ppy = periods_per_year(freq_detected, idx)
    print(f"Detected frequency: {freq_detected}  (periods/year = {ppy})")

    # Excess streams for Sharpe
    # Factors like MOM are already zero-investment → use as-is.
    mom_excess = mom.copy()

    # Market: MKT_RF is already excess; MARKET (total) needs RF subtraction.
    if args.mkt_col.upper() == "MKT_RF":
        mkt_excess = mkt.copy()
    else:
        mkt_excess = mkt - rf

    # Stats
    stats = pd.DataFrame(index=["MOM", args.mkt_col],
                         columns=["Start","End","N","CAGR","Ann.Vol","Sharpe","MaxDD"])

    for name, r, rex in [("MOM", mom, mom_excess), (args.mkt_col, mkt, mkt_excess)]:
        r = r.dropna()
        rex = rex.reindex(r.index).dropna()
        stats.loc[name, "Start"] = r.index.min().date() if len(r) else ""
        stats.loc[name, "End"]   = r.index.max().date() if len(r) else ""
        stats.loc[name, "N"]     = len(r)
        stats.loc[name, "CAGR"]  = cagr(r, ppy)
        stats.loc[name, "Ann.Vol"] = ann_vol(r, ppy)
        stats.loc[name, "Sharpe"]  = sharpe(rex, ppy)
        stats.loc[name, "MaxDD"] = max_drawdown(r)

    with pd.option_context("display.float_format", "{:0.4f}".format):
        print("\n== MOM vs Market ==")
        print(stats)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"MOM": mom, args.mkt_col: mkt, "RF": rf}).to_csv(out_path)
        stats.to_csv(out_path.with_suffix(".stats.csv"))
        print(f"\nSaved series to: {out_path}")
        print(f"Saved stats to:  {out_path.with_suffix('.stats.csv')}")

if __name__ == "__main__":
    main()

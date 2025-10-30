#!/usr/bin/env python3
"""
Build the modeling universe from QC results and filter datasets.

Inputs (defaults are overridable via CLI flags):
- Investment_universe/europe_assumption_tests.csv     (QC table with ADF/KPSS etc.)
- Investment_universe/europe_kept_tickers.csv         (tickers that passed basic data QC)
- Investment_universe/europe_returns_cleaned.csv      (panel returns, Date index)
- Investment_universe/europe_prices_cleaned.csv       (panel prices, Date index)

Outputs:
- Investment_universe/modeling_universe.csv           (final ticker list)
- Investment_universe/modeling_drop_log.csv           (why each ticker was dropped)
- Investment_universe/modeling_returns.csv            (filtered returns)
- Investment_universe/modeling_prices.csv             (filtered prices)

Usage examples:
    python build_modeling_universe.py
    python build_modeling_universe.py --force-drop CLI.L STAN.L
    python build_modeling_universe.py --force-drop-file bin/force_drop.txt
"""

from pathlib import Path
import argparse
import pandas as pd

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Build modeling universe and filter datasets.")
    p.add_argument("--base", type=Path, default=Path("Investment_universe"),
                   help="Base directory for inputs/outputs.")
    p.add_argument("--qc", type=Path, default=None,
                   help="Path to QC CSV (default: <base>/europe_assumption_tests.csv).")
    p.add_argument("--kept", type=Path, default=None,
                   help="Path to kept tickers CSV (default: <base>/europe_kept_tickers.csv).")
    p.add_argument("--returns", type=Path, default=None,
                   help="Path to returns CSV (default: <base>/europe_returns_cleaned.csv).")
    p.add_argument("--prices", type=Path, default=None,
                   help="Path to prices CSV (default: <base>/europe_prices_cleaned.csv).")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="Significance used for ADF/KPSS rule-of-thumb (ADF<=alpha & KPSS>=alpha).")
    p.add_argument("--force-drop", nargs="*", default=[],
                   help="Tickers to force-drop (space-separated).")
    p.add_argument("--force-drop-file", type=Path, default=None,
                   help="Optional path to text/CSV file with tickers to drop.")
    return p.parse_args()

# ---------- helpers ----------
def must_exist(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"❌ {label} not found: {path.resolve()}")
    return path

def coerce_ticker_col(df: pd.DataFrame, path_hint: str) -> pd.DataFrame:
    if "Ticker" in df.columns:
        return df
    if len(df.columns) == 1:
        out = df.copy(); out.columns = ["Ticker"]; return out
    for c in df.columns:
        if str(c).strip().lower() in {"ticker", "tickers", "symbol", "name"}:
            return df.rename(columns={c: "Ticker"})
    raise ValueError(f"Could not find a 'Ticker' column in {path_hint}. Columns: {list(df.columns)}")

def load_force_drop(args) -> set:
    tickers = set(t.strip() for t in args.force_drop if t.strip())
    if args.force_drop_file and args.force_drop_file.exists():
        if args.force_drop_file.suffix.lower() == ".txt":
            tickers |= {ln.strip() for ln in args.force_drop_file.read_text().splitlines() if ln.strip()}
        else:
            df = pd.read_csv(args.force_drop_file)
            for c in df.columns:
                tickers |= set(df[c].dropna().astype(str).str.strip())
    return {t for t in tickers if t}

# ---------- main ----------
def main():
    args = parse_args()
    BASE = args.base

    # Resolve defaults
    qc_path      = args.qc      or (BASE / "europe_assumption_tests.csv")
    kept_path    = args.kept    or (BASE / "europe_kept_tickers.csv")
    returns_path = args.returns or (BASE / "europe_returns_cleaned.csv")
    prices_path  = args.prices  or (BASE / "europe_prices_cleaned.csv")

    # 1) Read inputs with clear errors
    must_exist(qc_path, "QC file")
    must_exist(kept_path, "Kept tickers file")
    must_exist(returns_path, "Returns file")
    must_exist(prices_path, "Prices file")

    qc = pd.read_csv(qc_path)
    kept = pd.read_csv(kept_path)
    rets = pd.read_csv(returns_path, index_col=0)
    prices = pd.read_csv(prices_path, index_col=0)

    # Normalize QC columns
    qc.columns = [c.strip().lower() for c in qc.columns]
    if "asset" not in qc.columns:
        raise ValueError(f"'asset' column not found in QC file. Found: {list(qc.columns)}")

    # 2) Stationarity decision (ADF<=alpha & KPSS>=alpha), also respect 'stationary_pass' if present
    has_flag = "stationary_pass" in qc.columns
    adf_p   = qc.get("adf_p")
    kpss_p  = qc.get("kpss_p")

    rule = None
    if adf_p is not None and kpss_p is not None:
        rule = (adf_p <= args.alpha) & (kpss_p >= args.alpha)

    if has_flag and rule is not None:
        stationary_ok = (qc["stationary_pass"].astype(int) == 1) & rule
    elif has_flag:
        stationary_ok = (qc["stationary_pass"].astype(int) == 1)
    elif rule is not None:
        stationary_ok = rule
    else:
        raise ValueError("Cannot infer stationarity: need 'stationary_pass' or both 'adf_p' and 'kpss_p'.")

    qc["stationary_ok"] = stationary_ok.fillna(False)
    if "tested" in qc.columns:
        qc = qc[qc["tested"] == 1].copy()

    # 3) Kept tickers → ensure 'Ticker' column
    kept = coerce_ticker_col(kept, str(kept_path))
    kept["Ticker"] = kept["Ticker"].astype(str)

    # 4) Build modeling universe
    qc_assets = qc[["asset", "stationary_ok"]].copy()
    qc_assets["asset"] = qc_assets["asset"].astype(str)

    non_stat = set(qc_assets.loc[~qc_assets["stationary_ok"], "asset"])
    print(f"Found {len(non_stat)} non-stationary tickers via QC rules.")

    modeling_universe = kept[~kept["Ticker"].isin(non_stat)].copy()

    missing_in_qc = set(kept["Ticker"]) - set(qc_assets["asset"])
    if missing_in_qc:
        print(f"⚠️ {len(missing_in_qc)} kept tickers not in QC; keeping them (e.g., {sorted(list(missing_in_qc))[:6]})")

    # 5) Drop log
    drop_log = kept[kept["Ticker"].isin(non_stat)].copy()
    drop_log["reason"] = "non-stationary (ADF/KPSS and/or stationary_pass=0)"

    # 6) Optional forced drop (your old 1.py)
    forced = load_force_drop(args)
    if forced:
        before = len(modeling_universe)
        forced_now = modeling_universe[modeling_universe["Ticker"].isin(forced)].copy()
        modeling_universe = modeling_universe[~modeling_universe["Ticker"].isin(forced)].copy()
        if not forced_now.empty:
            forced_now["reason"] = "forced drop"
            drop_log = pd.concat([drop_log, forced_now], ignore_index=True)
        print(f"🔧 Forced drop: {sorted(forced)} (universe {before} → {len(modeling_universe)})")

    # 7) Filter returns & prices to modeling universe
    tickers = modeling_universe["Ticker"].tolist()
    missing_r = [t for t in tickers if t not in rets.columns]
    missing_p = [t for t in tickers if t not in prices.columns]
    if missing_r:
        print(f"⚠️ {len(missing_r)} modeling tickers missing in returns (e.g., {missing_r[:6]})")
    if missing_p:
        print(f"⚠️ {len(missing_p)} modeling tickers missing in prices  (e.g., {missing_p[:6]})")

    keep_r = [t for t in tickers if t in rets.columns]
    keep_p = [t for t in tickers if t in prices.columns]
    rets_f = rets[keep_r].copy()
    prices_f = prices[keep_p].copy()

    # 8) Save outputs
    (BASE / "modeling_universe.csv").write_text("")  # ensure folder exists on OneDrive race
    modeling_universe.to_csv(BASE / "modeling_universe.csv", index=False)
    drop_log.to_csv(BASE / "modeling_drop_log.csv", index=False)
    rets_f.to_csv(BASE / "modeling_returns.csv")
    prices_f.to_csv(BASE / "modeling_prices.csv")

    # 9) Summary
    print("\n=== Summary ===")
    print(f"Universe size: {len(modeling_universe)} tickers")
    print(f"Returns shape: {rets_f.shape} | Prices shape: {prices_f.shape}")
    if not drop_log.empty:
        why = drop_log["reason"].value_counts().to_dict()
        print(f"Dropped: {len(drop_log)} ({why})")
    else:
        print("Dropped: 0")

if __name__ == "__main__":
    main()

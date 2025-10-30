# clean_europe_universe.py
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------- DEFAULT PATHS (overridable via CLI) ----------
RETURNS_CSV_DEF = Path("europe_returns.csv")
PRICES_CSV_DEF  = Path("europe_prices.csv")
DIAG_CSV_DEF    = Path("europe_asset_diagnostics.csv")
OUTDIR_DEF      = Path("Investment_universe")   # <— default outdir

# ---------- EXPLICIT BAD TICKERS (from notes) ----------
DROP_ALWAYS = {
    # absurd/scaling error
    "SBB-B.ST",
    # huge tails / JB
    "API.L", "PHLL.L", "JUST.L", "IGG.L",
    "CRE.L", "LIT.L", "WJG.L", "DOXA.ST",
    "BMPS.MI", "VANQ.L", "ALCBI.PA",
}

# ---------- AUTO DROP RULES ----------
AUTO_RULES = {
    "kurtosis>=50":           lambda df: df["kurtosis"] >= 50,
    "abs(skew)>=10":          lambda df: df["skew"].abs() >= 10,
    "max_drawdown<=-0.95":    lambda df: df["max_drawdown"] <= -0.95,
    "outliers_gt6sigma>=100": lambda df: df["outliers_gt6sigma"] >= 100,
}

# ---------- CLI ----------
def sha1_of_file(p: Path) -> str:
    if not p or not Path(p).exists():
        return ""
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

parser = argparse.ArgumentParser(description="Clean Europe returns/prices by dropping broken tickers.")
parser.add_argument("--returns", type=str, default=str(RETURNS_CSV_DEF), help="Path to returns CSV.")
parser.add_argument("--prices",  type=str, default=str(PRICES_CSV_DEF),  help="Path to prices CSV (optional).")
parser.add_argument("--diag",    type=str, default=str(DIAG_CSV_DEF),    help="Path to diagnostics CSV (optional).")
parser.add_argument("--outdir",  type=str, default=str(OUTDIR_DEF),      help="Output directory (default: Investment_universe).")

# New knobs you’ll actually tweak
parser.add_argument("--min-obs",      type=int,   default=250,  help="Min non-NaN observations per ticker.")
parser.add_argument("--max-missing",  type=float, default=0.20, help="Max fraction of NaNs per ticker.")
parser.add_argument("--winsorize",    type=float, default=0.0,  help="Two-sided winsorize percent (e.g. 0.5).")
parser.add_argument("--blacklist-file", type=str, default="",   help="CSV (1 col) of extra tickers to drop.")
parser.add_argument("--whitelist-file", type=str, default="",   help="CSV (1 col) of tickers to force-keep.")
parser.add_argument("--dry-run", action="store_true", help="Do not write outputs; only print summary.")

args = parser.parse_args()

RETURNS_CSV = Path(args.returns)
PRICES_CSV  = Path(args.prices)
DIAG_CSV    = Path(args.diag)
OUTDIR      = Path(args.outdir)
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_RETURNS = OUTDIR / "europe_returns_cleaned.csv"
OUT_PRICES  = OUTDIR / "europe_prices_cleaned.csv"
OUT_DROPS   = OUTDIR / "europe_dropped_tickers.csv"
OUT_KEPT    = OUTDIR / "europe_kept_tickers.csv"
OUT_SUMMARY = OUTDIR / "europe_clean_summary.json"

# ---------- HELPERS ----------
def _read_df(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path} ({name})")
    return pd.read_csv(path, index_col=0)

def ensure_wide(df: pd.DataFrame) -> pd.DataFrame:
    # Try to make sure index is datetime (ok if it already is)
    if not np.issubdtype(df.index.dtype, np.datetime64):
        try:
            df.index = pd.to_datetime(df.index, errors="raise", utc=False)
        except Exception:
            pass
    return df

def winsorize_df(df: pd.DataFrame, pct: float) -> pd.DataFrame:
    if pct <= 0:
        return df
    lo = df.quantile(pct / 100.0, numeric_only=True)
    hi = df.quantile(1 - pct / 100.0, numeric_only=True)
    return df.clip(lower=lo, upper=hi, axis="columns")

def compute_diag_from_returns(returns: pd.DataFrame) -> pd.DataFrame:
    R = returns.apply(pd.to_numeric, errors="coerce")
    skew = R.skew(axis=0, skipna=True)
    kurt = R.kurtosis(axis=0, skipna=True)  # Fisher (normal=0)
    mu = R.mean(axis=0, skipna=True)
    sd = R.std(axis=0, skipna=True)
    z = (R - mu) / sd.replace(0, np.nan)
    out6 = (z.abs() > 6).sum(axis=0)
    prices_like = (1 + R.fillna(0)).cumprod()
    roll_max = prices_like.cummax()
    dd = (prices_like / roll_max - 1.0).min(axis=0)
    return pd.DataFrame({
        "asset": R.columns,
        "skew": skew.values,
        "kurtosis": kurt.values,
        "outliers_gt6sigma": out6.values,
        "max_drawdown": dd.values,
    })

def load_one_col_csv(path_str: str) -> set:
    p = Path(path_str)
    if not path_str or not p.exists():
        return set()
    s = pd.read_csv(p, header=None).iloc[:, 0].astype(str)
    return set(s)

# ---------- LOAD RETURNS (required) ----------
returns = ensure_wide(_read_df(RETURNS_CSV, "returns"))

# ---------- PRE-FILTERS (missingness/length) ----------
nn = returns.notna().sum()
n_rows = returns.shape[0]
missing_frac = 1 - (nn / n_rows)
bad_missing = set(missing_frac[missing_frac > args.max_missing].index)
bad_len = set(nn[nn < args.min_obs].index)
pre_drop = bad_missing | bad_len
if pre_drop:
    print(f"Pre-filter: dropping {len(pre_drop)} tickers (missing/length).")
    returns = returns.drop(columns=sorted(pre_drop), errors="ignore")

# ---------- OPTIONAL WINSORIZATION FOR DIAGNOSTICS ----------
returns_for_diag = winsorize_df(returns, args.winsorize) if args.winsorize > 0 else returns

# ---------- LOAD/BUILD DIAGNOSTICS ----------
if DIAG_CSV.exists():
    diag = _read_df(DIAG_CSV, "diagnostics").copy()
    if "asset" not in diag.columns:
        if diag.index.name in (None, "asset"):
            diag = diag.reset_index().rename(columns={"index": "asset"})
        else:
            diag = diag.reset_index(names="asset")
    needed = {"asset", "kurtosis", "skew", "max_drawdown", "outliers_gt6sigma"}
    missing_cols = needed - set(diag.columns)
    if missing_cols:
        diag_from_r = compute_diag_from_returns(returns_for_diag)
        diag = pd.merge(diag, diag_from_r, on="asset", how="outer", suffixes=("", "_ret"))
        for c in ["kurtosis", "skew", "max_drawdown", "outliers_gt6sigma"]:
            if c in missing_cols:
                diag[c] = diag[f"{c}_ret"]
else:
    print("No diagnostics CSV found. Computing diagnostics from returns...")
    diag = compute_diag_from_returns(returns_for_diag)

# ---------- BUILD DROP SET ----------
hits_by_rule = {}
mask_any = pd.Series(False, index=diag.index)
diag_filled = diag.copy()
for col in ["kurtosis", "skew", "max_drawdown", "outliers_gt6sigma"]:
    if col not in diag_filled:
        diag_filled[col] = 0

for rule_name, rule_fn in AUTO_RULES.items():
    hit_mask = rule_fn(diag_filled.fillna(0))
    hits = diag_filled.loc[hit_mask, "asset"].dropna().astype(str).tolist()
    hits_by_rule[rule_name] = hits
    mask_any |= hit_mask

auto_hits = set(diag_filled.loc[mask_any, "asset"].dropna().astype(str))

blacklist = load_one_col_csv(args.blacklist_file)
whitelist = load_one_col_csv(args.whitelist_file)

drop_set = (set(DROP_ALWAYS) | auto_hits | pre_drop | blacklist) - whitelist

# ---------- LOAD PRICES (optional) ----------
prices_found = True
try:
    prices = ensure_wide(_read_df(PRICES_CSV, "prices"))
except FileNotFoundError:
    prices_found = False
    prices = None
    print(f"Warning: {PRICES_CSV} not found. Will output returns only.")

# ---------- APPLY DROPS ----------
drop_returns = sorted(set(returns.columns).intersection(drop_set))
returns_clean = returns.drop(columns=drop_returns, errors="ignore")

if prices_found:
    drop_prices = sorted(set(prices.columns).intersection(drop_set))
    prices_clean = prices.drop(columns=drop_prices, errors="ignore")

# ---------- REPORTS ----------
report_rows = []
diag_keyed = diag.set_index("asset")
for t in sorted(drop_set):
    row = {
        "asset": t,
        "reason": "explicit_list" if t in DROP_ALWAYS else (
            "pre_filter" if t in pre_drop else ("blacklist" if t in blacklist else "auto_rules")
        ),
        "auto_rules_triggered": "|".join([r for r, hits in hits_by_rule.items() if t in hits])
    }
    if t in diag_keyed.index:
        d = diag_keyed.loc[t]
        for c in ["kurtosis", "skew", "max_drawdown", "outliers_gt6sigma"]:
            row[c] = float(d[c]) if c in d and pd.notna(d[c]) else np.nan
    report_rows.append(row)

kept = sorted([c for c in returns.columns if c not in drop_set])

summary = {
    "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
    "inputs": {
        "returns": str(RETURNS_CSV), "returns_sha1": sha1_of_file(RETURNS_CSV),
        "prices":  str(PRICES_CSV),  "prices_sha1":  sha1_of_file(PRICES_CSV),
        "diag":    str(DIAG_CSV),    "diag_sha1":    sha1_of_file(DIAG_CSV)
    },
    "params": {
        "outdir": str(OUTDIR),
        "min_obs": args.min_obs,
        "max_missing": args.max_missing,
        "winsorize_pct": args.winsorize,
        "auto_rules": list(AUTO_RULES.keys()),
        "blacklist_file": args.blacklist_file,
        "whitelist_file": args.whitelist_file,
        "dry_run": bool(args.dry_run),
    },
    "counts": {
        "n_input_returns_cols": int(returns.shape[1]),
        "n_dropped": int(len(drop_set)),
        "n_kept": int(len(kept)),
    },
    "rule_hit_counts": {r: len(h) for r, h in hits_by_rule.items()}
}

# ---------- WRITE OR DRY RUN ----------
print("=== Cleaning Summary ===")
print(f"Explicit drops: {len(DROP_ALWAYS)}")
print(f"Auto-rule drops: {len(auto_hits)}")
print(f"Pre-filter drops: {len(pre_drop)}")
print(f"Blacklist drops: {len(blacklist)}")
print(f"Whitelist keeps: {len(whitelist)}")
print(f"Total unique dropped: {len(drop_set)}")
print(f"Returns: {returns.shape[1]} -> {returns_clean.shape[1]} columns")
if prices_found:
    print(f"Prices : {prices.shape[1]}  -> {prices_clean.shape[1]} columns")
else:
    print("Prices : skipped (source file not found)")
print("\nAuto-rules hit counts:")
for r, hits in summary["rule_hit_counts"].items():
    print(f" - {r}: {hits}")

if args.dry_run:
    print("\n[DRY RUN] No files written.")
else:
    pd.DataFrame(report_rows).to_csv(OUT_DROPS, index=False)
    pd.Series(kept, name="asset").to_csv(OUT_KEPT, index=False)
    returns_clean.to_csv(OUT_RETURNS)
    if prices_found:
        prices_clean.to_csv(OUT_PRICES)
    with open(OUT_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nWrote:")
    print(f" - {OUT_RETURNS}")
    if prices_found:
        print(f" - {OUT_PRICES}")
    print(f" - {OUT_DROPS}")
    print(f" - {OUT_KEPT}")
    print(f" - {OUT_SUMMARY}")

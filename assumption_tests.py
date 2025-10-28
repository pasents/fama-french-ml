# assumption_tests.py
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd

# statsmodels
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch


def parse_args():
    p = argparse.ArgumentParser(
        description="Run ADF, KPSS, Ljung-Box, and ARCH LM tests on panel of returns."
    )
    p.add_argument(
        "--returns",
        type=Path,
        default=Path("Investment_universe/europe_returns_cleaned.csv"),
        help="Path to cleaned returns CSV (dates as index, tickers as columns).",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("Investment_universe"),
        help="Directory to write reports into.",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for pass/fail.",
    )
    p.add_argument(
        "--min-obs",
        type=int,
        default=250,
        help="Minimum non-NaN observations required for a series to be tested.",
    )
    p.add_argument(
        "--max-missing",
        type=float,
        default=0.20,
        help="Maximum allowed missing fraction (0-1). Series with more missing are skipped.",
    )
    p.add_argument(
        "--lb-lags",
        type=str,
        default="5,10,20",
        help="Comma-separated Ljung-Box lags, e.g. '5,10,20'.",
    )
    p.add_argument(
        "--arch-lag",
        type=int,
        default=5,
        help="Max lag to test in ARCH LM (het_arch).",
    )
    p.add_argument(
        "--kpss-reg",
        type=str,
        default="c",
        choices=["c", "ct"],
        help="KPSS regression: 'c' (level) or 'ct' (trend).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute tests and print summary without writing files.",
    )
    return p.parse_args()


def safe_adf(x):
    try:
        res = adfuller(x, autolag="AIC", regression="c")  # (stat, pval, lags, nobs, crit, icbest)
        return float(res[1]), float(res[0])
    except Exception:
        return np.nan, np.nan


def safe_kpss(x, regression="c"):
    try:
        stat, pval, *_ = kpss(x, regression=regression, nlags="auto")
        return float(pval), float(stat)
    except Exception:
        return np.nan, np.nan


def safe_lb(x, lags):
    try:
        out = acorr_ljungbox(x, lags=lags, return_df=True)
        # return dict of lag -> pval
        return {int(k): float(v) for k, v in out["lb_pvalue"].to_dict().items()}
    except Exception:
        return {int(l): np.nan for l in lags}


def safe_arch(x, maxlag=5):
    try:
        stat, pval, *_ = het_arch(x, maxlag=maxlag)
        return float(pval), float(stat)
    except Exception:
        return np.nan, np.nan


def series_quality_mask(s: pd.Series, min_obs: int, max_missing: float) -> bool:
    n = s.size
    n_nan = s.isna().sum()
    if n - n_nan < min_obs:
        return False
    if n_nan / max(n, 1) > max_missing:
        return False
    return True


def main():
    args = parse_args()
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    lags = [int(x.strip()) for x in args.lb_lags.split(",") if x.strip()]

    # Load returns (wide)
    df = pd.read_csv(args.returns, index_col=0)
    # Try to parse date index (not strictly required)
    try:
        df.index = pd.to_datetime(df.index)
    except Exception:
        pass

    rows = []
    tested = 0
    skipped = 0

    for col in df.columns:
        x = pd.to_numeric(df[col], errors="coerce")
        if not series_quality_mask(x, args.min_obs, args.max_missing):
            skipped += 1
            rows.append(
                {
                    "asset": col,
                    "tested": 0,
                    "reason_skipped": "quality_filter",
                    "adf_p": np.nan,
                    "adf_stat": np.nan,
                    "kpss_p": np.nan,
                    "kpss_stat": np.nan,
                    **{f"lb_p_lag{L}": np.nan for L in lags},
                    "arch_p": np.nan,
                    "arch_stat": np.nan,
                    "stationary_pass": np.nan,
                    "lb_pass": np.nan,
                    "arch_pass": np.nan,
                    "overall_pass": np.nan,
                    "n_obs": int(x.notna().sum()),
                    "missing_frac": float(x.isna().mean()),
                }
            )
            continue

        x = x.dropna()
        tested += 1

        # Tests
        adf_p, adf_stat = safe_adf(x)
        kpss_p, kpss_stat = safe_kpss(x, regression=args.kpss_reg)
        lb_pvals = safe_lb(x, lags)
        arch_p, arch_stat = safe_arch(x, maxlag=args.arch_lag)

        # Interpret:
        # Stationary if ADF rejects (p<alpha) AND KPSS does not reject (p>alpha)
        stationary_pass = (adf_p < args.alpha) and (kpss_p > args.alpha)

        # No serial correlation if ALL Ljung-Box p-values > alpha
        lb_pass = all((p > args.alpha) for p in lb_pvals.values())

        # No ARCH effects if ARCH LM p > alpha
        arch_pass = (arch_p > args.alpha)

        overall_pass = int(stationary_pass and lb_pass and arch_pass)

        row = {
            "asset": col,
            "tested": 1,
            "reason_skipped": "",
            "adf_p": adf_p,
            "adf_stat": adf_stat,
            "kpss_p": kpss_p,
            "kpss_stat": kpss_stat,
            **{f"lb_p_lag{L}": lb_pvals.get(L, np.nan) for L in lags},
            "arch_p": arch_p,
            "arch_stat": arch_stat,
            "stationary_pass": int(stationary_pass) if not np.isnan(adf_p) and not np.isnan(kpss_p) else np.nan,
            "lb_pass": int(lb_pass) if not any(np.isnan(list(lb_pvals.values()))) else np.nan,
            "arch_pass": int(arch_pass) if not np.isnan(arch_p) else np.nan,
            "overall_pass": overall_pass if overall_pass in (0, 1) else np.nan,
            "n_obs": int(x.size),
            "missing_frac": float(df[col].isna().mean()),
        }
        rows.append(row)

    report = pd.DataFrame(rows).set_index("asset").sort_index()

    # Summary
    tested_df = report[report["tested"] == 1]
    counts = {
        "n_columns": int(df.shape[1]),
        "n_tested": int(tested),
        "n_skipped_quality": int(skipped),
        "stationary_pass": int(tested_df["stationary_pass"].sum(skipna=True)) if not tested_df.empty else 0,
        "lb_pass": int(tested_df["lb_pass"].sum(skipna=True)) if not tested_df.empty else 0,
        "arch_pass": int(tested_df["arch_pass"].sum(skipna=True)) if not tested_df.empty else 0,
        "overall_pass": int(tested_df["overall_pass"].sum(skipna=True)) if not tested_df.empty else 0,
    }

    if args.dry_run:
        print("=== Assumption Tests (dry-run) ===")
        print(f"Inputs: {args.returns}")
        print(f"Alpha={args.alpha}, LB lags={lags}, ARCH lag={args.arch_lag}, KPSS reg='{args.kpss_reg}'")
        print(f"Columns: {df.shape[1]} | Tested: {tested} | Skipped (quality): {skipped}")
        print(f"Overall pass (all tests): {counts['overall_pass']} out of {tested}")
        return

    # Write outputs
    out_csv = outdir / "europe_assumption_tests.csv"
    out_json = outdir / "europe_assumption_summary.json"
    report.to_csv(out_csv)

    payload = {
        "alpha": args.alpha,
        "lb_lags": lags,
        "arch_lag": args.arch_lag,
        "kpss_reg": args.kpss_reg,
        "min_obs": args.min_obs,
        "max_missing": args.max_missing,
        "returns": str(args.returns),
        "out_csv": str(out_csv),
        "counts": counts,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("=== Assumption Tests ===")
    print(f"Tested {tested} series (skipped {skipped} for quality). Alpha={args.alpha}")
    print(f"Overall pass (all tests): {counts['overall_pass']} / {tested}")
    print(f"Wrote:\n - {out_csv}\n - {out_json}")
    print("Tip: use Newey–West (HAC) standard errors in any cross-sectional/time-series regressions regardless.")


if __name__ == "__main__":
    main()

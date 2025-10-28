# update_readme.py
"""
Auto-generate/refresh README sections from pipeline JSON & CSV outputs.

Usage (Windows / PowerShell):
  python update_readme.py
  python update_readme.py --root Investment_universe --readme README.md
  python update_readme.py --dry-run   # prints markdown instead of writing

It replaces content between these markers in README.md:
  <!-- AUTO-SUMMARY:START -->
  <!-- AUTO-SUMMARY:END -->
If README.md doesn't exist, it will create one.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import datetime as dt
import pandas as pd

MARK_START = "<!-- AUTO-SUMMARY:START -->"
MARK_END   = "<!-- AUTO-SUMMARY:END -->"

def load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None

def load_csv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return None

def fmt_pct(x, digits=1):
    try:
        return f"{100*float(x):.{digits}f}%"
    except Exception:
        return "n/a"

def build_markdown(root: Path) -> str:
    # --- inputs
    clean_json  = load_json(root / "europe_clean_summary.json")
    assume_json = load_json(root / "europe_assumption_summary.json")  # optional if you named it so
    if assume_json is None:
        # fallback: some runs only have the CSV with a separate config dump printed to console
        assume_json = {}

    drops_csv    = load_csv(root / "europe_dropped_tickers.csv")
    kept_csv     = load_csv(root / "europe_kept_tickers.csv")
    tests_csv    = load_csv(root / "europe_assumption_tests.csv")

    # --- time
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    # --- cleaning summary
    if clean_json:
        n_in   = clean_json["counts"]["n_input_returns_cols"]
        n_drop = clean_json["counts"]["n_dropped"]
        n_keep = clean_json["counts"]["n_kept"]
        rule_hits = clean_json.get("rule_hit_counts", {})
        params = clean_json.get("params", {})
        returns_path = clean_json["inputs"].get("returns", "n/a")
        prices_path  = clean_json["inputs"].get("prices",  "n/a")
        outdir       = params.get("outdir", str(root))
    else:
        n_in = n_drop = n_keep = 0
        rule_hits = {}
        returns_path = prices_path = "n/a"
        outdir = str(root)

    # --- assumption summary
    if assume_json:
        ac = assume_json.get("counts", {})
        n_cols = ac.get("n_columns", 0)
        n_tested = ac.get("n_tested", 0)
        stationary_pass = ac.get("stationary_pass", 0)
        lb_pass = ac.get("lb_pass", 0)
        arch_pass = ac.get("arch_pass", 0)

        alpha      = assume_json.get("alpha", 0.05)
        lb_lags    = assume_json.get("lb_lags", [5,10,20])
        arch_lag   = assume_json.get("arch_lag", 5)
        kpss_reg   = assume_json.get("kpss_reg", "c")
        min_obs    = assume_json.get("min_obs", 250)
        max_missing= assume_json.get("max_missing", 0.2)
    else:
        n_cols = n_tested = stationary_pass = lb_pass = arch_pass = 0
        alpha = 0.05; lb_lags=[5,10,20]; arch_lag=5; kpss_reg="c"; min_obs=250; max_missing=0.2

    # --- top offenders (from tests CSV)
    top_autocorr = []
    top_arch = []
    nonstationary = []
    if tests_csv is not None and not tests_csv.empty:
        # Expect columns like: asset, adf_p, kpss_p, lb_p_lag5, lb_p_lag10, lb_p_lag20, arch_p, ...
        alpha_ = float(alpha)

        if {"asset","adf_p","kpss_p"}.issubset(tests_csv.columns):
            nonstationary = tests_csv.loc[
                (tests_csv["adf_p"] > alpha_) | (tests_csv["kpss_p"] < alpha_), "asset"
            ].dropna().astype(str).tolist()

        # autocorr: any LB p < alpha
        lb_cols = [c for c in tests_csv.columns if c.startswith("lb_p_lag")]
        if lb_cols:
            lb_any = tests_csv.assign(lb_any = tests_csv[lb_cols].min(axis=1))
            top_autocorr = (
                lb_any.sort_values("lb_any")
                     .loc[lb_any["lb_any"] < alpha_, ["asset","lb_any"]]
                     .head(10)
                     .values.tolist()
            )

        # arch: p < alpha
        if "arch_p" in tests_csv.columns:
            top_arch = (
                tests_csv.sort_values("arch_p")
                         .loc[tests_csv["arch_p"] < alpha_, ["asset","arch_p"]]
                         .head(10)
                         .values.tolist()
            )

    # --- dropped tickers overview
    drop_rows = []
    if drops_csv is not None and not drops_csv.empty:
        cols = [c for c in ["asset","reason","auto_rules_triggered","kurtosis","skew","max_drawdown","outliers_gt6sigma"] if c in drops_csv.columns]
        preview = drops_csv[cols].head(12)
        drop_rows = preview.values.tolist()

    # --- kept size
    kept_count = int(n_keep or (0 if kept_csv is None else kept_csv.shape[0]))

    # --- build markdown
    md = []
    md.append(MARK_START)
    md.append("")
    md.append(f"_Last updated: **{ts}**_")
    md.append("")
    md.append("## Cleaning Summary")
    md.append("")
    md.append(f"- Input returns file: `{returns_path}`")
    md.append(f"- Input prices file:  `{prices_path}`")
    md.append(f"- Output directory:   `{outdir}`")
    md.append("")
    md.append(f"- Tickers in: **{n_in}**  → Dropped: **{n_drop}**  → Kept: **{kept_count}**")
    if rule_hits:
        md.append("")
        md.append("**Auto-rule hit counts:**")
        for k,v in rule_hits.items():
            md.append(f"- `{k}`: **{v}**")
    md.append("")
    md.append("Key outputs:")
    md.append(f"- `{root / 'europe_returns_cleaned.csv'}`")
    md.append(f"- `{root / 'europe_prices_cleaned.csv'}`")
    md.append(f"- `{root / 'europe_dropped_tickers.csv'}`")
    md.append(f"- `{root / 'europe_kept_tickers.csv'}`")
    md.append("")
    md.append("## Econometric Assumptions")
    md.append("")
    md.append(f"- Tested tickers: **{n_tested}/{n_cols}**")
    md.append(f"- Stationarity (ADF≤α & KPSS≥α): **{stationary_pass}** passed")
    md.append(f"- Ljung–Box (lags {lb_lags}, α={alpha}): **{lb_pass}** passed at all lags")
    md.append(f"- ARCH LM (lag {arch_lag}, α={alpha}): **{arch_pass}** passed (note: ARCH is expected for returns)")
    md.append("")
    md.append("Parameters:")
    md.append(f"- min_obs = `{min_obs}`, max_missing = `{max_missing}`")
    md.append(f"- KPSS reg = `{kpss_reg}`")
    md.append("")
    if nonstationary:
        md.append("<details><summary>Non-stationary tickers (by ADF/KPSS)</summary>")
        md.append("")
        md.append(", ".join(sorted(set(nonstationary))) or "_none_")
        md.append("")
        md.append("</details>")
        md.append("")
    if top_autocorr:
        md.append("<details><summary>Strongest autocorrelation offenders (lowest LB p)</summary>")
        md.append("")
        md.append("| Ticker | min LB p |")
        md.append("|---|---:|")
        for a,p in top_autocorr:
            md.append(f"| {a} | {p:.3g} |")
        md.append("")
        md.append("</details>")
        md.append("")
    if top_arch:
        md.append("<details><summary>Strongest ARCH offenders (lowest ARCH p)</summary>")
        md.append("")
        md.append("| Ticker | ARCH p |")
        md.append("|---|---:|")
        for a,p in top_arch:
            md.append(f"| {a} | {p:.3g} |")
        md.append("")
        md.append("</details>")
        md.append("")

    if drop_rows:
        md.append("<details><summary>Preview of dropped tickers</summary>")
        md.append("")
        md.append("| Asset | Reason | Rules | Kurtosis | Skew | MaxDD | >6σ |")
        md.append("|---|---|---|---:|---:|---:|---:|")
        for row in drop_rows:
            # Row order depends on available columns, normalize with safe get
            row_map = {k:v for k,v in zip(drops_csv.columns, row)}
            md.append(
                f"| {row_map.get('asset','')} | "
                f"{row_map.get('reason','')} | "
                f"{row_map.get('auto_rules_triggered','')} | "
                f"{row_map.get('kurtosis','')} | "
                f"{row_map.get('skew','')} | "
                f"{row_map.get('max_drawdown','')} | "
                f"{row_map.get('outliers_gt6sigma','')} |"
            )
        md.append("")
        md.append("</details>")
        md.append("")

    md.append(MARK_END)
    md.append("")
    return "\n".join(md)

def write_or_patch_readme(readme_path: Path, block: str, dry_run: bool=False):
    block = block.strip("\n")
    if dry_run:
        print(block)
        return

    if not readme_path.exists():
        # Create a minimal README
        readme_text = f"# European Equity Universe\n\n{block}\n"
        readme_path.write_text(readme_text, encoding="utf-8")
        print(f"Created {readme_path}")
        return

    text = readme_path.read_text(encoding="utf-8")
    if MARK_START in text and MARK_END in text:
        pre = text.split(MARK_START)[0]
        post = text.split(MARK_END)[-1]
        new_text = f"{pre}{block}\n{post}"
    else:
        # Append block at end
        new_text = f"{text.rstrip()}\n\n{block}\n"

    readme_path.write_text(new_text, encoding="utf-8")
    print(f"Updated {readme_path}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="Investment_universe", help="Folder containing JSON/CSV outputs")
    p.add_argument("--readme", default="README.md", help="README file to update")
    p.add_argument("--dry-run", action="store_true", help="Print markdown instead of writing README")
    args = p.parse_args()

    root = Path(args.root)
    readme = Path(args.readme)

    md_block = build_markdown(root)
    write_or_patch_readme(readme, md_block, dry_run=args.dry_run)

if __name__ == "__main__":
    main()

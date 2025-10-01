import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --------- config ---------
DEFAULT_MONTHLY = "datasets/europe_ff5_plus_mom_monthly.csv"
DEFAULT_DAILY   = "datasets/europe_ff5_plus_mom_daily.csv"
FIGDIR          = Path("figs")
ROLL_MONTHS     = 36   # for monthly rolling stats
ROLL_DAYS       = 252  # for daily rolling stats

# --------- helpers ---------
def ensure_cols(df):
    # Uppercase & canonical names
    cols = [c.strip().upper() for c in df.columns]
    df.columns = cols
    # Accept common aliases
    alias = {"MKT-RF":"MKT_RF","MKT_RF":"MKT_RF","SMB":"SMB","HML":"HML","RMW":"RMW","CMA":"CMA","RF":"RF","MOM":"MOM","UMD":"MOM","WML":"MOM"}
    df = df.rename(columns={c: alias.get(c, c) for c in df.columns})
    return df

def growth_of_1(returns):
    return (1.0 + returns.fillna(0)).cumprod()

def drawdown(series):
    nav  = (1+series.fillna(0)).cumprod()
    peak = nav.cummax()
    return nav/peak - 1.0

def rolling_sharpe(r, window, periods_per_year):
    mu  = r.rolling(window).mean()
    sig = r.rolling(window).std()
    return np.sqrt(periods_per_year) * (mu / sig)

def corr_heatmap(ax, C, xticklabels, yticklabels, title):
    im = ax.imshow(C, origin="lower", aspect="auto")
    ax.set_xticks(np.arange(len(xticklabels)))
    ax.set_xticklabels(xticklabels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(yticklabels)))
    ax.set_yticklabels(yticklabels)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

def savefig(name):
    FIGDIR.mkdir(exist_ok=True, parents=True)
    plt.tight_layout()
    plt.savefig(FIGDIR / name, dpi=150)
    plt.close()

def data_health_report(df, freq_label):
    print("\n=== Data health report ("+freq_label+") ===")
    print("Date range:", df.index.min().date(), "→", df.index.max().date())
    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    miss = df.isna().sum()
    if miss.any():
        print("Missing values (per column):")
        print(miss[miss>0].to_string())
    else:
        print("Missing values: none")
    print("\nSummary stats (monthly/daily returns in decimals):")
    print(df.describe(percentiles=[0.01,0.05,0.5,0.95,0.99]).to_string())
    print("="*60)

# --------- main viz ---------
def visualize(path, freq="monthly"):
    # load
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    df = ensure_cols(df)

    # choose roll window & annualization base
    is_monthly = (freq.lower()=="monthly")
    roll_win = ROLL_MONTHS if is_monthly else ROLL_DAYS
    ann_base = 12 if is_monthly else 252

    # market total return (MARKET = RF + MKT_RF), if available
    if "MKT_RF" in df.columns and "RF" in df.columns:
        df["MARKET"] = df["MKT_RF"] + df["RF"]

    # factors to plot (exclude RF to keep visuals clean)
    factors = [c for c in df.columns if c not in ("RF",)]
    # prefer to show MARKET if present; otherwise show MKT_RF
    anchor = "MARKET" if "MARKET" in df.columns else ("MKT_RF" if "MKT_RF" in df.columns else factors[0])

    # 0) health report
    data_health_report(df[factors], freq)

    # 1) Monthly/Daily returns (timeseries)
    ax = df[factors].plot(figsize=(11,6), linewidth=0.8)
    ax.set_title(f"Europe FF Factors ({freq}) — returns")
    ax.set_ylabel("Return")
    ax.grid(True, alpha=0.3)
    savefig(f"eu_ff_{freq}_returns_timeseries.png")

    # 2) Growth of $1
    ax = growth_of_1(df[factors]).plot(figsize=(11,6))
    ax.set_title(f"Europe FF Factors ({freq}) — growth of $1")
    ax.set_ylabel("NAV")
    ax.grid(True, alpha=0.3)
    savefig(f"eu_ff_{freq}_growth.png")

    # 3) Rolling Sharpe
    rs = rolling_sharpe(df[factors], roll_win, ann_base)
    ax = rs.plot(figsize=(11,6))
    win_label = f"{roll_win}m" if is_monthly else f"{roll_win}d"
    ax.set_title(f"Rolling Sharpe ({win_label}) — annualized")
    ax.set_ylabel("Sharpe")
    ax.axhline(0.0, lw=1, alpha=0.6)
    ax.grid(True, alpha=0.3)
    savefig(f"eu_ff_{freq}_rolling_sharpe.png")

    # 4) Static correlation heatmap
    C = df[factors].corr().values
    fig, ax = plt.subplots(figsize=(8,6))
    corr_heatmap(ax, C, factors, factors, f"Correlation heatmap ({freq})")
    savefig(f"eu_ff_{freq}_corr_heatmap.png")

    # 5) Rolling correlation vs anchor
    others = [c for c in factors if c != anchor]
    roll_corr = pd.concat(
        {c: df[c].rolling(roll_win).corr(df[anchor]) for c in others}, axis=1
    )
    ax = roll_corr.plot(figsize=(11,6))
    ax.set_title(f"Rolling correlation vs {anchor} ({win_label})")
    ax.set_ylabel("Corr")
    ax.axhline(0.0, lw=1, alpha=0.6)
    ax.grid(True, alpha=0.3)
    savefig(f"eu_ff_{freq}_rolling_corr_vs_{anchor}.png")

    # 6) Drawdowns (on growth series)
    nav = growth_of_1(df[factors])
    dds = nav.divide(nav.cummax()) - 1
    ax = dds.plot(figsize=(11,6))
    ax.set_title(f"Drawdowns — based on compounded returns ({freq})")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    savefig(f"eu_ff_{freq}_drawdowns.png")

    # 7) Histograms (distribution) — one figure with multiple overlays is messy;
    # instead plot a grid *but* to respect "one figure per chart", we save one per factor.
    for c in factors:
        plt.figure(figsize=(7,5))
        plt.hist(df[c].dropna().values, bins=60, density=True, alpha=0.85)
        plt.title(f"{c} — distribution ({freq})")
        plt.xlabel("Return")
        plt.ylabel("Density")
        plt.grid(True, alpha=0.3)
        savefig(f"eu_ff_{c.lower()}_{freq}_hist.png")

    print(f"\n✅ Saved figures to: {FIGDIR.resolve()}")
    print("Key files:")
    print(f"- eu_ff_{freq}_returns_timeseries.png")
    print(f"- eu_ff_{freq}_growth.png")
    print(f"- eu_ff_{freq}_rolling_sharpe.png")
    print(f"- eu_ff_{freq}_corr_heatmap.png")
    print(f"- eu_ff_{freq}_rolling_corr_vs_{anchor}.png")
    print(f"- eu_ff_{freq}_drawdowns.png")
    print(f"- eu_ff_*_{freq}_hist.png (one per factor)")

def main():
    parser = argparse.ArgumentParser(description="Visualize Europe Fama–French factors (monthly/daily).")
    parser.add_argument("--freq", choices=["monthly","daily"], default="monthly",
                        help="Which dataset to plot (default: monthly).")
    parser.add_argument("--path", default=None, help="Path to CSV. If omitted, uses default for chosen freq.")
    args = parser.parse_args()

    path = args.path or (DEFAULT_MONTHLY if args.freq=="monthly" else DEFAULT_DAILY)
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}\nUse --path to point to your merged file.")
    visualize(path, freq=args.freq)

if __name__ == "__main__":
    main()

# visuals.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ====== settings ======
FACTOR_FILE = os.path.join("datasets", "ff5_plus_mom_monthly.csv")  # or ff3_plus_mom_monthly.csv
FIG_DIR = "figs"
ROLL_WINDOW = 36  # months

os.makedirs(FIG_DIR, exist_ok=True)

def load_factors(path):
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    # Ensure expected names; earlier scripts already standardized to decimals
    df.columns = [c.upper() for c in df.columns]
    # Make a "MARKET" total return = RF + MKT_RF for comparison
    if "MKT_RF" in df.columns and "RF" in df.columns:
        df["MARKET"] = df["RF"] + df["MKT_RF"]
    return df

def plot_cumulative_growth(fret, title, fname):
    # fret: DataFrame of monthly decimal returns; columns = factors
    growth = (1.0 + fret).fillna(0).cumprod()
    ax = growth.plot(figsize=(10, 6))
    ax.set_title(title)
    ax.set_ylabel("Growth of $1")
    ax.set_xlabel("")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, fname), dpi=150)
    plt.close()

def plot_rolling_sharpe(fret, title, fname, window=36):
    # annualized Sharpe with monthly data
    roll_mean = fret.rolling(window).mean()
    roll_std = fret.rolling(window).std()
    roll_sharpe = np.sqrt(12) * (roll_mean / roll_std)
    ax = roll_sharpe.plot(figsize=(10, 6))
    ax.set_title(title + f" (window={window}m)")
    ax.set_ylabel("Annualized Sharpe")
    ax.set_xlabel("")
    ax.axhline(0.0, lw=1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, fname), dpi=150)
    plt.close()

def plot_rolling_corr_heatmap(fret, title, fname, window=36, step=6):
    """
    Build a time × factor heatmap of rolling correlations to the Market factor (if present),
    otherwise to MKT_RF (excess market). This keeps a single chart per figure.
    """
    # Pick the anchor series
    anchor_col = "MARKET" if "MARKET" in fret.columns else ("MKT_RF" if "MKT_RF" in fret.columns else fret.columns[0])
    others = [c for c in fret.columns if c != anchor_col]
    # Compute rolling correlation with the anchor for each factor
    corrs = {}
    for c in others:
        corrs[c] = fret[c].rolling(window).corr(fret[anchor_col])
    corr_df = pd.DataFrame(corrs).iloc[::step]  # thin the index so labels are readable

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(corr_df.T.values, aspect="auto", origin="lower",
                   extent=[0, corr_df.shape[0], 0, corr_df.shape[1]])

    # Axis ticks
    ax.set_yticks(np.arange(len(corr_df.columns)))
    ax.set_yticklabels(corr_df.columns.tolist())
    ax.set_xticks(np.linspace(0, corr_df.shape[0]-1, 6))
    # map ticks back to dates
    xt_idx = (corr_df.index[np.clip(np.round(np.linspace(0, corr_df.shape[0]-1, 6)).astype(int),0,corr_df.shape[0]-1)]
              .strftime("%Y-%m").tolist())
    ax.set_xticklabels(xt_idx, rotation=0)

    ax.set_title(f"{title}\nRolling corr (window={window}m) vs {anchor_col}")
    fig.colorbar(im, ax=ax, label="Correlation")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, fname), dpi=150)
    plt.close()

def plot_drawdowns(fret, title, fname):
    nav = (1 + fret).cumprod()
    peak = nav.cummax()
    dd = (nav / peak) - 1.0
    ax = dd.plot(figsize=(10,6))
    ax.set_title(title)
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, fname), dpi=150)
    plt.close()

def main():
    df = load_factors(FACTOR_FILE)

    # Choose which series to visualize (exclude RF to avoid noise in growth/DD)
    candidates = [c for c in df.columns if c not in ("RF",)]
    factors = df[candidates].dropna()

    # 1) Cumulative growth
    plot_cumulative_growth(
        factors,
        "Fama–French Factors — Growth of $1 (monthly)",
        "ff_growth.png",
    )

    # 2) Rolling 36m Sharpe
    plot_rolling_sharpe(
        factors,
        "Fama–French Factors — Rolling Sharpe",
        "ff_rolling_sharpe.png",
        window=ROLL_WINDOW,
    )

    # 3) Rolling 36m correlation heatmap vs Market/MKT_RF
    plot_rolling_corr_heatmap(
        factors,
        "Fama–French Factors — Rolling Correlation",
        "ff_rolling_corr.png",
        window=ROLL_WINDOW,
        step=6,
    )

    # 4) Drawdowns
    plot_drawdowns(
        factors,
        "Fama–French Factors — Drawdowns",
        "ff_drawdowns.png",
    )

    print("✅ Figures saved in:", os.path.abspath(FIG_DIR))

if __name__ == "__main__":
    main()

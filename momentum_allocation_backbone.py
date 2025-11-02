
# momentum_allocation_backbone.py
# --------------------------------------------------------------
# Backbone: Momentum-Driven Equity Sleeve + Bond Allocation (60-80 / 40-20)
# --------------------------------------------------------------
# What this script does
# - Builds a cross-sectional momentum long-short equity sleeve on your universe.
# - Mixes it with a bond proxy using either constant or dynamic equity weights (60% to 80%).
# - Reports performance metrics and (optionally) plots results.
# - Designed to work with FREE data you already have (your equity returns CSV + a bond ETF CSV).
#
# Assumptions / Inputs
# - Equity returns CSV is monthly, wide format: index=month-end dates, columns=tickers, values=returns in decimal.
#   If you only have daily returns, first convert them to monthly in your pipeline.
# - Bond returns CSV is monthly, one column named 'BOND' (or 'IEAG.DE' etc.).
# - Dates should align on month-end; if not, the script will inner-join on common months.
#
# How to use
# 1) Set EQUITY_RETURNS_CSV and BOND_RETURNS_CSV to your files.
# 2) Choose a momentum definition (default: 12-1, i.e., last 12 months excluding the most recent month).
# 3) Run and review printed metrics. Plots are optional (set MAKE_PLOTS=True).
# --------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional

# ========================
# ====== CONFIG =========
# ========================

EQUITY_RETURNS_CSV = "Investment_universe/europe_returns_monthly.csv"  # <-- TODO: set your path (monthly, wide, decimal returns)
BOND_RETURNS_CSV   = "Investment_universe/bond_returns_monthly.csv"    # <-- TODO: set your path (monthly, single-column)
RESULTS_PREFIX     = "reports/momentum_allocation"

DECILE = 0.10             # long top 10%, short bottom 10%
TC_BP  = 0.0010           # round-trip transaction cost per notional change (e.g., 10 bps = 0.0010). Applied on turnover.
REBALANCE_FREQ = "M"      # 'M' monthly, 'Q' quarterly (the code assumes monthly momentum by default)
MAKE_PLOTS = False        # set True to save plots under reports/

# Dynamic weighting mode:
# 'sigmoid' -> 0.6..0.8 smooth based on z-scored MOM factor
# 'tiers'   -> discrete 0.60/0.70/0.80 based on MOM sign/thresholds
DYNAMIC_MODE = "sigmoid"  

# ========================
# ===== UTILITIES ========
# ========================

def _ensure_monthly(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        if df.index.freq is None:
            # Keep order; downstream logic uses explicit monthly alignment by merge
            df = df.sort_index()
        return df
    raise ValueError("DataFrame index must be a DatetimeIndex (month-end).")

def compound_monthly_from_daily(daily_wide: pd.DataFrame) -> pd.DataFrame:
    # Helper if you only have daily and want monthly here:
    return (1.0 + daily_wide).resample("M").prod() - 1.0

def zscore(x: pd.Series, min_periods=24) -> pd.Series:
    mu = x.rolling(min_periods, min_periods=min_periods).mean()
    sd = x.rolling(min_periods, min_periods=min_periods).std(ddof=0)
    return (x - mu) / sd

@dataclass
class Perf:
    cagr: float
    vol: float
    sharpe: float
    maxdd: float
    n_months: int
    turnover: float

def perf_metrics(ret_m: pd.Series, rf_annual: float = 0.0, turnover_m: Optional[pd.Series] = None) -> Perf:
    ret_m = ret_m.dropna()
    n = len(ret_m)
    if n == 0:
        return Perf(np.nan, np.nan, np.nan, np.nan, 0, np.nan)
    g = (1.0 + ret_m).prod()
    cagr = g ** (12.0 / n) - 1.0
    vol = ret_m.std(ddof=0) * np.sqrt(12.0)
    sharpe = np.nan
    if vol != 0:
        sharpe = ((ret_m - rf_annual/12.0).mean() * 12.0) / vol
    # Max drawdown on cumulative curve
    eq = (1.0 + ret_m).cumprod()
    peak = eq.cummax()
    dd = (eq / peak - 1.0).min()
    avg_turnover = np.nan
    if turnover_m is not None and len(turnover_m) > 0:
        avg_turnover = turnover_m.mean()
    return Perf(cagr, vol, sharpe, dd, n, avg_turnover)

# ========================
# === MOMENTUM LOGIC =====
# ========================

def mom_12_1(scores_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 12-1 cross-sectional momentum score:
    Product of last 12 months up to t-1, excluding the most recent month.
    Using log-return sum approximation for stability.
    """
    logret = np.log1p(scores_wide)
    # Cum sum of logs over 12 months up to t-1
    cs = logret.rolling(12).sum()  # window t-11..t (inclusive)
    # Exclude the last month: subtract the most recent logret
    cs_ex_last = cs - logret  # t-12..t-1
    return np.expm1(cs_ex_last)

def long_short_equity(returns_m: pd.DataFrame, decile: float = 0.10, tc_bp: float = 0.0010):
    """
    Build monthly long-short equity sleeve based on momentum ranks.
    - returns_m: monthly returns (wide, tickers columns)
    - decile: top/bottom fraction to long/short
    - tc_bp: transaction cost applied to turnover (sum of absolute weight changes)
    Returns:
      ret_ls: pd.Series monthly L-S returns after costs
      w_long, w_short: DataFrames of weights
      turnover: Series of monthly turnover
    """
    ret = returns_m.copy().sort_index()
    mom = mom_12_1(ret)

    w_long = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    w_short = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)

    for dt in ret.index:
        s = mom.loc[dt].dropna()
        if len(s) < 10:
            continue
        q_low  = s.quantile(decile)
        q_high = s.quantile(1.0 - decile)
        longs  = s[s >= q_high].index
        shorts = s[s <= q_low].index
        if len(longs) > 0:
            w_long.loc[dt, longs] =  1.0 / len(longs)
        if len(shorts) > 0:
            w_short.loc[dt, shorts] = -1.0 / len(shorts)

    # Lag weights by 1 month to avoid look-ahead bias
    w_long = w_long.shift(1).fillna(0.0)
    w_short = w_short.shift(1).fillna(0.0)
    w = w_long + w_short  # net exposure around 0 (market-neutral equity sleeve)

    # Compute turnover (sum of absolute weight changes)
    dw = w.diff().abs().sum(axis=1)
    turnover = dw  # already in units of 1.0 = 100% of capital rotated
    # Apply transaction costs on notional change
    tc = turnover * tc_bp

    # L-S returns (weights at start of month times realized returns)
    ret_ls_gross = (w * ret).sum(axis=1)
    ret_ls = ret_ls_gross - tc

    return ret_ls.rename("Equity_LS"), w_long, w_short, turnover

# ========================
# === ALLOCATION LOGIC ===
# ========================

def dynamic_equity_weight(mom_factor: pd.Series, mode: str = "sigmoid") -> pd.Series:
    mom_factor = mom_factor.dropna()
    if mode == "sigmoid":
        z = zscore(mom_factor, min_periods=24)
        # map z to 0.6..0.8 via logistic
        sig = 1.0 / (1.0 + np.exp(-z.clip(-3,3)))
        w_eq = 0.6 + 0.2 * sig  # in [0.6, 0.8]
    elif mode == "tiers":
        # Simple discrete regime: negative -> 0.60, mildly positive -> 0.70, strong -> 0.80
        w_eq = pd.Series(0.7, index=mom_factor.index)
        w_eq[mom_factor <= 0] = 0.60
        w_eq[mom_factor >  mom_factor.rolling(12).std(ddof=0)] = 0.80
    else:
        raise ValueError("Unknown mode. Use 'sigmoid' or 'tiers'.")
    return w_eq.clip(0.6, 0.8).rename("w_equity")

def blend_with_bonds(ret_equity: pd.Series, ret_bond: pd.Series, w_equity: pd.Series) -> pd.Series:
    df = pd.concat([ret_equity, ret_bond, w_equity], axis=1).dropna()
    df.columns = ["eq", "bond", "w"]
    port = df["w"] * df["eq"] + (1.0 - df["w"]) * df["bond"]
    return port.rename("Port")

# ========================
# ======= MAIN ===========
# ========================

def main():
    # ---- Load data
    eq = pd.read_csv(EQUITY_RETURNS_CSV, parse_dates=[0], index_col=0)
    bond = pd.read_csv(BOND_RETURNS_CSV, parse_dates=[0], index_col=0)
    eq = _ensure_monthly(eq)
    bond = _ensure_monthly(bond)

    if bond.shape[1] != 1:
        # assume first column is bond return
        bond = bond.iloc[:, [0]]
    bond_col = bond.columns[0]
    bond = bond.rename(columns={bond_col: "BOND"})

    # Align dates by inner join later
    # ---- Build Equity L-S
    ret_ls, w_long, w_short, turnover = long_short_equity(eq, decile=DECILE, tc_bp=TC_BP)

    # ---- Constant mixes for comparison
    bond_m = bond["BOND"].reindex(ret_ls.index).dropna()
    ret_ls = ret_ls.reindex(bond_m.index)
    const80_20 = 0.80 * ret_ls + 0.20 * bond_m
    const60_40 = 0.60 * ret_ls + 0.40 * bond_m
    const70_30 = 0.70 * ret_ls + 0.30 * bond_m

    # ---- Dynamic equity weight from the *factor* (here we reuse ret_ls as proxy for MOM factor)
    # If you have an external MOM factor series, plug it here instead of ret_ls.
    w_eq = dynamic_equity_weight(ret_ls, mode=DYNAMIC_MODE)
    port_dyn = blend_with_bonds(ret_ls, bond_m, w_eq)

    # ---- Metrics
    p_ls   = perf_metrics(ret_ls)
    p_80   = perf_metrics(const80_20)
    p_70   = perf_metrics(const70_30)
    p_60   = perf_metrics(const60_40)
    p_dyn  = perf_metrics(port_dyn)
    print("=== Equity L-S sleeve (after costs) ===")
    print(p_ls)
    print("\n=== Constant mixes ===")
    print("80/20:", p_80)
    print("70/30:", p_70)
    print("60/40:", p_60)
    print("\n=== Dynamic mix ===")
    print(DYNAMIC_MODE, ":", p_dyn)

    # ---- Save basic outputs
    out = pd.concat([ret_ls, bond_m, const80_20.rename("Const80_20"),
                     const70_30.rename("Const70_30"), const60_40.rename("Const60_40"),
                     w_eq, port_dyn], axis=1)
    out.to_csv(f"{RESULTS_PREFIX}_timeseries.csv", float_format="%.8f")

    # Save weights for auditability
    w_long.to_csv(f"{RESULTS_PREFIX}_weights_long.csv", float_format="%.6f")
    w_short.to_csv(f"{RESULTS_PREFIX}_weights_short.csv", float_format="%.6f")

    # ---- Optional plots
    if MAKE_PLOTS:
        # Cumulative performance
        plt.figure()
        (1.0 + out[["Const80_20","Const70_30","Const60_40","Port"]].dropna()).cumprod().plot()
        plt.title("Cumulative Returns: Constant vs Dynamic Mix")
        plt.xlabel("Date"); plt.ylabel("Growth of 1")
        plt.tight_layout()
        plt.savefig(f"{RESULTS_PREFIX}_cum.png", dpi=140)

        # Equity weight over time
        plt.figure()
        out["w_equity"].dropna().plot()
        plt.title("Dynamic Equity Weight")
        plt.xlabel("Date"); plt.ylabel("Weight")
        plt.tight_layout()
        plt.savefig(f"{RESULTS_PREFIX}_weights.png", dpi=140)

        # Drawdown of dynamic portfolio
        plt.figure()
        port = out["Port"].dropna()
        eq_curve = (1.0 + port).cumprod()
        peak = eq_curve.cummax()
        dd = eq_curve/peak - 1.0
        dd.plot()
        plt.title("Drawdown: Dynamic Portfolio")
        plt.xlabel("Date"); plt.ylabel("Drawdown")
        plt.tight_layout()
        plt.savefig(f"{RESULTS_PREFIX}_dd.png", dpi=140)

if __name__ == "__main__":
    main()

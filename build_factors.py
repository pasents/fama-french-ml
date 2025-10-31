"""
build_factors.py
Generates monthly Fama–French-style factors for your European dataset.
Outputs:
  - Investment_universe/factors_europe.csv
  - reports/factors_correlation_heatmap.png
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# ----------------------------
# CONFIG
# ----------------------------
DATA_DIR = Path("Investment_universe")
REPORTS_DIR = Path("reports")
OUT_CSV = DATA_DIR / "factors_europe.csv"
HEATMAP_PNG = REPORTS_DIR / "factors_correlation_heatmap.png"

# ----------------------------
# Utilities
# ----------------------------
def to_month_end(df, date_col="Date"):
    """Ensure Date index is monthly end."""
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.sort_index()
    df.index = df.index.to_period("M").to_timestamp("M")
    return df

def winsorize(s, p=0.01):
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lo, hi)

def _read_first_existing(candidates):
    for name in candidates:
        path = DATA_DIR / name
        if path.exists():
            print(f"[INFO] Found {path.name}")
            try:
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                df = df.rename_axis("Date").reset_index()
                return df
            except Exception:
                return pd.read_csv(path)
    raise FileNotFoundError(
        f"None of these files were found in {DATA_DIR}:\n  - " + "\n  - ".join(candidates)
    )

# ----------------------------
# Option A: Load official FF Europe factors if available
# ----------------------------
def try_load_ff_europe(path=DATA_DIR / "ff_europe_5f_mom.csv"):
    if path.exists():
        ff = pd.read_csv(path)
        date_col = "date" if "date" in ff.columns else "Date"
        ff[date_col] = pd.to_datetime(ff[date_col])
        ff = ff.set_index(date_col).sort_index()
        ff.index = ff.index.to_period("M").to_timestamp("M")
        cols = ["MKT", "SMB", "HML", "RMW", "CMA", "MOM"]
        outcols = [c for c in cols if c in ff.columns]
        rf = ff["RF"] if "RF" in ff.columns else pd.Series(0.0, index=ff.index, name="RF")
        ff = ff[outcols].copy()
        ff["RF"] = rf
        return ff
    return None

# ----------------------------
# Option B: Build in-house factors
# ----------------------------
def load_core_prices_returns():
    price_candidates = [
        "europe_prices_cleaned.csv",
        "modeling_prices.csv",
        "europe_prices.csv",
        "prices.csv",
    ]
    return_candidates = [
        "europe_returns_cleaned.csv",
        "modeling_returns.csv",
        "europe_returns.csv",
        "returns.csv",
    ]
    prices = _read_first_existing(price_candidates)
    rets = _read_first_existing(return_candidates)
    prices = to_month_end(prices, date_col="Date")
    rets = to_month_end(rets, date_col="Date")
    common = prices.columns.intersection(rets.columns)
    prices = prices[common]
    rets = rets[common]
    return prices, rets

def load_fundamentals():
    fpath = DATA_DIR / "fundamentals.csv"
    if fpath.exists():
        f = pd.read_csv(fpath)
        f["date"] = pd.to_datetime(f["date"]).dt.to_period("M").dt.to_timestamp("M")
        return f
    return None

def monthly_equity_market_return(rets, mktcap=None):
    if mktcap is None:
        mkt = rets.mean(axis=1)
    else:
        w = mktcap.div(mktcap.sum(axis=1), axis=0).fillna(0.0)
        mkt = (w * rets).sum(axis=1)
    mkt.name = "MKT"
    return mkt

def cross_sectional_factor_long_short(rets_m, signal_m, top=0.2, bottom=0.2):
    fac = []
    for dt in rets_m.index:
        if dt not in signal_m.index:
            fac.append(np.nan)
            continue
        r = rets_m.loc[dt].dropna()
        s = signal_m.loc[dt].reindex(r.index).dropna()
        r = r.reindex(s.index)
        if len(s) < 10:
            fac.append(np.nan)
            continue
        q_hi, q_lo = s.quantile(1 - top), s.quantile(bottom)
        long_names = s.index[s >= q_hi]
        short_names = s.index[s <= q_lo]
        if len(long_names) == 0 or len(short_names) == 0:
            fac.append(np.nan)
            continue
        fac_val = r[long_names].mean() - r[short_names].mean()
        fac.append(fac_val)
    return pd.Series(fac, index=rets_m.index, name="LS")

def build_factors_inhouse():
    prices, rets = load_core_prices_returns()

    # Convert to monthly if daily
    if rets.index.to_period("M").nunique() < len(rets.index):
        rets = (1.0 + rets).resample("M").prod() - 1.0
        prices = prices.resample("M").last()

    fundamentals = load_fundamentals()
    mktcap = None
    if fundamentals is not None and "mktcap" in fundamentals.columns:
        mktcap = fundamentals.pivot_table(index="date", columns="ticker", values="mktcap")
        mktcap = mktcap.reindex(rets.index).reindex(columns=rets.columns)

    MKT = monthly_equity_market_return(rets, mktcap=mktcap)

    # Momentum (12-1)
    mom_signal = (prices.shift(1) / prices.shift(13)) - 1.0
    MOM = cross_sectional_factor_long_short(rets, mom_signal)
    MOM.name = "MOM"

    # Size (SMB)
    if mktcap is not None:
        size_signal = -np.log(mktcap.replace(0, np.nan))
    else:
        size_signal = -np.log(prices.replace(0, np.nan))
    SMB = cross_sectional_factor_long_short(rets, size_signal, top=0.5, bottom=0.5)
    SMB.name = "SMB"

    # Value (HML)
    if fundamentals is not None and "bm" in fundamentals.columns:
        bm = fundamentals.pivot_table(index="date", columns="ticker", values="bm")
        bm = bm.reindex(rets.index).reindex(columns=rets.columns)
        HML = cross_sectional_factor_long_short(rets, bm, top=0.3, bottom=0.3)
    else:
        HML = pd.Series(np.nan, index=rets.index, name="HML")

    # Profitability (RMW)
    if fundamentals is not None and "profitability" in fundamentals.columns:
        prof = fundamentals.pivot_table(index="date", columns="ticker", values="profitability")
        prof = prof.reindex(rets.index).reindex(columns=rets.columns)
        RMW = cross_sectional_factor_long_short(rets, prof, top=0.3, bottom=0.3)
    else:
        RMW = pd.Series(np.nan, index=rets.index, name="RMW")

    # Investment (CMA)
    if fundamentals is not None and "asset_growth" in fundamentals.columns:
        inv = -fundamentals.pivot_table(index="date", columns="ticker", values="asset_growth")
        inv = inv.reindex(rets.index).reindex(columns=rets.columns)
        CMA = cross_sectional_factor_long_short(rets, inv, top=0.3, bottom=0.3)
    else:
        CMA = pd.Series(np.nan, index=rets.index, name="CMA")

    RF = pd.Series(0.0, index=rets.index, name="RF")

    factors = pd.concat([MKT, SMB, HML, RMW, CMA, MOM, RF], axis=1)
    for c in ["MKT", "SMB", "HML", "RMW", "CMA", "MOM"]:
        factors[c] = winsorize(factors[c].dropna()).reindex(factors.index)
    factors = factors.dropna(how="all")
    return factors

# ----------------------------
# Main
# ----------------------------
def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ff = try_load_ff_europe()
    if ff is not None:
        factors = ff
        source = "Loaded official FF Europe file"
    else:
        factors = build_factors_inhouse()
        source = "Built from modeling_prices.csv and modeling_returns.csv"
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    factors.to_csv(OUT_CSV, float_format="%.8f")

    # Correlation heatmap
    corr = factors.drop(columns=[c for c in ["RF"] if c in factors.columns]).corr(method="spearman")
    plt.figure(figsize=(7, 5))
    plt.imshow(corr, interpolation="nearest")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title("Factor Spearman Correlation (monthly)")
    plt.tight_layout()
    plt.savefig(HEATMAP_PNG, dpi=160)
    plt.close()

    print(f"[OK] {source}")
    print(f"[OK] Saved factors to: {OUT_CSV}")
    print(f"[OK] Saved correlation heatmap to: {HEATMAP_PNG}")
    print(factors.tail())

if __name__ == "__main__":
    main()

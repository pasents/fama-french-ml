import os
import pandas as pd

# Output location for your backbone
OUT = "Investment_universe/bond_returns_monthly.csv"

# --- Configuration ---
TICKER = "XGLE.DE"  # iShares Core Euro Government Bond UCITS ETF
START_DATE = "2013-01-01"
print(f"[INFO] Fetching {TICKER} from {START_DATE} onward...")

try:
    import yfinance as yf
except ImportError:
    raise SystemExit("Please install yfinance first:  pip install yfinance")

# === Download daily adjusted prices ===
df = yf.download(TICKER, start=START_DATE, progress=False, auto_adjust=True)

if df.empty:
    raise RuntimeError(f"Failed to fetch data for {TICKER}. "
                       "Try IEGA.AS or EUNA.AS instead.")

# Normalize columns and index
if "Close" in df.columns:
    px = df[["Close"]].rename(columns={"Close": "PX"})
elif "Adj Close" in df.columns:
    px = df[["Adj Close"]].rename(columns={"Adj Close": "PX"})
else:
    raise RuntimeError(f"No Close/Adj Close column found for {TICKER}.")

if getattr(px.index, "tz", None) is not None:
    px.index = px.index.tz_localize(None)
px = px.sort_index()

# === Convert to monthly returns ===
monthly_px = px.resample("ME").last()
bond_ret = monthly_px.pct_change().dropna()
bond_ret.columns = ["BOND"]

# === Save results ===
os.makedirs("Investment_universe", exist_ok=True)
bond_ret.to_csv(OUT, float_format="%.8f")
print(f"[OK] Saved {OUT} from {TICKER}  shape={bond_ret.shape}")
print(bond_ret.head())
print(bond_ret.tail())

# Quick sanity check: range
print(f"\nRange: {bond_ret.index.min().date()} to {bond_ret.index.max().date()}")

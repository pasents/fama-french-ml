# make_benchmark_iwda.py
# -------------------------------------------------------------
# Downloads IWDA.AS (MSCI World ETF in EUR) from Yahoo Finance
# Creates monthly total returns CSV for use as a benchmark.
# -------------------------------------------------------------

import os
import pandas as pd

OUT = "Investment_universe/MSCI_BENCHMARK.csv"
TICKER = "IWDA.AS"

try:
    import yfinance as yf
except ImportError:
    raise SystemExit("Please install yfinance first:  pip install yfinance")

print(f"[INFO] Fetching {TICKER} from Yahoo Finance (max history)...")

# Fetch daily adjusted prices (auto_adjust=True = total-return adjusted)
df = yf.download(TICKER, period="max", interval="1d", auto_adjust=True, progress=False)

if df is None or df.empty:
    raise RuntimeError(f"[ERROR] No data returned for {TICKER}. Check ticker or network connection.")

# Prefer Close; fallback to Adj Close
if "Close" in df.columns:
    px = df[["Close"]].rename(columns={"Close": "PX"})
elif "Adj Close" in df.columns:
    px = df[["Adj Close"]].rename(columns={"Adj Close": "PX"})
else:
    raise RuntimeError("No Close or Adj Close column found in Yahoo data.")

# Clean index
if getattr(px.index, "tz", None) is not None:
    px.index = px.index.tz_localize(None)
px = px.sort_index()

# Compute monthly total returns
monthly_px = px.resample("ME").last()
monthly_ret = monthly_px.pct_change().dropna()
monthly_ret.columns = ["Benchmark"]

# Save to CSV
os.makedirs("Investment_universe", exist_ok=True)
monthly_ret.to_csv(OUT, float_format="%.8f")

print(f"[OK] Saved monthly benchmark returns -> {OUT}")
print(f"Range: {monthly_ret.index.min().date()} → {monthly_ret.index.max().date()}")
print(f"Rows: {len(monthly_ret)}")

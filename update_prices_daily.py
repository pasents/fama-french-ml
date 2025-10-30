import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pandas as pd

# --- Config ---
BASE = Path("Investment_universe")
UNIVERSE_CSV = BASE / "modeling_universe.csv"
PRICES_CSV   = BASE / "modeling_prices.csv"
RETURNS_CSV  = BASE / "modeling_returns.csv"
LOG_PATH     = Path("reports") / "update_log.txt"

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        LOG_PATH.parent.mkdir(exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

def ensure_yfinance():
    try:
        import yfinance as yf  # noqa: F401
    except Exception:
        log("yfinance not available. Install with: pip install yfinance")
        raise

def read_universe() -> pd.Series:
    if not UNIVERSE_CSV.exists():
        raise FileNotFoundError(f"Universe list not found: {UNIVERSE_CSV.resolve()}")
    df = pd.read_csv(UNIVERSE_CSV)
    # Normalize column name
    if "Ticker" not in df.columns:
        if len(df.columns) == 1:
            df.columns = ["Ticker"]
        else:
            for c in df.columns:
                if str(c).strip().lower() in {"ticker", "tickers", "symbol", "name"}:
                    df = df.rename(columns={c: "Ticker"})
                    break
            else:
                raise ValueError(f"Could not find ticker column in {UNIVERSE_CSV}")
    return df["Ticker"].dropna().astype(str).drop_duplicates()

def read_prices() -> pd.DataFrame:
    if not PRICES_CSV.exists():
        raise FileNotFoundError(f"Prices file not found: {PRICES_CSV.resolve()}")
    df = pd.read_csv(PRICES_CSV, index_col=0, parse_dates=True).sort_index()
    return df

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    rets = prices.pct_change()
    rets = rets.dropna(how="all")
    return rets

def next_start_date(prices: pd.DataFrame) -> datetime:
    if prices.empty:
        return datetime(2010, 1, 1, tzinfo=timezone.utc)
    last_dt = pd.to_datetime(prices.index[-1]).to_pydatetime()
    return (last_dt + timedelta(days=1)).replace(tzinfo=timezone.utc)

def fetch_new_prices_yf(tickers, start_dt_utc: datetime) -> pd.DataFrame:
    import yfinance as yf
    start_str = start_dt_utc.strftime("%Y-%m-%d")
    end_str = datetime.utcnow().strftime("%Y-%m-%d")
    if start_str >= end_str:
        return pd.DataFrame()  # nothing new

    log(f"Fetching from yfinance: {len(tickers)} tickers, {start_str} -> {end_str}")
    data = yf.download(
        tickers=list(tickers),
        start=start_str,
        end=end_str,
        auto_adjust=True,          # prices are adjusted already
        group_by="ticker",         # -> MultiIndex: (ticker, field)
        progress=False,
        threads=True,
        interval="1d",
    )

    if data is None or data.empty:
        return pd.DataFrame()

    # If single ticker, yfinance returns a single-index DF; normalize to (ticker,) columns
    if not isinstance(data.columns, pd.MultiIndex):
        # Prefer Close, else Adj Close
        if "Close" in data.columns:
            out = data[["Close"]].copy()
        elif "Adj Close" in data.columns:
            out = data[["Adj Close"]].copy()
        else:
            return pd.DataFrame()
        # Rename the single column to the ticker symbol
        tkr = list(tickers)[0]
        out.columns = [tkr]
        out.index.name = "Date"
        return out.sort_index()

    # MultiIndex case: (ticker, field)
    # Prefer 'Close' (already adjusted). If missing, fall back to 'Adj Close'.
    fields_lvl = data.columns.get_level_values(1)
    if "Close" in set(fields_lvl):
        out = data.xs("Close", axis=1, level=1, drop_level=True).copy()
    elif "Adj Close" in set(fields_lvl):
        out = data.xs("Adj Close", axis=1, level=1, drop_level=True).copy()
    else:
        # Some edge builds: reverse order (field, ticker). Handle gracefully.
        lvl0 = set(map(str, data.columns.get_level_values(0)))
        if "Close" in lvl0:
            tmp = data.xs("Close", axis=1, level=0, drop_level=True).copy()
            # columns are tickers already
            out = tmp
        elif "Adj Close" in lvl0:
            tmp = data.xs("Adj Close", axis=1, level=0, drop_level=True).copy()
            out = tmp
        else:
            return pd.DataFrame()

    out.index.name = "Date"
    out = out.sort_index()
    # Drop columns (tickers) with no new data
    out = out.dropna(axis=1, how="all")
    return out


def main():
    ensure_yfinance()
    tickers = read_universe()
    prices = read_prices()

    # Skip weekends to avoid pointless runs
    now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
    if now_utc.weekday() >= 5:
        log("Weekend detected. No update performed.")
        return

    start_dt = next_start_date(prices)
    new_px = fetch_new_prices_yf(tickers, start_dt)

    if new_px is None or new_px.empty:
        log("No new rows available. Dataset already up to date.")
        return

    # Ensure all universe columns exist (preserve order)
    for col in tickers:
        if col not in new_px.columns:
            new_px[col] = pd.NA
    new_px = new_px[list(tickers)]  # consistent column order

    # Append non-duplicate dates, keep latest for duplicates
    combined = pd.concat([prices, new_px], axis=0)
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()

    # Basic integrity checks
    if combined.index.duplicated().any():
        raise ValueError("Duplicate dates after merge; aborting.")
    if combined.shape[1] == 0:
        raise ValueError("No columns in combined prices after merge; aborting.")

    returns = compute_returns(combined)

    BASE.mkdir(parents=True, exist_ok=True)
    combined.to_csv(PRICES_CSV)
    returns.to_csv(RETURNS_CSV)

    added_rows = len(combined) - len(prices)
    log(f"Appended {added_rows} row(s). New shapes: prices={combined.shape}, returns={returns.shape}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"ERROR: {e}")
        raise

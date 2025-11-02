# make_inputs_for_momentum.py
# Builds inputs for momentum_allocation_backbone.py
# - europe_returns_monthly.csv  (from daily modeling_returns.csv)
# - bond_returns_monthly.csv    (from bond ETF or RF cash proxy)
# - rf_estr_monthly.csv         (from factors_europe.csv)

import os, sys
import pandas as pd

UNIV_DIR = "Investment_universe"
DAILY_RET = os.path.join(UNIV_DIR, "modeling_returns.csv")        # daily, wide
MONTHLY_RET = os.path.join(UNIV_DIR, "europe_returns_monthly.csv")

# Bond sources (checked in order)
LOCAL_BOND_RET = os.path.join(UNIV_DIR, "bond_returns_monthly.csv")  # if exists, we keep it
LOCAL_BOND_PRICE = os.path.join(UNIV_DIR, "BOND_LOCAL.csv")          # optional: your own price CSV (Date, Adj Close)

# Common, free Yahoo bond ETFs (any one is fine)
#  - EUNA.DE   iShares Core Euro Government Bond UCITS ETF (EUR, Xetra)
#  - IEAC.AS   iShares Core Euro Corporate Bond UCITS ETF (EUR, Amsterdam)
#  - AGGH.L    iShares Core Global Aggregate Bond UCITS ETF USD Hedged (London; hedged, but fine as proxy)
#  - BNDX      Vanguard Total Intl Bond ETF (USD hedged; last-resort proxy)
BOND_TICKERS = ["EUNA.DE", "IEAC.AS", "AGGH.L", "BNDX"]

BOND_RET_OUT = os.path.join(UNIV_DIR, "bond_returns_monthly.csv")
FACTORS_EU = os.path.join(UNIV_DIR, "factors_europe.csv")
RF_OUT = os.path.join(UNIV_DIR, "rf_estr_monthly.csv")

def to_monthly_from_daily_returns(daily_wide: pd.DataFrame) -> pd.DataFrame:
    # pandas 3: 'ME' = month-end
    return (1.0 + daily_wide).resample("ME").prod() - 1.0

def make_equity_monthly():
    if not os.path.exists(DAILY_RET):
        print(f"[ERROR] Not found: {DAILY_RET}")
        sys.exit(1)
    daily = pd.read_csv(DAILY_RET, parse_dates=[0], index_col=0).sort_index()
    if not isinstance(daily.index, pd.DatetimeIndex):
        raise ValueError("modeling_returns.csv must have a date index column named 'Date'.")
    monthly = to_monthly_from_daily_returns(daily)
    monthly.to_csv(MONTHLY_RET, float_format="%.8f")
    print(f"[OK] Saved monthly equity returns -> {MONTHLY_RET}  shape={monthly.shape}")

def try_bond_from_local_price():
    if not os.path.exists(LOCAL_BOND_PRICE):
        return False
    df = pd.read_csv(LOCAL_BOND_PRICE)
    cols = {c.lower().replace(" ", ""): c for c in df.columns}
    date_col = next((cols[k] for k in cols if k in ("date",)), None)
    adj_col  = next((cols[k] for k in cols if k in ("adjclose","adjustedclose","adjclose*","adjclose(€)","adjclose($)")), None)
    if date_col is None or adj_col is None:
        print("[WARN] BOND_LOCAL.csv does not have Date/Adj Close columns (Yahoo format).")
        return False
    px = df[[date_col, adj_col]].dropna()
    px[date_col] = pd.to_datetime(px[date_col])
    px = px.set_index(date_col).sort_index()
    monthly_px = px.resample("ME").last()
    bond_ret = monthly_px.pct_change().dropna()
    bond_ret.columns = ["BOND"]
    bond_ret.to_csv(BOND_RET_OUT, float_format="%.8f")
    print(f"[OK] Saved bond returns from local price -> {BOND_RET_OUT}  shape={bond_ret.shape}")
    return True

def try_bond_with_yfinance():
    try:
        import yfinance as yf
    except ImportError:
        print("[INFO] yfinance not installed; skip online fetch. `pip install yfinance` if you want it.")
        return False

    for tkr in BOND_TICKERS:
        try:
            s = yf.download(tkr, start="2013-01-01", progress=False, auto_adjust=True)
            if s is None or len(s) == 0:
                print(f"[WARN] yfinance returned empty data for {tkr}. Trying next.")
                continue

            # Prefer Close if auto_adjust=True; otherwise fall back to Adj Close
            if isinstance(s, pd.Series):
                px = s.to_frame("Close")
            else:
                col = "Close" if "Close" in s.columns else ("Adj Close" if "Adj Close" in s.columns else None)
                if col is None:
                    print(f"[WARN] {tkr}: no Close/Adj Close column. Trying next.")
                    continue
                px = s[[col]].rename(columns={col: "Close"})

            monthly_px = px.resample("ME").last()
            bond_ret = monthly_px.pct_change().dropna()
            if bond_ret.empty:
                print(f"[WARN] Could not compute monthly returns for {tkr}.")
                continue

            bond_ret.columns = ["BOND"]
            bond_ret.to_csv(BOND_RET_OUT, float_format="%.8f")
            print(f"[OK] Saved bond returns from yfinance ({tkr}) -> {BOND_RET_OUT}  shape={bond_ret.shape}")
            return True
        except Exception as e:
            print(f"[WARN] {tkr} fetch failed: {e}")
            continue
    return False


def make_rf_from_french():
    if not os.path.exists(FACTORS_EU):
        print("[INFO] factors_europe.csv not found; skipping RF creation.")
        return None
    try:
        f = pd.read_csv(FACTORS_EU, parse_dates=[0])
        # Normalize headers
        f.columns = [c.strip().upper().replace(" ", "_") for c in f.columns]
        rf_col = next((c for c in f.columns if c in ("RF","R_F","RISK_FREE","RISKFREE")), None)
        date_col = next((c for c in f.columns if c in ("DATE","TIME_PERIOD","YEAR","MONTH")), None)
        if rf_col is None or date_col is None:
            print("[INFO] Could not identify RF/Date in factors_europe.csv; skipping RF creation.")
            return None
        rf = f[[date_col, rf_col]].copy()
        # If numeric YYYYMM, convert; else use parsed Date and normalize to month-end
        if pd.api.types.is_integer_dtype(rf[date_col]):
            rf["Date"] = pd.to_datetime(rf[date_col].astype(str) + "01") + pd.offsets.MonthEnd(0)
        else:
            rf["Date"] = pd.to_datetime(rf[date_col]) + pd.offsets.MonthEnd(0)
        rf["ESTR"] = pd.to_numeric(rf[rf_col], errors="coerce") / 100.0  # percent -> decimal
        rf = rf[["Date","ESTR"]].dropna().set_index("Date").sort_index()
        rf.to_csv(RF_OUT, float_format="%.6f")
        print(f"[OK] Saved monthly RF (proxy) from French factors -> {RF_OUT}  shape={rf.shape}")
        return rf
    except Exception as e:
        print(f"[INFO] RF creation from factors_europe.csv skipped: {e}")
        return None

def fallback_bond_from_rf(rf: pd.DataFrame | None):
    """Create a 'cash' proxy bond series BOND = RF/12 when no ETF is available."""
    if rf is None or rf.empty:
        print("[ERROR] No RF available to create cash proxy. Supply BOND_LOCAL.csv or install yfinance.")
        sys.exit(1)
    bond = (rf.copy())
    bond.columns = ["BOND_RATE"]
    # Convert annualized RF to monthly simple return
    bond["BOND"] = bond["BOND_RATE"] / 12.0
    bond = bond[["BOND"]]
    bond.to_csv(BOND_RET_OUT, float_format="%.8f")
    print(f"[OK] Created bond CASH proxy from RF -> {BOND_RET_OUT}  shape={bond.shape}")

if __name__ == "__main__":
    os.makedirs(UNIV_DIR, exist_ok=True)

    # 1) Equity monthly from daily
    make_equity_monthly()

    # 2) Bond returns: keep if already exists; else try local CSV; else yfinance; else RF cash proxy
    if os.path.exists(LOCAL_BOND_RET):
        print(f"[SKIP] Bond returns already exist -> {LOCAL_BOND_RET}")
    else:
        got_bond = try_bond_from_local_price()
        if not got_bond:
            got_bond = try_bond_with_yfinance()
        if not got_bond:
            print("[INFO] Could not fetch any bond ETF. Falling back to RF cash proxy.")
            rf = make_rf_from_french()
            fallback_bond_from_rf(rf)

    # 3) Ensure RF CSV exists (nice to have for Sharpe/excess returns)
    if not os.path.exists(RF_OUT):
        make_rf_from_french()

    print("\nAll set. Now run:  python momentum_allocation_backbone.py")

# dataeu.py
import re
import pandas as pd
from pathlib import Path
from pandas_datareader import data as web
from pandas_datareader._utils import RemoteDataError

OUTDIR = Path("datasets")
OUTDIR.mkdir(exist_ok=True)

# -------- column normalization helpers --------
def _canonize(name: str) -> str:
    """Lowercase, remove non-letters, then map to canonical FF names."""
    raw = re.sub(r"[^A-Za-z]+", "", name or "").lower()  # e.g., "Mkt-RF " -> "mktrf"
    mapping = {
        "mktrf": "MKT_RF",
        "smb": "SMB",
        "hml": "HML",
        "rmw": "RMW",
        "cma": "CMA",
        "rf": "RF",
        "mom": "MOM",   # common
        "umd": "MOM",   # alt label used in some packs
        "wml": "MOM",   # "winners minus losers" alias
    }
    return mapping.get(raw, name.strip())

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    newcols = []
    for c in df.columns:
        canon = _canonize(str(c))
        # final clean if we didn't map
        canon = canon.strip()
        newcols.append(canon)
    df.columns = newcols

    # final pass: if still no MOM but a 'Mom' exists, map it
    if "MOM" not in df.columns:
        if "Mom" in df.columns:
            df = df.rename(columns={"Mom": "MOM"})
    return df

# -------- index normalization helpers --------
def _to_month_end_index(idx):
    if isinstance(idx, pd.PeriodIndex):
        return idx.to_timestamp(how="end")
    if isinstance(idx, pd.DatetimeIndex):
        return idx.to_period("M").to_timestamp("M")
    return pd.to_datetime(idx).to_period("M").to_timestamp("M")

def _to_day_index(idx):
    if isinstance(idx, pd.PeriodIndex):
        return idx.to_timestamp()
    if isinstance(idx, pd.DatetimeIndex):
        return idx.tz_localize(None).normalize()
    return pd.to_datetime(idx).tz_localize(None).normalize()

def _clean(df: pd.DataFrame, is_daily: bool = False) -> pd.DataFrame:
    df = df.copy()
    df = _normalize_columns(df)
    # percent -> decimal
    df = df / 100.0
    idx = _to_day_index(df.index) if is_daily else _to_month_end_index(df.index)
    df.index = idx
    return df

def _fetch_one(key: str, is_daily: bool) -> pd.DataFrame:
    try:
        tbl = web.DataReader(key, "famafrench", start=1900)[0]
        df = _clean(tbl, is_daily=is_daily)
        return df
    except RemoteDataError as e:
        raise RuntimeError(f"Could not fetch '{key}' from Ken French: {e}") from e

# -------- fetchers --------
def fetch_europe_ff5():
    m = _fetch_one("Europe_5_Factors",        is_daily=False)
    d = _fetch_one("Europe_5_Factors_Daily",  is_daily=True)
    return m, d

def fetch_europe_mom():
    m = _fetch_one("Europe_Mom_Factor",        is_daily=False)
    d = _fetch_one("Europe_Mom_Factor_Daily",  is_daily=True)
    return m, d

# -------- io/report --------
def save_and_report(df: pd.DataFrame, path: Path, label: str):
    df.to_csv(path, index_label="Date")
    print(f" - {path.name:<34} {df.index.min().date()} → {df.index.max().date()} | rows={len(df)} | cols={list(df.columns)} [{label}]")

def main():
    ff5_m, ff5_d = fetch_europe_ff5()
    mom_m, mom_d = fetch_europe_mom()

    print(f"✅ Saving CSVs to: {OUTDIR.resolve()}")
    save_and_report(ff5_m, OUTDIR / "europe_ff5_monthly.csv", "FF5 monthly")
    save_and_report(ff5_d, OUTDIR / "europe_ff5_daily.csv",   "FF5 daily")
    save_and_report(mom_m, OUTDIR / "europe_mom_monthly.csv", "MOM monthly")
    save_and_report(mom_d, OUTDIR / "europe_mom_daily.csv",   "MOM daily")

    # Ensure we actually have a MOM column before joining
    def ensure_mom(df: pd.DataFrame, label: str):
        if "MOM" not in df.columns:
            # try to find any column containing 'mom' ignoring case
            candidates = [c for c in df.columns if "mom" in c.lower() or c.upper() in ("UMD", "WML")]
            if candidates:
                df = df.rename(columns={candidates[0]: "MOM"})
        if "MOM" not in df.columns:
            raise KeyError(f"{label}: Momentum column missing after normalization. Columns: {list(df.columns)}")
        return df

    mom_m = ensure_mom(mom_m, "Monthly momentum")
    mom_d = ensure_mom(mom_d, "Daily momentum")

    # merged FF5 + MOM
    ff5p_m = ff5_m.join(mom_m[["MOM"]], how="left")
    ff5p_d = ff5_d.join(mom_d[["MOM"]], how="left")

    save_and_report(ff5p_m, OUTDIR / "europe_ff5_plus_mom_monthly.csv", "FF5+MOM monthly")
    save_and_report(ff5p_d, OUTDIR / "europe_ff5_plus_mom_daily.csv",   "FF5+MOM daily")

if __name__ == "__main__":
    main()

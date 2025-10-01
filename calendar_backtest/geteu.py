# iwda_dayofweek_backtest.py
# Backtests "Buy DOW, Sell Next Trading Day" strategies on IWDA.AS (iShares MSCI World UCITS ETF)
# PLUS: Buy & Hold benchmark and a richer metrics comparison.
#
# Strategies:
#   - Buy Mon, Sell Tue
#   - Buy Tue, Sell Wed
#   - Buy Wed, Sell Thu
#   - Buy Thu, Sell Fri
#   - Buy Fri, Sell Mon (next trading day after Friday)
#   - Buy & Hold (Adj Close, close-to-close compounding)
#
# Notes:
# - Uses close-to-next-close returns for DOW legs (no intraday slippage).
# - Optional transaction costs in basis points (per side) are supported for DOW legs
#   (charged once per round trip). Buy & Hold does not include fees by default.
# - Handles holidays/weekends automatically via next available trading day.
#
# Usage:
#   python iwda_dayofweek_backtest.py
#
# Dependencies:
#   pip install yfinance pandas numpy matplotlib
# iwda_dayofweek_backtest.py
# (same header as before)

import sys
import math
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from dataclasses import dataclass

# -------------------- Config --------------------
TICKER = "IWDA.AS"
START = "2010-01-01"
END   = None              # e.g., "2025-01-01" or None for latest
FEE_BPS_PER_SIDE = 0.0
PLOT_TITLE = f"Day-of-Week One-Day Strategies vs Buy & Hold on {TICKER}"
# ------------------------------------------------

@dataclass
class StratResult:
    name: str
    n_trades: int
    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    ann_vol: float
    win_rate: float
    max_drawdown: float
    calmar: float
    exposure: float
    equity: pd.Series
    rets: pd.Series

def _tz_naive(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    if getattr(idx, "tz", None) is not None:
        df = df.tz_convert(None)
    return df

def _download_prices(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    data = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if data.empty:
        raise RuntimeError(f"No data returned for {ticker}. Check the symbol or date range.")
    data = _tz_naive(data).sort_index()
    data["PX"] = data.get("Adj Close", data["Close"]).copy()
    data["weekday"] = data.index.weekday
    data["PX_next"] = data["PX"].shift(-1)
    data["date_next"] = data.index.to_series().shift(-1).values
    data["r_cc"] = data["PX_next"] / data["PX"] - 1.0
    return data

def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())

def _annualize(daily_rets: pd.Series, periods_per_year: int = 252) -> tuple[float, float, float]:
    r = daily_rets.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mean = float(r.mean())
    vol = float(r.std(ddof=0))
    down = r[r < 0.0]
    down_vol = float(down.std(ddof=0)) if len(down) else 0.0
    ann_ret = (1.0 + mean) ** periods_per_year - 1.0
    ann_vol = vol * math.sqrt(periods_per_year)
    ann_down_vol = down_vol * math.sqrt(periods_per_year)
    return ann_ret, ann_vol, ann_down_vol

def _cagr_from_equity(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    t_years = (equity.index[-1] - equity.index[0]).days / 365.25
    if t_years <= 0:
        return 0.0
    return float(equity.iloc[-1]) ** (1.0 / t_years) - 1.0

def _build_dow_strategy(data: pd.DataFrame, buy_weekday: int, label: str, fee_bps_per_side: float) -> StratResult:
    mask = data["weekday"] == buy_weekday
    r = data.loc[mask, "r_cc"].dropna().copy()
    tc = (fee_bps_per_side / 10_000.0) * 2.0
    r_net = r - tc

    equity_trades = (1.0 + r_net).cumprod()
    exit_dates = data.loc[mask & data["PX_next"].notna(), "date_next"].astype("datetime64[ns]")
    eq_tmp = pd.Series(equity_trades.values, index=exit_dates.values)

    full_equity = eq_tmp.reindex(data.index).ffill()
    full_equity.iloc[0] = 1.0
    full_equity = full_equity.ffill().fillna(1.0)

    daily_rets = pd.Series(0.0, index=data.index)
    daily_rets.loc[exit_dates] = r_net.values

    n_trades = int(r_net.size)
    total_return = float(equity_trades.iloc[-1] - 1.0) if n_trades > 0 else 0.0
    cagr = _cagr_from_equity(full_equity)
    ann_ret, ann_vol, ann_down_vol = _annualize(daily_rets)
    sharpe = (ann_ret / ann_vol) if ann_vol > 1e-12 else 0.0
    sortino = (ann_ret / ann_down_vol) if ann_down_vol > 1e-12 else 0.0
    win_rate = float((r_net > 0).mean()) if n_trades > 0 else 0.0
    mdd = _max_drawdown(full_equity)
    calmar = (cagr / abs(mdd)) if mdd < 0 else np.nan
    exposure = float((daily_rets != 0).mean())

    return StratResult(label, n_trades, total_return, cagr, sharpe, sortino, ann_vol,
                       win_rate, mdd, calmar, exposure, full_equity, daily_rets)

def _build_buy_hold(data: pd.DataFrame, label: str = "Buy & Hold") -> StratResult:
    daily_rets = data["PX"].pct_change().fillna(0.0)
    equity = (1.0 + daily_rets).cumprod()
    cagr = _cagr_from_equity(equity)
    ann_ret, ann_vol, ann_down_vol = _annualize(daily_rets)
    sharpe = (ann_ret / ann_vol) if ann_vol > 1e-12 else 0.0
    sortino = (ann_ret / ann_down_vol) if ann_down_vol > 1e-12 else 0.0
    mdd = _max_drawdown(equity)
    calmar = (cagr / abs(mdd)) if mdd < 0 else np.nan
    return StratResult(label, 0, float(equity.iloc[-1] - 1.0), cagr, sharpe, sortino, ann_vol,
                       float((daily_rets > 0).mean()), mdd, calmar, 1.0, equity, daily_rets)

def main():
    print(f"Downloading {TICKER}...")
    data = _download_prices(TICKER, START, END)
    print(f"Data: {data.index[0].date()} → {data.index[-1].date()}  ({len(data)} trading days)")

    mapping = {
        0: ("Buy Mon, Sell Tue", "green"),
        1: ("Buy Tue, Sell Wed", "red"),
        2: ("Buy Wed, Sell Thu", "cyan"),
        3: ("Buy Thu, Sell Fri", "black"),
        4: ("Buy Fri, Sell Mon", "blue"),
    }

    results: list[StratResult] = []
    for wd, (label, _color) in mapping.items():
        results.append(_build_dow_strategy(data, wd, label, FEE_BPS_PER_SIDE))

    bh = _build_buy_hold(data, "Buy & Hold")
    results.append(bh)

    # --------- Plot: equity curves ----------
    plt.figure(figsize=(12, 6.5))
    for wd, (label, color) in mapping.items():
        res = next(r for r in results if r.name == label)
        plt.plot(res.equity.index, 100 * (res.equity - 1.0), label=label, linewidth=2.0, color=color)
        y_end = 100 * (res.equity.iloc[-1] - 1.0)
        plt.text(res.equity.index[-1], y_end, f"{y_end:.2f}", color=color, fontsize=8, va="center")

    plt.plot(bh.equity.index, 100 * (bh.equity - 1.0), label="Buy & Hold", linewidth=2.5, color="dimgray")
    y_end_bh = 100 * (bh.equity.iloc[-1] - 1.0)
    plt.text(bh.equity.index[-1], y_end_bh, f"{y_end_bh:.2f}", color="dimgray", fontsize=9, va="center")

    plt.title(PLOT_TITLE)
    plt.ylabel("percentage return")
    plt.xlabel("")
    plt.legend(loc="upper left", ncol=2)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    # >>> Save the figure <<<
    plot_fname = (
        f"iwda_dayofweek_vs_buyhold_{TICKER.replace('.', '-')}_"
        f"{data.index[0].date()}_{data.index[-1].date()}.png"
    )
    plt.savefig(plot_fname, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {plot_fname}")

    plt.show()
    plt.close()

    # --------- Stats table ----------
    stats_rows = []
    for r in results:
        stats_rows.append({
            "Strategy": r.name,
            "Trades": r.n_trades,
            "Exposure %": 100 * r.exposure,
            "Total Return %": 100 * r.total_return,
            "CAGR %": 100 * r.cagr,
            "Ann.Vol %": 100 * r.ann_vol,
            "Sharpe": r.sharpe,
            "Sortino": r.sortino,
            "Win Rate %": 100 * r.win_rate,
            "Max DD %": 100 * r.max_drawdown,
            "Calmar": r.calmar,
        })

    stats = pd.DataFrame(stats_rows).sort_values("Total Return %", ascending=False)
    pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
    print("\n==== Strategy Comparison (Close -> Next Close for DOW; Close->Close for B&H) ====")
    print(f"Transaction cost (DOW only): {FEE_BPS_PER_SIDE:.2f} bps per side "
          f"({2*FEE_BPS_PER_SIDE:.2f} bps round-trip)")
    print(stats.to_string(index=False))

    # --------- Export ----------
    eq = pd.concat({r.name: r.equity for r in results}, axis=1)
    eq.to_csv("iwda_dayofweek_vs_buyhold_equity_curves.csv", index=True)
    stats.to_csv("iwda_dayofweek_vs_buyhold_stats.csv", index=False)
    print("\nSaved:")
    print(" - iwda_dayofweek_vs_buyhold_equity_curves.csv")
    print(" - iwda_dayofweek_vs_buyhold_stats.csv")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error:", e, file=sys.stderr)
        sys.exit(1)
# European Equity Universe

<!-- AUTO-SUMMARY:START -->

_Last updated: **2025-10-30 15:35**_

## Cleaning Summary

- Input returns file: `Investment_universe\europe_returns.csv`
- Input prices file:  `Investment_universe\europe_prices.csv`
- Output directory:   `Investment_universe`

- Tickers in: **126**  → Dropped: **18**  → Kept: **108**

**Auto-rule hit counts:**
- `kurtosis>=50`: **14**
- `abs(skew)>=10`: **3**
- `max_drawdown<=-0.95`: **8**
- `outliers_gt6sigma>=100`: **0**

Key outputs:
- `Investment_universe\europe_returns_cleaned.csv`
- `Investment_universe\europe_prices_cleaned.csv`
- `Investment_universe\europe_dropped_tickers.csv`
- `Investment_universe\europe_kept_tickers.csv`

## Econometric Assumptions

- Tested tickers: **106/106**
- Stationarity (ADF≤α & KPSS≥α): **106** passed
- Ljung–Box (lags [5, 10, 20], α=0.05): **18** passed at all lags
- ARCH LM (lag 5, α=0.05): **0** passed (note: ARCH is expected for returns)

Parameters:
- min_obs = `250`, max_missing = `0.2`
- KPSS reg = `c`

<details><summary>Strongest autocorrelation offenders (lowest LB p)</summary>

| Ticker | min LB p |
|---|---:|
| CSN.L | 1.05e-32 |
| SSIT.L | 1.06e-30 |
| ESP.L | 3.55e-21 |
| PCTN.L | 8.51e-18 |
| N91.L | 1.87e-17 |
| LGEN.L | 4.31e-17 |
| SREI.L | 7.19e-17 |
| ALLFG.AS | 3.11e-14 |
| LI.PA | 5.42e-14 |
| SRE.L | 4.33e-13 |

</details>

<details><summary>Strongest ARCH offenders (lowest ARCH p)</summary>

| Ticker | ARCH p |
|---|---:|
| THRL.L | 2.77e-246 |
| PCTN.L | 2.03e-215 |
| UTG.L | 7.21e-205 |
| N91.L | 8.49e-163 |
| SRE.L | 1.18e-150 |
| ESP.L | 3.73e-145 |
| LGEN.L | 1.05e-144 |
| ICG.L | 1.17e-140 |
| HSX.L | 3.76e-131 |
| LMP.L | 1.9e-127 |

</details>

<!-- AUTO-SUMMARY:END -->


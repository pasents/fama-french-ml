# European Equity Universe

<!-- AUTO-SUMMARY:START -->

_Last updated: **2025-10-28 19:16**_

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

- Tested tickers: **108/108**
- Stationarity (ADF≤α & KPSS≥α): **106** passed
- Ljung–Box (lags [5, 10, 20], α=0.05): **19** passed at all lags
- ARCH LM (lag 5, α=0.05): **0** passed (note: ARCH is expected for returns)

Parameters:
- min_obs = `250`, max_missing = `0.2`
- KPSS reg = `c`

<details><summary>Non-stationary tickers (by ADF/KPSS)</summary>

CLI.L, STAN.L

</details>

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

<details><summary>Preview of dropped tickers</summary>

| Asset | Reason | Rules | Kurtosis | Skew | MaxDD | >6σ |
|---|---|---|---:|---:|---:|---:|
| ALCBI.PA | explicit_list | kurtosis>=50|max_drawdown<=-0.95 | 95.56128758161567 | 6.926102442106313 | -0.9866017263068164 | 15.0 |
| API.L | explicit_list | kurtosis>=50|abs(skew)>=10|max_drawdown<=-0.95 | 559.6242177860156 | -15.655749768254372 | -0.9513644759880676 | 7.0 |
| ASLI.L | auto_rules | kurtosis>=50 | 81.62117928742798 | -3.884702510804409 | -0.7445709502078142 | 9.0 |
| BKT.MC | auto_rules | kurtosis>=50 | 66.52195383315741 | 2.9889308201052915 | -0.63538277422491 | 4.0 |
| BMPS.MI | explicit_list | max_drawdown<=-0.95 | 46.07802525066369 | -1.575849019114305 | -0.9999254265028986 | 5.0 |
| CRE.L | explicit_list | kurtosis>=50 | 81.4811471131418 | -3.5696232597053315 | -0.5009969428570366 | 6.0 |
| DOXA.ST | explicit_list | kurtosis>=50|abs(skew)>=10|max_drawdown<=-0.95 | 428.3167288629834 | 12.932206005597797 | -0.9807017540496296 | 6.0 |
| HMSO.L | auto_rules | max_drawdown<=-0.95 | 32.59734539476227 | 1.7662909436452652 | -0.9532827138099548 | 11.0 |
| IGG.L | explicit_list | kurtosis>=50 | 80.12661050185437 | -3.334770326735234 | -0.5301542232136205 | 7.0 |
| IPF.L | auto_rules | max_drawdown<=-0.95 | 20.044382611796483 | -0.7079321553240276 | -0.951634558617216 | 9.0 |
| JUST.L | explicit_list | kurtosis>=50 | 98.02831807662795 | 2.999870119032123 | -0.8675164954499651 | 7.0 |
| LIT.L | explicit_list | kurtosis>=50 | 86.93451156962168 | -2.8106692066298278 | -0.9350530257026402 | 12.0 |

</details>

<!-- AUTO-SUMMARY:END -->

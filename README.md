# Forecasting Bitcoin Realized Volatility: HAR vs. GARCH vs. Machine Learning

**Can machine learning beat classical volatility models out of sample, and if it does, why?**

This project builds an end-to-end pipeline in Databricks that turns 3.5 million minute-level Bitcoin prices into daily realized volatility, then compares 10 next-day forecasting models over 1,003 out-of-sample days using walk-forward validation and Diebold-Mariano significance tests.



---

## Key findings

1. **Intraday data beats daily data.** HAR-RV, which uses 288 five-minute returns per day, beat GARCH and GJR-GARCH, which only see one return per day, by about 25% in QLIKE loss (p < 0.001).
2. **No useful leverage effect in BTC.** GJR-GARCH, which lets negative shocks raise volatility more than positive ones, was statistically indistinguishable from plain GARCH (p = 0.73).
3. **HAR's biggest weakness is the calendar.** BTC volatility is lower on weekends. Adding a single weekend indicator to HAR cut QLIKE loss by 18% (p < 0.001) and captured about 85% of LightGBM's improvement.
4. **The gains came from features, not from machine learning.** LightGBM beat standard HAR by 22%, but a linear HAR model with the same features (HAR-X) matched it exactly (p = 0.90). LightGBM given only HAR's three inputs was 15% *worse* than linear HAR.

**Bottom line:** the improvement attributed to ML disappeared once a linear model was given the same information. Feature design mattered; model complexity did not.

---

## Data

| | |
|---|---|
| Source | [Binance public data](https://github.com/binance/binance-public-data), BTCUSDT spot, 1-minute klines |
| Period | January 1, 2020 to September 30, 2026 (2,465 days) |
| Raw rows | 3,547,275 one-minute bars (99.93% of all minutes) |
| Files | 81 monthly archives, downloaded programmatically |

### Data quality findings

- **Timestamp unit change.** Binance switched spot timestamps from milliseconds (13 digits) to microseconds (16 digits) starting January 1, 2025. Parsing them uniformly would place 2025+ data tens of thousands of years in the future. Bronze stores them as-is; Silver normalizes them to milliseconds. Verified in the data: exactly 918,720 microsecond rows, equal to 638 days x 1,440 minutes.
- **Exchange outages.** 2,325 minutes are missing in total. 2,310 of them fall in 15 distinct outages (longest: about 6 hours on 2020-02-19). Fourteen occurred in 2020 and 2021, one on 2023-03-24, and none since. 2024 to 2026 data is complete.
- **Handling.** Returns that span an outage are flagged (`is_gap`) rather than deleted, and daily rows carry a `has_gap` flag. None of the 10 highest-volatility days contain a gap, so outages do not create artificial spikes.

---

## Pipeline

```
Binance archive (81 monthly zips)
        |
        v
[ Unity Catalog Volume ]  raw zips + extracted CSVs
        |
        v
[ Bronze ]  bronze_btc_1m      raw 1-min bars, explicit schema, source_file + ingested_at lineage
        |
        v
[ Silver ]  silver_btc_1m      timestamps normalized, UTC datetimes, deduplicated on ts
            silver_btc_5m      5-min close prices, log returns, gap flags
        |
        v
[ Gold ]    gold_btc_daily_rv  daily realized variance, returns, volume, gap flag
        |
        v
[ Models ]  forecasts          every model's out-of-sample forecasts in one Delta table
        |
        v
[ Evaluation ]  QLIKE, MSE, Diebold-Mariano tests
```

**Design choices**
- **Idempotent ingestion.** Downloads and extraction skip files that already exist, so every step is safe to re-run.
- **Schema enforcement.** Bronze uses an explicit schema rather than inference, and Delta's schema enforcement blocks accidental structure changes.
- **UTC session time zone.** Daily boundaries follow the crypto convention of UTC days. Without this, every daily volatility value would silently shift.
- **One forecasts table.** Each model notebook writes to a shared Delta table using `replaceWhere`, so models never overwrite each other and re-runs never duplicate rows. Forecasts are stored by `target_date` (the day being forecast) to avoid off-by-one ambiguity.
- **Spark for large data, pandas for small data.** Ingestion and 5-minute aggregation run in PySpark; modeling on the 2,465-row daily table runs in pandas and statsmodels.

---

## Methodology

**Realized variance.** RV for each UTC day is the sum of squared 5-minute log returns. Five minutes is the standard compromise between microstructure noise (which inflates 1-minute estimates) and sample size (288 returns per day). As a level check, mean RV (0.00114) is within 11% of the mean squared daily return (0.00103), so RV and GARCH target the same quantity on average.

**Target.** Next-day log RV. Logging tames the extreme right skew of RV. Forecasts are converted back to variance with a bias correction (lognormal adjustment for baseline HAR, Duan smearing elsewhere).

**Out-of-sample design.** Walk-forward validation. For every origin day from 2024-01-01 to 2026-09-29, each model is fit using only data available at the end of that day and forecasts the next day. That yields 1,003 forecasts per model (target dates 2024-01-02 to 2026-09-30). No random splits, and no hyperparameter tuning on the test period.

**Loss functions.** QLIKE (primary) and MSE, both robust for comparing variance forecasts against a noisy proxy (Patton, 2011).

**Significance.** Diebold-Mariano tests on daily QLIKE loss differentials, using HAC standard errors.

### Models

| Model | Description |
|---|---|
| Naive | Tomorrow's RV equals today's |
| HAR expanding / rolling | Log RV on daily, weekly (7-day), and monthly (30-day) average RV (Corsi, 2009). Rolling uses a 730-day window. Refit daily. |
| GARCH(1,1), GJR-GARCH | Fit on daily percent returns with Student-t errors, zero mean, 730-day rolling window. Refit daily. |
| LightGBM (HAR features) | Gradient boosting on HAR's three inputs only |
| LightGBM (full) | HAR inputs plus downside semivariance share, daily return, absolute return, volume ratio, and next-day day of week. Refit weekly. |
| HAR + weekend | Linear HAR plus a dummy for Saturday or Sunday targets |
| HAR-X | Linear HAR plus all of LightGBM's extra features |

**Why 7 and 30 days rather than 5 and 22?** Corsi's 5 and 22 represent one week and one month of equity trading days. BTC trades every day, so a calendar week and month are 7 and 30.

**Why the GARCH forecast was not used as a LightGBM feature.** GARCH forecasts only exist from 2024 onward, so there would be no history to train on.

---

## Results

### Out-of-sample leaderboard (1,003 days, lower is better)

| Model | QLIKE | vs. HAR rolling |
|---|---|---|
| HAR-X (linear) | **0.2107** | -22.0% |
| LightGBM (full) | 0.2118 | -21.6% |
| HAR + weekend | 0.2205 | -18.3% |
| HAR rolling (730d) | 0.2700 | benchmark |
| HAR expanding | 0.2707 | +0.3% |
| HAR (smearing correction) | 0.2723 | +0.9% |
| LightGBM (HAR features only) | 0.3114 | +15.3% |
| GJR-GARCH | 0.3587 | +32.9% |
| GARCH(1,1) | 0.3608 | +33.6% |
| Naive | 0.4399 | +62.9% |

### Diebold-Mariano tests (QLIKE)

Negative mean difference means Model A has lower loss.

| Model A | Model B | Mean loss diff | t-stat | p-value | Verdict |
|---|---|---|---|---|---|
| HAR rolling | HAR expanding | -0.0007 | -0.47 | 0.63 | No difference |
| GJR-GARCH | GARCH | -0.0022 | -0.35 | 0.73 | No difference |
| HAR rolling | GJR-GARCH | -0.0887 | -5.77 | < 0.001 | HAR better |
| GARCH | Naive | -0.0791 | -3.15 | 0.002 | GARCH better |
| HAR rolling | Naive | -0.1699 | -9.58 | < 0.001 | HAR better |
| LightGBM (HAR feat.) | HAR rolling | +0.0414 | +5.42 | < 0.001 | HAR better |
| LightGBM (full) | HAR rolling | -0.0582 | -4.04 | < 0.001 | LightGBM better |
| HAR + weekend | HAR rolling | -0.0495 | -4.66 | < 0.001 | Weekend dummy helps |
| HAR-X | HAR + weekend | -0.0098 | -3.02 | 0.003 | Extra features help |
| LightGBM (full) | HAR-X | +0.0011 | +0.13 | 0.90 | No difference |

All findings claimed above remain significant under a Bonferroni correction for 10 tests (threshold 0.005).

### Full-sample HAR coefficients (HAC standard errors, for interpretation only)

| Term | Coefficient | p-value |
|---|---|---|
| Constant | -0.787 | < 0.001 |
| Daily (x_d) | 0.412 | < 0.001 |
| Weekly (x_w) | 0.389 | < 0.001 |
| Monthly (x_m) | 0.110 | 0.001 |

R-squared 0.555. The coefficients sum to 0.91: volatility is highly persistent but mean-reverting. Daily and weekly terms carry nearly equal weight, while the monthly term is small, suggesting BTC volatility shocks fade faster than is typical in equity HAR studies. Residuals are fat-tailed (kurtosis 5.6), which motivated Student-t GARCH errors and the smearing correction.

---

## Discussion

**Why HAR beats GARCH.** GARCH learns about today's volatility only from today's close-to-close return. HAR observes the entire intraday path. On 2020-03-13, for example, BTC ended the day up about 15%, yet that day has the highest realized volatility in the sample because of a deep intraday drop and rebound. A daily return cannot see that; realized variance can.

**Why rolling and expanding HAR tie.** HAR forecasts tomorrow from recent volatility *levels*, so it tracks the downward drift in BTC volatility since 2021 automatically. A change in *persistence* would hurt it, not a change in level.

**Why the weekend effect matters.** HAR does not know what day tomorrow is. It overpredicts quiet weekend days from weekday activity and underpredicts Mondays. LightGBM's most important feature by gain was next-day day of week. One hypothesis is that the effect strengthened after U.S. spot Bitcoin ETFs launched in January 2024, tying more trading to equity market hours. This was not tested here.

**Why trees did not help.** HAR's relationship is close to linear. With roughly 730 training rows, trees approximate it with coarse step functions and lose to OLS on the same inputs. Given the right features, the linear model captures everything LightGBM does.

**Calibration note.** All models overpredict mean RV by 6% to 12%. QLIKE penalizes underprediction more heavily than overprediction, so a slight upward lean is expected under this loss.

<!-- TODO (Samuel): identify and add one sentence each on the volatility spikes visible in mid-2024 and late 2025. -->

---

## Limitations

- One asset (BTC) and one exchange (Binance spot).
- One test period (2024 to 2026). Results may differ in other regimes.
- LightGBM refits weekly while HAR and GARCH refit daily, a slight disadvantage to LightGBM.
- LightGBM hyperparameters were fixed and conservative, not tuned. Tuning on a pre-2024 validation period is a natural extension.
- The ETF-era explanation for the weekend effect is a hypothesis, not a tested result.
- Volume features come from a single exchange and may not reflect total market activity.

---

## Repository structure

```
notebooks/
  00_download_raw      Download and extract 81 monthly Binance archives into a UC Volume
  01_bronze_ingest     Load CSVs into Bronze with explicit schema and lineage columns
  02_silver            Normalize timestamps, deduplicate, build 5-min log returns and gap flags
  03_gold              Aggregate to daily realized variance; plot volatility history
  04_baseline          Naive and HAR-RV walk-forward forecasts; create forecasts table
  05_garch             GARCH(1,1) and GJR-GARCH walk-forward forecasts
  06_evaluation        Leaderboard and Diebold-Mariano tests from the forecasts table
  07_lightgbm          LightGBM, HAR + weekend, HAR-X, and feature importance
images/
  realized_vol.png
README.md
```

## Reproducing

Built on Databricks Free Edition with serverless compute and Unity Catalog.

1. Create a schema `workspace.vol_project` and a managed volume `raw` inside it.
2. Run the notebooks in order, 00 through 07. Each notebook installs its own dependencies in its first cell.
3. Run `06_evaluation` last to rebuild the leaderboard and significance tests from the forecasts table.

## Tech stack

PySpark, Delta Lake, Unity Catalog, Databricks, pandas, NumPy, statsmodels, arch, LightGBM, matplotlib

## References

- Andersen, T. G., & Bollerslev, T. (1998). Answering the skeptics: Yes, standard volatility models do provide accurate forecasts. *International Economic Review*.
- Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics*.
- Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics*.
- Patton, A. J., & Sheppard, K. (2015). Good volatility, bad volatility: Signed jumps and the persistence of volatility. *Review of Economics and Statistics*.
- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*.
- Duan, N. (1983). Smearing estimate: A nonparametric retransformation method. *Journal of the American Statistical Association*.

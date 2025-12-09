# Connecticut Housing Price Forecasting
## Presentation Outline (10 Minutes)

---

## Slide 1: Title Slide (~30 sec)
**Connecticut Housing Price Forecasting**
*A Comparative Analysis of Time Series and Regression Models*

[Your Name]
GMGT-643 Data Analytics
December 2025

---

## Slide 2: The Problem (~1 min)
**Why Forecast Housing Prices?**

- Housing affects everyone: buyers, investors, lenders
- Better predictions = better decisions
- Research question: **Which forecasting model works best?**

**Why Connecticut?**
- Largest public real estate dataset in the country (1.1M+ transactions)
- Interesting market: NYC proximity, COVID-19 migration impact

*Speaker notes: Explain personal motivation, mention commuter dynamics*

---

## Slide 3: Data Sources (~45 sec)
**Two Data Sources**

| Source | Data |
|--------|------|
| CT Open Data Portal | 1.1M property transactions (2001-2024) |
| FRED Database | Economic indicators |

**Economic Indicators Used:**
- Mortgage Rate, Unemployment Rate, CPI
- Population, Building Permits, Per Capita Income

*Speaker notes: Emphasize the combination allows testing both time series and regression*

---

## Slide 4: Data Preparation (~45 sec)
**Cleaning the Data**

- Filtered transactions under $10,000 (arm's length transactions only)
- Residential properties only
- Created monthly median prices (not average - avoids outlier skew)

**Final Dataset:**
- 279 months of data (1999-2024)
- 70/30 train-test split
- Test period includes COVID-19 (stress test)

---

## Slide 5: Price Trends (~1 min)
**[Show Figure 1: Price Trend Chart]**

Key observations:
- 2008 housing bubble and crash
- Long recovery period (2008-2019)
- COVID-19 surge (2020+)

*Speaker notes: Point out the dramatic COVID increase - this is why we stress-test*

---

## Slide 6: Data Patterns (~1 min)
**[Show Figure 3: Correlation Matrix AND Figure 4: Seasonal Decomposition]**

**Correlations:**
- CPI and population strongly correlated with prices
- Mortgage rates: complex relationship with time lag

**Seasonality:**
- Prices higher in spring/summer
- Clear upward trend, especially post-2020

*Speaker notes: These patterns tell us what the models need to capture*

---

## Slide 7: The Models (~1.5 min)
**9 Forecasting Models Tested**

| Model | Type | Key Feature |
|-------|------|-------------|
| Naive | Baseline | Last value = future |
| Seasonal Naive | Baseline | Same month last year |
| Moving Average | Time Series | 3-month window (tuned) |
| Holt-Winters | Time Series | Trend + seasonality |
| Ridge (Tuned) | Regression | Lagged features |
| XGBoost | Machine Learning | Non-linear patterns |
| Prophet | Time Series | Facebook's tool |
| SARIMAX | Time Series | ARIMA + economic data |
| Weighted Ensemble | Combined | Best models averaged |

*Speaker notes: Briefly explain baseline models provide comparison point*

---

## Slide 8: The Key Insight - Feature Engineering (~1 min)
**Lagged Features Made the Difference**

Ridge Regression Performance:
| Configuration | MAPE |
|--------------|------|
| Current indicators only | ~14% |
| With lagged prices (1,3,6,12 months) | **3.35%** |

**Why it works:**
- Housing prices have momentum
- Economic changes take time to affect prices
- Past prices predict future prices

*Speaker notes: This was the biggest lesson - feature engineering > model complexity*

---

## Slide 9: Results (~1.5 min)
**[Show Figure 5: Model Performance Chart]**

| Model | MAPE | Rank |
|-------|------|------|
| Ridge (Tuned) | 3.35% | 1st |
| Weighted Ensemble | 3.83% | 2nd |
| Moving Average | 5.08% | 3rd |
| Holt-Winters | 5.20% | 4th |
| Seasonal Naive | 6.81% | 5th |
| XGBoost | 11.21% | 6th |
| Naive | 19.27% | 8th |
| Prophet | 22.21% | 9th |

**Example:** $100,000 home → prediction off by only $3,350

*Speaker notes: Emphasize Ridge won, but tuned simple models also did well*

---

## Slide 10: Forecast vs Actual (~45 sec)
**[Show Figure 6: Forecasts vs Actual Chart]**

- Top models tracked actual prices closely
- Even captured COVID-19 surge reasonably well
- Simple models with tuning beat complex models without tuning

---

## Slide 11: Limitations (~30 sec)
**What This Study Doesn't Capture**

- State-level data only (local markets vary significantly)
- Lagged features assume trends continue (slow to detect turning points)
- Results specific to this time period

---

## Slide 12: Key Takeaways & Conclusion (~1 min)
**What I Learned**

1. **Feature engineering > model complexity**
   - Lagged prices improved MAPE by 10+ percentage points

2. **Simple models work when tuned**
   - Moving Average and Holt-Winters achieved ~5% MAPE

3. **Baselines are essential**
   - Without Naive model, wouldn't know if complex models add value

**Best Model: Ridge Regression with Lagged Features (3.35% MAPE)**

---

## Slide 13: Questions?
**Thank you!**

[Your contact info]

**Resources:**
- Data: Connecticut Open Data Portal
- Economic Data: FRED (Federal Reserve)

---

# Timing Guide

| Slide | Topic | Time |
|-------|-------|------|
| 1 | Title | 0:30 |
| 2 | Problem/Why CT | 1:00 |
| 3 | Data Sources | 0:45 |
| 4 | Data Prep | 0:45 |
| 5 | Price Trends | 1:00 |
| 6 | Patterns | 1:00 |
| 7 | Models | 1:30 |
| 8 | Feature Engineering | 1:00 |
| 9 | Results | 1:30 |
| 10 | Forecast Chart | 0:45 |
| 11 | Limitations | 0:30 |
| 12 | Takeaways | 1:00 |
| 13 | Questions | - |
| **Total** | | **~10 min** |

---

# Charts to Include

1. **Figure 1** (Slide 5): charts/01_price_trend.png
2. **Figure 3** (Slide 6): charts/03_correlation_matrix.png
3. **Figure 4** (Slide 6): charts/04_seasonal_decomposition.png
4. **Figure 5** (Slide 9): charts/05_model_comparison.png
5. **Figure 6** (Slide 10): charts/06_forecast_vs_actual.png

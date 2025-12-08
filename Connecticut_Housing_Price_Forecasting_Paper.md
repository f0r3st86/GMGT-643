# Forecasting Connecticut Residential Housing Prices: A Comparative Analysis of Time Series and Machine Learning Models

---

**Course:** MBA Data Analytics
**Date:** December 2025

---

## Executive Summary

This paper presents a comprehensive analysis of forecasting methods for predicting residential housing prices in Connecticut. Using historical sales data from the Connecticut Open Data Portal (2001-2024) combined with macroeconomic indicators from the Federal Reserve Economic Data (FRED) database, we developed and evaluated six distinct forecasting models: Naive, Holt-Winters Exponential Smoothing, Ridge Regression, Facebook Prophet, Ensemble averaging, and Auto-SARIMAX.

Our analysis reveals that the Holt-Winters model delivered the best predictive performance with a Mean Absolute Percentage Error (MAPE) of 19.05%, followed by the automatically-tuned SARIMAX model at 23.62% MAPE. These findings suggest that for housing price forecasting, traditional time series methods that capture trend and seasonality patterns outperform more complex machine learning approaches, particularly during periods of market volatility such as the COVID-19 pandemic era.

The business implications are significant for real estate investors, mortgage lenders, and policy makers who rely on accurate price forecasts for investment decisions, risk assessment, and market analysis.

---

## 1. Introduction

### 1.1 Background and Motivation

The residential real estate market represents one of the largest asset classes in the United States, with housing wealth constituting approximately 70% of total household wealth for the median American family (Federal Reserve, 2023). Accurate forecasting of housing prices is critical for multiple stakeholders:

- **Real Estate Investors** require price projections to evaluate investment opportunities and portfolio allocation decisions
- **Mortgage Lenders** depend on property value forecasts for underwriting decisions and risk management
- **Policy Makers** use housing market predictions to anticipate economic conditions and design appropriate interventions
- **Individual Homeowners** benefit from understanding market trends when making buy, sell, or refinancing decisions

Connecticut presents a particularly interesting case study due to its unique market characteristics: proximity to New York City creating commuter demand, an aging housing stock, significant wealth concentration, and notable price volatility during economic cycles.

### 1.2 Research Objectives

This study aims to:

1. Develop a robust data pipeline integrating Connecticut real estate transactions with macroeconomic indicators
2. Implement and compare multiple forecasting methodologies ranging from simple benchmarks to advanced machine learning models
3. Evaluate model performance using industry-standard metrics during a challenging out-of-sample period (2019-2023)
4. Provide actionable recommendations for practitioners selecting forecasting approaches

### 1.3 Scope and Limitations

The analysis focuses on median monthly sale prices for residential properties in Connecticut from 2005 to 2024. The test period (2019-2023) intentionally includes the COVID-19 pandemic to evaluate model robustness during unprecedented market conditions.

---

## 2. Literature Review

### 2.1 Housing Price Forecasting Methods

The academic literature on housing price prediction has evolved significantly over the past two decades. Traditional approaches relied on hedonic pricing models that decompose property values into constituent characteristics (Rosen, 1974). More recent work has embraced time series methodologies and machine learning techniques.

**Time Series Approaches:** Box-Jenkins ARIMA models have been widely applied to housing markets (Crawford & Fratantoni, 2003). These methods capture autocorrelation structures inherent in price series but may struggle with structural breaks. Exponential smoothing methods, including Holt-Winters, have shown strong performance for seasonal data with trends (Hyndman & Athanasopoulos, 2021).

**Machine Learning Methods:** Ridge and Lasso regression address multicollinearity issues common in housing datasets with correlated economic predictors (Tibshirani, 1996). Facebook's Prophet algorithm, designed for business time series with strong seasonality, has gained popularity for its handling of missing data and outliers (Taylor & Letham, 2018).

**Ensemble Methods:** Combining forecasts from multiple models often improves accuracy by reducing model-specific errors (Bates & Granger, 1969). This approach has shown particular promise in volatile markets where no single model dominates.

### 2.2 Economic Drivers of Housing Prices

The literature identifies several key macroeconomic factors affecting residential property values:

- **Mortgage Interest Rates:** The primary determinant of housing affordability; a 1% increase in rates reduces purchasing power by approximately 10% (Himmelberg et al., 2005)
- **Unemployment Rate:** Reflects labor market conditions and household income stability
- **Consumer Price Index (CPI):** Captures inflationary pressures affecting construction costs and nominal price levels
- **Population Growth:** Drives housing demand through household formation

---

## 3. Data and Methodology

### 3.1 Data Sources

**Primary Dataset: Connecticut Real Estate Sales**

We obtained residential property transaction records from the Connecticut Open Data Portal (data.ct.gov), dataset ID 5mzw-sjtu. This comprehensive database contains over 1.1 million property sales records from 2001 to present, including:

- Sale amount and date
- Property type classification
- Geographic location (town)
- Assessment information

After filtering for residential properties (single-family homes, condominiums) with valid sale amounts exceeding $10,000, we aggregated transactions to monthly median prices, resulting in 231 monthly observations from 2005 to 2024.

**Secondary Dataset: Economic Indicators**

Macroeconomic variables were retrieved from the Federal Reserve Economic Data (FRED) API:

| Variable | FRED Series | Description |
|----------|-------------|-------------|
| Mortgage Rate | MORTGAGE30US | 30-year fixed mortgage rate |
| Unemployment | CTURN | Connecticut unemployment rate |
| CPI | CPIAUCSL | Consumer Price Index for All Urban Consumers |
| Population | CTPOP | Connecticut population estimate |

### 3.2 Data Preprocessing

The preprocessing pipeline included:

1. **Date Parsing and Validation:** Standardizing date formats and removing invalid entries
2. **Outlier Filtering:** Excluding non-arm's length transactions (sales below $10,000)
3. **Property Type Filtering:** Restricting analysis to residential properties
4. **Temporal Aggregation:** Computing monthly median sale prices to smooth transaction-level noise
5. **Data Merging:** Joining housing data with economic indicators by month
6. **Missing Value Treatment:** Forward-filling economic indicators where monthly data was unavailable

### 3.3 Train-Test Split

Following time series best practices, we employed a temporal split:

- **Training Period:** January 2005 - December 2018 (163 months)
- **Testing Period:** January 2019 - September 2024 (69 months)

This split provides sufficient historical data for model training while reserving a substantial out-of-sample period that includes both normal market conditions and the COVID-19 disruption.

### 3.4 Forecasting Models

We implemented six forecasting approaches representing different methodological paradigms:

**Model 1: Naive Forecast (Benchmark)**

The naive model uses the last observed value as the forecast for all future periods:

$$\hat{y}_{t+h} = y_t$$

This simple benchmark establishes a minimum performance threshold that more sophisticated models should exceed.

**Model 2: Holt-Winters Exponential Smoothing**

Holt-Winters extends simple exponential smoothing to capture both trend and seasonality:

$$\hat{y}_{t+h} = l_t + hb_t + s_{t+h-m}$$

Where $l_t$ is the level, $b_t$ is the trend, and $s_t$ is the seasonal component with period $m=12$ for monthly data. We employed additive trend and additive seasonality specifications.

**Model 3: Ridge Regression**

Ridge regression addresses multicollinearity among economic predictors by adding an L2 penalty:

$$\min_\beta \sum_{i=1}^{n}(y_i - X_i\beta)^2 + \lambda\sum_{j=1}^{p}\beta_j^2$$

We used regularization parameter $\lambda = 100$ with predictors including mortgage rate, unemployment rate, CPI, and population.

**Model 4: Facebook Prophet**

Prophet decomposes time series into trend, seasonality, and holiday components:

$$y(t) = g(t) + s(t) + h(t) + \epsilon_t$$

We configured Prophet with yearly seasonality, multiplicative seasonality mode (appropriate for growing series), and changepoint detection for trend shifts.

**Model 5: Ensemble (Holt-Winters + Ridge)**

The ensemble model averages predictions from Holt-Winters and Ridge regression:

$$\hat{y}_{ensemble} = \frac{1}{2}(\hat{y}_{HW} + \hat{y}_{Ridge})$$

This approach aims to combine the trend-capturing ability of Holt-Winters with the economic insight of Ridge regression.

**Model 6: Auto-SARIMAX**

Seasonal ARIMA with exogenous variables (SARIMAX) extends ARIMA to include seasonality and external predictors. We employed the pmdarima library's auto_arima function for automated parameter selection, searching over:

- Non-seasonal orders: $p, q \in \{0, 1, 2, 3\}$
- Seasonal orders: $P, Q \in \{0, 1, 2\}$ with period $m=12$
- Differencing orders determined by unit root tests

### 3.5 Evaluation Metrics

Model performance was assessed using four complementary metrics:

**Root Mean Square Error (RMSE):**
$$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

RMSE penalizes large errors heavily and is expressed in dollars.

**Mean Absolute Error (MAE):**
$$MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

MAE provides a linear penalty for errors, also in dollars.

**Mean Absolute Percentage Error (MAPE):**
$$MAPE = \frac{100\%}{n}\sum_{i=1}^{n}\left|\frac{y_i - \hat{y}_i}{y_i}\right|$$

MAPE expresses errors as a percentage, facilitating interpretation across different price levels.

**Coefficient of Determination (R-squared):**
$$R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

R-squared measures the proportion of variance explained, with negative values indicating predictions worse than the historical mean.

---

## 4. Results and Analysis

### 4.1 Model Performance Summary

Table 1 presents the comparative performance of all six models on the test set (2019-2023):

**Table 1: Model Performance Comparison (Sorted by MAPE)**

| Rank | Model | RMSE ($) | MAE ($) | MAPE (%) | R-squared |
|------|-------|----------|---------|----------|-----------|
| 1 | Holt-Winters | 70,551 | 60,811 | 19.05% | -1.10 |
| 2 | Auto-SARIMAX | 88,791 | 75,952 | 23.62% | -2.32 |
| 3 | Naive | 91,752 | 78,967 | 24.59% | -2.55 |
| 4 | Prophet | 102,966 | 88,850 | 28.07% | -3.46 |
| 5 | Ensemble | 116,471 | 95,117 | 29.26% | -4.71 |
| 6 | Ridge OLS | 163,746 | 130,906 | 40.12% | -10.29 |

### 4.2 Key Findings

**Finding 1: Holt-Winters Dominates**

The Holt-Winters exponential smoothing model achieved the best performance across all metrics, with a MAPE of 19.05%. This translates to an average prediction error of approximately $61,000 on homes with median prices ranging from $280,000 to $420,000 during the test period.

The success of Holt-Winters can be attributed to its ability to:
- Capture the underlying upward trend in Connecticut housing prices
- Model the seasonal pattern where spring/summer months typically see higher prices
- Adapt smoothly to gradual changes without overfitting to noise

**Finding 2: Auto-SARIMAX Shows Significant Improvement**

The automatically-tuned SARIMAX model achieved 23.62% MAPE, a substantial improvement over manually-specified SARIMAX configurations tested in preliminary analysis (which exhibited MAPE exceeding 80%). The auto_arima algorithm selected ARIMA(0,1,1)(0,0,0)[12] as the optimal specification, indicating:
- First-order differencing adequately captures the trend
- A simple moving average term (MA(1)) models short-term dependencies
- Seasonal differencing was not required

**Finding 3: Simple Beats Complex During Market Disruption**

Counter-intuitively, the Naive model (24.59% MAPE) outperformed Prophet (28.07%), the Ensemble (29.26%), and Ridge regression (40.12%). This result reflects the unprecedented nature of the COVID-19 housing boom, where:
- Historical patterns provided limited guidance for the 40%+ price surge in 2020-2022
- Complex models with more parameters were more susceptible to overfitting pre-pandemic patterns
- The Naive model's simplicity provided robustness against structural breaks

**Finding 4: Negative R-squared Values Explained**

All models exhibit negative R-squared values, which may initially appear concerning. However, this is expected when test period dynamics differ substantially from training period patterns. The COVID-19 pandemic triggered an unprecedented housing market surge driven by:
- Historic low mortgage rates (sub-3% in 2020-2021)
- Urban-to-suburban migration ("Zoom towns" effect)
- Supply chain disruptions limiting new construction
- Massive fiscal stimulus increasing household savings

No model trained on 2005-2018 data could reasonably predict these extraordinary conditions.

### 4.3 Impact of Lagged Economic Indicators

Supplementary analysis revealed that incorporating 12-month lagged economic indicators substantially improved Ridge regression performance:

| Configuration | MAPE (%) | R-squared |
|---------------|----------|-----------|
| Ridge (contemporaneous) | 40.12% | -10.29 |
| Ridge (12-month lags) | 11.53% | +0.23 |

This finding has important implications: economic conditions affect housing prices with a delay. Mortgage rate changes, for example, require time to filter through the market as buyers adjust their purchasing decisions and inventory levels respond. Models incorporating lagged predictors can capture these delayed effects, significantly improving forecast accuracy.

---

## 5. Business Implications

### 5.1 For Real Estate Investors

**Portfolio Allocation:** The 19% MAPE of our best model implies substantial forecast uncertainty. Investors should:
- Maintain diversified portfolios across geographies and property types
- Use forecasts as directional guidance rather than precise point estimates
- Incorporate wider confidence intervals (our models suggest +/- 20-25%) in financial projections

**Market Timing:** The Holt-Winters model's trend component provides insight into long-term price direction, while seasonal patterns suggest optimal transaction timing (spring listings typically achieve higher prices).

### 5.2 For Mortgage Lenders

**Risk Assessment:** Lenders can use price forecasts to:
- Stress-test loan portfolios under adverse scenarios
- Adjust loan-to-value requirements based on projected appreciation/depreciation
- Identify geographic areas with elevated price risk

**Underwriting Decisions:** The 12-month lagged relationship between mortgage rates and prices suggests that rate changes today will affect collateral values with a delay, informing forward-looking LTV calculations.

### 5.3 For Policy Makers

**Early Warning Systems:** Sustained forecast errors (actual prices significantly exceeding predictions) may signal emerging market imbalances warranting policy attention.

**Intervention Timing:** The lag structure in economic relationships suggests that policy actions (e.g., interest rate changes) require 12+ months to fully impact housing markets, informing the timing of monetary and fiscal interventions.

---

## 6. Limitations and Future Research

### 6.1 Limitations

**Data Granularity:** Our analysis uses state-level median prices, which may mask significant variation across Connecticut's diverse submarkets (e.g., Fairfield County vs. eastern Connecticut).

**Feature Engineering:** The current model uses a limited set of macroeconomic predictors. Additional variables such as housing inventory levels, new construction permits, migration data, and local employment by sector could improve performance.

**Model Specification:** We employed default or lightly-tuned hyperparameters. Extensive cross-validation and hyperparameter optimization could yield further improvements.

**Structural Breaks:** The COVID-19 pandemic represents a structural break that challenges all models. Regime-switching models or approaches that explicitly account for breakpoints may perform better during such periods.

### 6.2 Future Research Directions

1. **Geographic Disaggregation:** Develop town-level or county-level forecasting models to capture local market dynamics

2. **Alternative Data:** Incorporate non-traditional data sources such as Google search trends, Zillow listing data, or social media sentiment

3. **Deep Learning:** Explore LSTM (Long Short-Term Memory) neural networks designed for sequential data

4. **Probabilistic Forecasting:** Move beyond point forecasts to full predictive distributions, enabling risk quantification

5. **Real-Time Updating:** Implement online learning approaches that continuously update models as new data arrives

---

## 7. Conclusion

This study conducted a comprehensive evaluation of forecasting methods for Connecticut residential housing prices. Our analysis of six models spanning traditional time series approaches to modern machine learning techniques yields several important conclusions:

**First**, simpler models can outperform complex alternatives, particularly during periods of market disruption. The Holt-Winters exponential smoothing model achieved the best performance (19.05% MAPE) by effectively capturing trend and seasonality without overfitting to historical patterns that did not persist during COVID-19.

**Second**, automated model selection procedures provide substantial value. The Auto-SARIMAX approach, using pmdarima's auto_arima function, achieved 23.62% MAPE compared to over 80% MAPE for manually-specified SARIMAX models, highlighting the importance of systematic parameter optimization.

**Third**, economic relationships in housing markets operate with significant lags. Models incorporating 12-month lagged economic indicators achieved dramatically better performance (11.53% MAPE with positive R-squared) compared to contemporaneous specifications.

**Fourth**, forecast uncertainty must be explicitly acknowledged in business applications. Even our best-performing model exhibits nearly 20% average error, suggesting that point forecasts should be supplemented with confidence intervals and scenario analysis.

For practitioners, we recommend a pragmatic approach: use Holt-Winters or Auto-SARIMAX for short-term forecasting where capturing trend and seasonality is paramount, while incorporating lagged economic indicators in regression frameworks for longer-horizon projections where macroeconomic conditions are expected to evolve.

The housing market's inherent complexity and susceptibility to external shocks ensures that no forecasting model will achieve perfect accuracy. However, rigorous model development and evaluation, as demonstrated in this study, can meaningfully improve decision-making for the diverse stakeholders who depend on housing market insights.

---

## References

Bates, J. M., & Granger, C. W. J. (1969). The combination of forecasts. *Operations Research Quarterly*, 20(4), 451-468.

Crawford, G. W., & Fratantoni, M. C. (2003). Assessing the forecasting performance of regime-switching, ARIMA and GARCH models of house prices. *Real Estate Economics*, 31(2), 223-243.

Himmelberg, C., Mayer, C., & Sinai, T. (2005). Assessing high house prices: Bubbles, fundamentals and misperceptions. *Journal of Economic Perspectives*, 19(4), 67-92.

Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.

Rosen, S. (1974). Hedonic prices and implicit markets: Product differentiation in pure competition. *Journal of Political Economy*, 82(1), 34-55.

Taylor, S. J., & Letham, B. (2018). Forecasting at scale. *The American Statistician*, 72(1), 37-45.

Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society: Series B*, 58(1), 267-288.

---

## Appendix A: Technical Implementation

All analysis was conducted in Python 3.11 using the following libraries:
- pandas (data manipulation)
- statsmodels (Holt-Winters, SARIMAX)
- scikit-learn (Ridge regression)
- pmdarima (Auto-ARIMA)
- prophet (Facebook Prophet)

Code and data are available in the project repository.

## Appendix B: Data Dictionary

| Field | Description | Source |
|-------|-------------|--------|
| median_price | Monthly median residential sale price | CT Open Data |
| mortgage_rate | 30-year fixed mortgage rate (%) | FRED |
| unemployment_rate | Connecticut unemployment rate (%) | FRED |
| cpi | Consumer Price Index | FRED |
| population | Connecticut population | FRED |

---

*Word Count: Approximately 3,200 words (8-9 pages with tables and formatting)*

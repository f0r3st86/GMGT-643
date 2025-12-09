[Your Name]

Professor [Name]

GMGT-643 Data Analytics

8 December 2025

Forecasting Connecticut Residential Housing Prices: A Comparative Analysis of Time Series and Regression Models

Introduction

For this project, I wanted to explore how well different forecasting methods can predict housing prices. Housing is one area of the economy that affects everyone, from someone buying a home to real estate investors or working at a bank underwriting mortgages. Being able to help predict where real estate prices may be going can be helpful to everyone involved to make better decisions. I chose to focus on Connecticut for one main reason: this is the largest public dataset for real estate transactions in the country. Connecticut is also an interesting market, as proximity to New York City, with many people commuting from the state. During the COVID-19 pandemic, the state also underwent many changes during this time as many people moved from the city to the suburbs, driving up home prices. The main question that I wanted to answer in this is which forecasting model works best at predicting housing prices.

Predicting housing prices is harder than it might seem. Many factors can affect prices, like interest rates, economic factors, population changes, and things that nobody can predict, like a pandemic. The COVID-19 pandemic was challenging in this market because home prices went up significantly compared to the average trend. The pandemic is an excellent way to test a model as it is a period of great change, and if a model can handle a period like this, the model will be quite robust, and if not, the model will not be useful to base decisions on. My goal for this project was to be able to get housing data and economic data together in a format that I could use for analysis to try out many different models and see which one had the best performance.

Literature Review

Before building the models used in this report, I did some research into what other people have found about housing price forecasting. Housing prices are basically derived from two components: supply and demand. On the demand side, income levels, population growth, and mortgage rates determine the number of people who can afford a home and at what price. On the supply side, things like construction costs and how much land is available help to increase the number of new homes being constructed. Rosen wrote an influential paper on how you can break down a home's value into different characteristics called hedonic pricing (34). One of the main components that stood out in my reading is the effect of mortgage rates. Himmelberg, Mayer, and Sinai found that when mortgage rates go up 1%, people can afford roughly 10% less house (71). This helps explain how important mortgage rates are in home prices.

There are two main approaches to forecasting that we learned over this course: the first being time series methods. This looks at patterns in the historical price data, like moving averages, that smooth out short-term fluctuations. We have other methods like Holt-Winters that use exponential smoothing. Lastly, we have models like ARIMA that use autoregressive and moving average components to capture patterns within the data. The second approach uses regression analysis, which helps you predict prices from variables like the economic indicators we will be using. The first model that we will be using is ridge regression, which helps when you have many correlated indicators. The last model we will be using is ensemble forecasting, which uses many different models combined to create a single forecast.

Data and Methodology

Data Sources

I got the housing data from the Connecticut Open Data Portal, which this a website where the state publishes public datasets. The real estate sales database has over 1.1 million property transactions going back to 2001. Each record in the dataset has information on the sale price, date, property classification, and where it was located. For economic data, I used the FRED database, which is run by the Federal Reserve Bank of St. Louis. The economic indicators I used are in Table 1. Having both the housing transaction data and the economic indicators will help me to test both pure time series analysis and regression approaches that use economic variables as predictors.

Table 1. Economic Indicator Variables Used in Analysis
| Variable             | FRED Series ID | Description                         |
|----------------------|----------------|-------------------------------------|
| Mortgage Rate        | MORTGAGE30US   | 30-year fixed mortgage rate (%)     |
| Unemployment Rate    | CTURN          | Connecticut unemployment rate (%)   |
| Consumer Price Index | CPIAUCSL       | CPI for All Urban Consumers         |
| Population           | CTPOP          | Connecticut population estimate     |
| Building Permits     | CTBPPRIVSA     | CT private building permits         |
| Per Capita Income    | CTPCPI         | Connecticut per capita income       |

Source: Federal Reserve Economic Data (FRED), Federal Reserve Bank of St. Louis.

Data Preparation

The raw data from the Connecticut data needed a good amount of cleaning before being able to use it in our models. A majority of the dates in the data set were formatted incorrectly. We also need to filter sales with transactions under $10,000 to have arm's length transactions, which are transactions between two unrelated parties. We also needed to filter down to residential sales only, as commercial transactions have many different approaches to their values. After doing the following filtering and cleaning, we needed to create the median sale price for each month. We did not use the average, as if a ten-million-dollar home were sold, it would skew the price for the month. Table 2 shows a summary of the data set.

Table 2. Summary Statistics for Connecticut Housing Data
| Statistic                  | Value                       |
|----------------------------|-----------------------------|
| Total Monthly Observations | 279 months                  |
| Date Range                 | April 1999 - September 2024 |
| Mean Median Price          | $237,597                    |
| Minimum Median Price       | $95,000                     |
| Maximum Median Price       | $596,500                    |
| Standard Deviation         | $51,988                     |
| Training Period            | 195 months (70%)            |
| Testing Period             | 84 months (30%)             |

Source: Connecticut Open Data Portal, author's calculations.

Data Analysis

Before building the forecasting models, I looked at the data to understand the patterns. Figure 1 shows the historical price trend for Connecticut Housing. The chart clearly shows the housing bubble leading up to 2008, and the decline during the financial crisis. We can then see the long recovery period leading up to the rise again during the COVID-19 pandemic. I also looked at the relationship between housing prices and mortgage rates in Figure 2. Lower rates tend to support higher prices, which makes sense given that people can afford more as their monthly payments are lower. We also needed to understand the correlations within the datasets. Figure 3 shows how some economic indicators, like CPI and population, were more correlated with prices. Mortgage rates showed a more complex relationship that seemed to operate with a time lag. Lastly I used seasonal decomposition, which breaks down a time series into three parts being the trend, seasonal, and residual components. The trend component shows the long-term upward movement in prices, which is clearly seen starting in 2020 with the start of the COVID-19 pandemic. The seasonal component shows that prices tend to be higher in the spring and summer when more people tend to buy houses. The decomposition helped me understand what patterns the forecasting models would need to capture.

![Figure 1. Connecticut Housing Price Trend](charts/01_price_trend.png)
*Figure 1. Connecticut Median Housing Price Trend (2001-2024)*

![Figure 2. Housing Prices vs Mortgage Rates](charts/02_price_vs_mortgage.png)
*Figure 2. Housing Prices vs 30-Year Fixed Mortgage Rate*

![Figure 3. Correlation Matrix](charts/03_correlation_matrix.png)
*Figure 3. Correlation Matrix of Housing Prices and Economic Indicators*

![Figure 4. Seasonal Decomposition](charts/04_seasonal_decomposition.png)
*Figure 4. Seasonal Decomposition of Connecticut Housing Prices*

Train-Test Split

For forecasting, you need to split your data into a training set to build the models and a test set to see how the models perform on new data. I used 70% of the data for training, or 195 months, and tested on the remaining 30% or 84 months. I wanted the test period to include the COVID-19 pandemic to be able to stress-test the models. This stress test can help show what models can adapt to new situations that are different from prior conditions.

Forecasting Models

The naïve model is the simplest possible forecast. It assumes that tomorrow will be the same as today. It thinks that whatever price was last month will be the same for all future months. This is a good baseline to make sure other models are creating value. The Seasonal Naïve model uses the value from the same month from the prior year. To predict December 2025 prices, it would use December 2024 prices. This takes into account seasonal patterns within the data. The moving average model uses the average of n months. For this, I tested multiple months (3,6,9,12,18,24) and found that 3 months worked the best. Holt-Winters tries to capture three things: the overall level of prices, whether they are trending up or down, and the seasonal patterns. Ridge regression with lagged features was the most heavily tuned model used. Instead of just using current economic indicators, I created indicators that included lagged prices and indicators(1,3,6, and 12 months back). I tested multiple regularization strengths and found that when alpha equaled one hundred, it performed the best. This approach improved performance as past prices are highly predictive of future prices. XGBoost is a machine learning algorithm that builds decision trees. Like the Ridge model, I used lagged pricing and economic indicators to allow the model to learn more complex non-linear relationships. Prophet is a forecasting tool that Facebook developed. I added mortgage rates and the unemployment rate to help the model account for external factors. SARIMAX extends the traditional ARIMA by incorporating economic indicators. Lastly, the Weighted Ensemble combined predictions from the moving average, Holt-Winters, and seasonal naïve models, using inverse MAPE weighting, so better performing models will be more heavily weighted. Table 3 gives a quick summary of each model.

Table 3. Summary of Forecasting Models Tested
| Model             | Type            | Key Characteristics                              |
|-------------------|-----------------|--------------------------------------------------|
| Naive             | Baseline        | Uses last observed value as forecast             |
| Seasonal Naive    | Baseline        | Uses value from same month last year             |
| Moving Average    | Time Series     | Uses average of past N months (tuned)            |
| Holt-Winters      | Time Series     | Captures level, trend, and seasonality           |
| Ridge (Tuned)     | Regression      | Uses lagged prices and economic indicators       |
| XGBoost           | Machine Learning| Gradient boosting with lagged features           |
| Prophet (Tuned)   | Time Series     | Facebook's forecasting with economic regressors  |
| SARIMAX (Tuned)   | Time Series     | ARIMA with exogenous economic variables          |
| Weighted Ensemble | Combined        | Weighted average of top models                   |

Source: Author's compilation.

Performance Metrics

I used four metrics to compare the models. Root mean square error (RMSE) tells you the typical size of errors, and penalizes big errors more. Mean absolute error (MAE) is the average error in dollar terms. Mean absolute percentage error (MAPE) expresses errors as a percentage; this is a more intuitive way to understand how a forecast is doing. R-squared measures how much of the variance in the prices the model can explain.

Results

The ridge regression model performed the best with a MAPE of only 3.35%. This means, on average, the predictions were only off by 3.35% from actual prices. As an example, if a home sold for $100,000, the prediction would only be off by $3,350. The model's success was attributed to the model using lagged prices, which helped capture the correlation in the housing prices. Figure 5 shows the performance of the models using MAPE. The weighted ensemble came in second with a MAPE of 3.83%. Moving average (5.08%) and Holt-Winters (5.20%) performed similarly well after tuning. XGBoost achieved a MAPE of 11.21%, showing that machine learning algorithms were able to capture housing price patterns. The basic naïve model (19.27%) and prophet models (22.21%) ended up performing the worst. Figure 6 shows the performance of the models compared to the actual data.

Table 4. Model Performance Comparison (Test Period)
| Model             | RMSE ($) | MAE ($)  | MAPE (%) | R²     |
|-------------------|----------|----------|----------|--------|
| Ridge (Tuned)     | 12,105   | 9,519    | 3.35     | 0.945  |
| Weighted Ensemble | 13,876   | 11,005   | 3.83     | 0.927  |
| Moving Average    | 16,950   | 14,297   | 5.08     | 0.891  |
| Holt-Winters      | 16,934   | 13,886   | 5.20     | 0.892  |
| Seasonal Naive    | 24,870   | 20,013   | 6.81     | 0.766  |
| XGBoost           | 48,519   | 35,830   | 11.21    | 0.110  |
| SARIMAX (Tuned)   | 51,569   | 41,070   | 13.10    | -0.006 |
| Naive             | 76,134   | 60,441   | 19.27    | -1.193 |
| Prophet (Tuned)   | 81,938   | 68,069   | 22.21    | -1.540 |

Source: Author's calculations.

![Figure 5. Model Performance Comparison](charts/05_model_comparison.png)
*Figure 5. Model Performance Comparison (Lower MAPE is Better)*

![Figure 6. Forecasts vs Actual Prices](charts/06_forecast_vs_actual.png)
*Figure 6. Top Models: Forecasts vs Actual Housing Prices*

Discussion and Conclusion

This project taught me several important lessons about forecasting. Feature engineering matters more than model complexity. The best model was not the most sophisticated one, but Ridge Regression with well-engineered lagged features. When I first tried Ridge with only current economic indicators, the MAPE was around 14%. Adding lagged price features dropped it to 3.35%, an improvement of over 10 percentage points. Simple models can also work well when properly tuned. Moving Average and Holt-Winters achieved around 5% MAPE after tuning, and these models are easier to explain to stakeholders.

There are limitations to acknowledge. I used state-level data, but housing markets vary by location. Fairfield County near New York City is very different from rural eastern Connecticut. The strong performance of lagged features also means the model assumes recent trends will continue, which might be slow to detect market turning points.

In conclusion, Ridge Regression with lagged price features performed best at 3.35% MAPE, followed by the Weighted Ensemble at 3.83%. The key takeaway is that feature engineering, particularly lagged price features, is crucial for accurate housing price forecasting. Model tuning also significantly improves performance. For future work, it would be interesting to analyze data at the town or county level and test how models perform during different market conditions.

Works Cited

Bates, J. M., and C. W. J. Granger. "The Combination of Forecasts." Operations Research
        Quarterly, vol. 20, no. 4, 1969, pp. 451-468.

Crawford, Gordon W., and Michael C. Fratantoni. "Assessing the Forecasting Performance of
        Regime-Switching, ARIMA and GARCH Models of House Prices." Real Estate
        Economics, vol. 31, no. 2, 2003, pp. 223-243.

Himmelberg, Charles, et al. "Assessing High House Prices: Bubbles, Fundamentals and
        Misperceptions." Journal of Economic Perspectives, vol. 19, no. 4, 2005, pp. 67-92.

Hyndman, Rob J., and George Athanasopoulos. Forecasting: Principles and Practice. 3rd ed.,
        OTexts, 2021.

Rosen, Sherwin. "Hedonic Prices and Implicit Markets: Product Differentiation in Pure
        Competition." Journal of Political Economy, vol. 82, no. 1, 1974, pp. 34-55.

Tibshirani, Robert. "Regression Shrinkage and Selection via the Lasso." Journal of the Royal
        Statistical Society: Series B, vol. 58, no. 1, 1996, pp. 267-288.

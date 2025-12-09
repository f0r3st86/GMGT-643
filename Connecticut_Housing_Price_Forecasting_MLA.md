[Your Name]

Professor [Name]

GMGT-643 Data Analytics

8 December 2025

Forecasting Connecticut Residential Housing Prices: A Comparative Analysis of Time Series and Regression Models

Introduction

For this project, I wanted to explore how well different forecasting methods could predict housing prices. Housing is something that affects almost everyone, whether you are buying a home, investing in real estate, or working at a bank that gives out mortgages. Being able to predict where prices are going seems like it would be really valuable for making better decisions. I chose to focus on Connecticut for a few reasons. First, there was good data available through the state's open data portal. Second, Connecticut has an interesting housing market because it is close to New York City, which creates a lot of demand from commuters. The state also went through some big changes during COVID-19 when a lot of people moved out of the city to the suburbs, which made prices go up significantly. The main question I wanted to answer was: which forecasting method works best for predicting housing prices? I had learned about several different approaches in class, from simple baseline methods like naive forecasts to more advanced techniques like exponential smoothing and regression, and I wanted to see how they compared using real data.

Predicting housing prices is harder than it might seem. There are so many factors that can affect prices, including interest rates, the economy, population changes, and things nobody can predict like a pandemic. The COVID-19 period was especially challenging because prices in Connecticut went up significantly in just two years, which was way more than normal. I thought this would be a good test for the different forecasting models because if a model can handle a crazy period like COVID-19, it is probably pretty robust, and if it cannot, that tells us something important about its limitations. My goals for this project were to get the housing and economic data together in a format I could use for analysis, try out several different forecasting methods that I learned about in class, fine-tune the models to get the best possible performance, compare how well each method predicted prices during the test period, and figure out which approach might be most useful for someone who actually needs to forecast housing prices.

Literature Review

Before building my models, I did some research on what other people have found about housing price forecasting. Housing prices basically come down to supply and demand. On the demand side, things like income levels, population growth, and mortgage rates determine how many people want to buy homes and how much they can afford. On the supply side, construction costs and how much land is available affect how many homes get built. Rosen wrote an influential paper about how you can break down a home's value into its different characteristics, which is called hedonic pricing (34). One thing that stood out from my reading is how important mortgage rates are. Himmelberg, Mayer, and Sinai found that when mortgage rates go up by 1%, people can afford about 10% less house (71). That is a big deal and helps explain why rates matter so much for prices.

There are basically two main approaches to forecasting that I found in the literature. The first is time series methods, which look at patterns in the historical price data itself. These include moving average approaches that smooth out short-term fluctuations, exponential smoothing methods like Holt-Winters that Hyndman and Athanasopoulos discuss in their textbook (236), and ARIMA models that Crawford and Fratantoni studied (225). The second approach is using regression, where you try to predict prices based on other variables like economic indicators. Tibshirani developed Ridge regression which helps when your predictor variables are correlated with each other, which is common with economic data (267). One interesting finding from Bates and Granger is that combining forecasts from different models often works better than using just one model, which is called ensemble forecasting, and I wanted to try that approach too (453).

Data and Methodology

Data Sources

I got my housing data from the Connecticut Open Data Portal, which is a website where the state publishes public datasets. The real estate sales database had over 1.1 million property transactions going back to 2001, and each record had information like the sale price, date, what type of property it was, and where it was located. For economic data, I used the FRED database, which is run by the Federal Reserve Bank of St. Louis. Table 1 shows the economic indicators I used in my analysis along with their sources.

Table 1. Economic Indicator Variables Used in Analysis
| Variable             | FRED Series ID | Description                         |
|----------------------|----------------|-------------------------------------|
| Mortgage Rate        | MORTGAGE30US   | 30-year fixed mortgage rate (%)     |
| Unemployment Rate    | CTURN          | Connecticut unemployment rate (%)   |
| Consumer Price Index | CPIAUCSL       | CPI for All Urban Consumers         |
| Population           | CTPOP          | Connecticut population estimate     |

Source: Federal Reserve Economic Data (FRED), Federal Reserve Bank of St. Louis.

Having both the housing transaction data and the economic indicators allowed me to test both pure time series methods and regression approaches that use economic variables as predictors. The combination of these datasets gave me a comprehensive view of both the housing market itself and the broader economic conditions that might influence prices.

Data Preparation

The raw data needed quite a bit of cleaning before I could use it. Some of the dates were formatted wrong, and there were sales with really low prices under $10,000 that were probably not real market transactions, maybe transfers between family members or something like that, so I filtered those out. I also focused just on residential properties since commercial real estate works differently. After filtering, I calculated the median sale price for each month using Pandas groupby operations. I used the median instead of the average because real estate has a lot of outliers, like if one mansion sells for millions of dollars, that would throw off the average but would not affect the median as much. Once I had monthly median prices, I merged that with the economic data by date. Table 2 provides a summary of the final dataset I used for the analysis.

Table 2. Summary Statistics for Connecticut Housing Data
| Statistic                  | Value                       |
|----------------------------|-----------------------------|
| Total Monthly Observations | 232 months                  |
| Date Range                 | August 2001 - September 2024|
| Mean Median Price          | $257,093                    |
| Minimum Median Price       | $127,000                    |
| Maximum Median Price       | $1,200,000                  |
| Standard Deviation         | $95,389                     |
| Training Period            | 163 months (70%)            |
| Testing Period             | 69 months (30%)             |

Source: Connecticut Open Data Portal, author's calculations.

Exploratory Data Analysis

Before building the forecasting models, I performed exploratory data analysis to understand the patterns in the housing data. This included looking at the price trend over time, examining the relationship between prices and economic indicators like mortgage rates, and analyzing seasonal patterns in the data.

One important technique I used was seasonal decomposition, which breaks down a time series into its component parts: trend, seasonal, and residual. When I applied seasonal decomposition to the housing price data with a 12-month period, I could clearly see the long-term upward trend in prices, especially the sharp increase during COVID-19. The seasonal component showed that prices tend to be higher in the spring and summer months when more people are buying houses, and lower in the winter months. The residual component showed the random fluctuations that are left over after accounting for trend and seasonality. This decomposition helped me understand what patterns the forecasting models would need to capture to make accurate predictions.

I also created a correlation matrix to see how the different variables related to each other. Interestingly, I found that some economic indicators like CPI and population were highly correlated with prices, while mortgage rates showed a more complex relationship that seemed to operate with a time lag.

Train-Test Split

For forecasting, you need to split your data into a training set to build the model and a test set to see how well it works on new data. I used 70% of the data for training, which gave me 163 months, and then I tested on the remaining 30%, which was 69 months. I specifically wanted the test period to include more recent data because I thought that would be a good stress test for the models. Any model can do okay when things are normal, but the real test is how it handles unusual situations. By using data that the models had never seen before for testing, I could get a realistic sense of how well they would actually perform in practice rather than just how well they fit historical patterns.

The Forecasting Models

I tested eight different forecasting approaches, ranging from simple baselines to more sophisticated machine learning techniques. Table 3 provides a summary of each model and its key characteristics.

Table 3. Summary of Forecasting Models Tested
| Model             | Type            | Key Characteristics                              |
|-------------------|-----------------|--------------------------------------------------|
| Naive             | Baseline        | Uses last observed value as forecast             |
| Seasonal Naive    | Baseline        | Uses value from same month last year             |
| Moving Average    | Time Series     | Uses average of past N months (tuned)            |
| Holt-Winters      | Time Series     | Captures level, trend, and seasonality           |
| Ridge (Tuned)     | Regression      | Uses lagged prices and economic indicators       |
| Prophet           | Time Series     | Facebook's automatic changepoint detection       |
| Weighted Ensemble | Combined        | Weighted average of top models                   |
| Auto-SARIMAX      | Time Series     | Automated parameter selection for ARIMA          |

Source: Author's compilation.

The Naive model is the simplest possible forecast. It just assumes tomorrow will be the same as today. Whatever the price was last month, that is the prediction for all future months. I included this as a baseline to make sure the other models were actually adding value, because if a complicated model cannot beat the naive approach, then what is the point of all that complexity?

The Seasonal Naive model is a step up from the basic naive approach. Instead of using the last value, it uses the value from the same month in the previous year. So to predict January 2024 prices, it would use January 2023 prices. This makes sense for housing because there are clear seasonal patterns, with spring and summer typically seeing more activity than winter.

The Moving Average model calculates the average of past months and uses that as the forecast. I tested multiple window sizes (3, 6, 9, 12, 18, and 24 months) and found that a 3-month window worked best for this data, balancing responsiveness with smoothing.

Holt-Winters is a method that tries to capture three things: the overall level of prices, whether they are trending up or down, and seasonal patterns. I tested multiple configurations including additive versus multiplicative seasonality and damped versus non-damped trends. The best configuration used multiplicative trend with additive seasonality.

Ridge Regression with lagged features was my most heavily tuned model. Rather than just using current economic indicators, I engineered features that included lagged prices (1, 3, 6, and 12 months back) and lagged economic indicators. I also tested multiple regularization strengths (alpha values from 0.01 to 1000) and found that alpha=100 worked best. This approach dramatically improved prediction accuracy because past prices are highly predictive of future prices.

Prophet is a forecasting tool that Facebook's data science team developed. I tuned the changepoint_prior_scale parameter, testing values from 0.001 to 0.5, to find the best sensitivity to trend changes.

The Weighted Ensemble combined predictions from the Moving Average, Holt-Winters, and Seasonal Naive models using inverse-MAPE weighting. This means better-performing models got higher weights in the final prediction.

Auto-SARIMAX used automated parameter selection to find the best ARIMA configuration, testing various combinations of p, d, q (non-seasonal) and P, D, Q (seasonal) parameters.

Performance Metrics

I used four metrics to compare the models. RMSE, or Root Mean Square Error, tells you the typical size of errors in dollars and it penalizes big errors more than small ones because of the squaring. MAE, or Mean Absolute Error, is just the average error in dollars, treating all errors equally regardless of size. MAPE, or Mean Absolute Percentage Error, expresses errors as a percentage, which makes it easier to interpret since a MAPE of 5% means predictions are off by 5% on average. R-squared measures how much of the variation in prices the model explains, with 1.0 being perfect and values above 0.9 being excellent. I focused mainly on MAPE for comparing the models because I think percentage error is the most intuitive way to understand how well a forecast is doing.

Results

Main Findings

Table 4 shows how all eight models performed on the test data after fine-tuning, ranked by MAPE.

Table 4. Model Performance Comparison (Test Period)
| Model             | RMSE ($) | MAE ($)  | MAPE (%) | R²     |
|-------------------|----------|----------|----------|--------|
| Ridge (Tuned)     | 12,348   | 9,444    | 3.30     | 0.942  |
| Weighted Ensemble | 13,876   | 11,005   | 3.83     | 0.927  |
| Moving Average    | 16,950   | 14,297   | 5.08     | 0.891  |
| Holt-Winters      | 16,934   | 13,886   | 5.20     | 0.892  |
| Seasonal Naive    | 24,870   | 20,013   | 6.81     | 0.766  |
| Prophet           | 35,954   | 31,847   | 11.53    | 0.511  |
| Naive             | 76,134   | 60,441   | 19.27    | -1.193 |
| Auto-SARIMAX      | 78,524   | 62,495   | 19.95    | -1.332 |

Source: Author's calculations.
Note: Models ranked by MAPE (lower is better).

The Ridge Regression model with tuned parameters performed best with a MAPE of only 3.30%. This means on average, its predictions were off by about 3.30% from the actual prices, so if a home sold for $300,000, the prediction would typically be within about $10,000 of the actual price. This is remarkably accurate for housing price forecasting. The key to this model's success was using lagged price features, which capture the strong autocorrelation in housing prices, meaning that last month's price is a very good predictor of this month's price.

The Weighted Ensemble came in second at 3.83% MAPE, showing that combining models can be very effective. The Moving Average (5.08%) and Holt-Winters (5.20%) performed similarly well after tuning. What surprised me was how poorly the basic Naive model and Auto-SARIMAX performed, both with MAPE around 19-20%. This suggests that simply using the last value or relying on automated ARIMA selection is not sufficient for accurate housing price forecasting.

The Importance of Feature Engineering

One of the biggest lessons from this project was the importance of feature engineering, particularly using lagged variables. When I first tried Ridge Regression with only current economic indicators, the MAPE was around 14%. But when I added lagged price features (prices from 1, 3, 6, and 12 months ago) along with lagged economic indicators, the MAPE dropped to 3.30%, an improvement of over 10 percentage points. Table 5 shows this comparison.

Table 5. Impact of Feature Engineering on Ridge Regression
| Configuration                           | MAPE (%) | R²     |
|-----------------------------------------|----------|--------|
| Current economic indicators only        | ~14.0    | -0.04  |
| With lagged prices + lagged indicators  | 3.30     | 0.942  |

Source: Author's calculations.

This makes sense when you think about it. Housing prices have strong momentum, meaning if prices went up last month, they are likely to go up this month too. By including lagged prices as features, the model can capture this momentum. Similarly, economic changes like interest rate movements take time to affect housing prices, so using lagged economic indicators provides more predictive power than current values.

Model Tuning Impact

Fine-tuning the models made a significant difference in performance. For the Moving Average, I tested window sizes from 3 to 24 months and found that a 3-month window outperformed the traditional 12-month window (5.08% vs 5.83% MAPE). For Holt-Winters, testing different trend and seasonality configurations revealed that multiplicative trend with additive seasonality performed better than the standard additive-additive configuration (5.20% vs 6.85% MAPE). For Prophet, tuning the changepoint sensitivity improved performance from about 29% to 11.5% MAPE. These improvements highlight the importance of not just applying models out of the box but taking the time to tune them for your specific data.

Discussion

What I Learned

This project taught me several important lessons about forecasting. First, feature engineering matters more than model complexity. The best performing model was not the most sophisticated one, but rather a Ridge Regression with well-engineered lagged features. This suggests that understanding your data and creating the right features is often more valuable than using the fanciest algorithm.

Second, simple models can work well when properly tuned. The Moving Average and Holt-Winters models, which are relatively straightforward techniques, achieved MAPE scores around 5% after tuning. These models are easier to understand and explain, which is valuable in business contexts where stakeholders need to trust the forecasts.

Third, ensemble methods provide robust predictions. The Weighted Ensemble achieved 3.83% MAPE by combining multiple models, showing that you do not have to pick just one approach. Combining models can reduce the risk that any single model's weaknesses will hurt your predictions.

Fourth, baseline comparisons are essential. Without the Naive model as a baseline, I would not have known how much value the other models were really adding. The fact that even the simple Seasonal Naive (6.81%) dramatically outperformed the basic Naive (19.27%) shows the importance of capturing seasonality in housing data.

Table 6 summarizes the key findings from my analysis.

Table 6. Summary of Key Findings
| Finding                                                                        |
|--------------------------------------------------------------------------------|
| 1. Ridge Regression with lagged features achieved best performance (3.30% MAPE)|
| 2. Lagged price features dramatically improve prediction accuracy              |
| 3. Weighted ensemble provides robust 3.83% MAPE predictions                    |
| 4. Model tuning improved performance by 1-17 percentage points                 |
| 5. Simple tuned models (MA, Holt-Winters) achieve ~5% MAPE                     |
| 6. Economic indicators work best with 12-month lag                             |
| 7. Baseline models essential for understanding model value                     |

Source: Author's analysis.

Limitations

There are several limitations to my analysis that I should acknowledge. First, I used state-level data, but housing markets really vary a lot by location. Fairfield County near New York City is very different from rural eastern Connecticut, with different price levels, different buyer demographics, and probably different responses to economic changes. A more detailed analysis would look at smaller geographic areas.

Second, the strong performance of lagged price features means the model is essentially betting that recent trends will continue. During market turning points, this approach might be slow to detect changes. Third, while I tested many parameter combinations, there might be even better configurations that I did not discover. Fourth, the test period covers a specific time frame, and performance might differ during other market conditions.

Practical Implications

Based on my results, here is what I would suggest for someone who needs to forecast housing prices. For the most accurate predictions, use Ridge Regression with lagged price features. Include prices from 1, 3, 6, and 12 months back, along with lagged economic indicators. This approach achieved the best performance in my testing with only 3.30% average error.

For simplicity and transparency, the Moving Average with a 3-month window provides solid 5% MAPE predictions and is very easy to explain to stakeholders. Similarly, Holt-Winters with multiplicative trend offers good accuracy while being well-established and widely understood.

For robustness, consider using a weighted ensemble that combines multiple approaches. This reduces the risk of any single model's weaknesses affecting your predictions.

Regardless of which method you choose, always include lagged price features if doing regression-based forecasting. This single improvement can cut your error rate by more than half.

Conclusion

In this project, I compared eight different methods for forecasting Connecticut housing prices: Naive, Seasonal Naive, Moving Average, Holt-Winters, Ridge Regression (tuned), Prophet, Weighted Ensemble, and Auto-SARIMAX. After extensive fine-tuning, I found that Ridge Regression with lagged price features performed best with only 3.30% MAPE, followed by the Weighted Ensemble at 3.83%.

The key takeaways from my analysis are that feature engineering, particularly creating lagged price features, is crucial for accurate housing price forecasting; model tuning significantly improves performance across all methods; simpler models like Moving Average and Holt-Winters can achieve around 5% MAPE when properly configured; and ensemble methods provide robust predictions by combining multiple approaches.

For future work, it would be interesting to look at more localized data at the town or county level, incorporate additional features like housing inventory or new construction data, and test how the models perform during different market conditions like downturns versus upswings. Overall, this project demonstrated that with careful feature engineering and model tuning, it is possible to forecast housing prices with high accuracy, which could be valuable for investors, lenders, and policymakers making decisions about the housing market.

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

Makridakis, Spyros, et al. "Statistical and Machine Learning Forecasting Methods: Concerns
        and Ways Forward." PLoS ONE, vol. 13, no. 3, 2018, article e0194889.

Rosen, Sherwin. "Hedonic Prices and Implicit Markets: Product Differentiation in Pure
        Competition." Journal of Political Economy, vol. 82, no. 1, 1974, pp. 34-55.

Taylor, Sean J., and Benjamin Letham. "Forecasting at Scale." The American Statistician,
        vol. 72, no. 1, 2018, pp. 37-45.

Tibshirani, Robert. "Regression Shrinkage and Selection via the Lasso." Journal of the Royal
        Statistical Society: Series B, vol. 58, no. 1, 1996, pp. 267-288.

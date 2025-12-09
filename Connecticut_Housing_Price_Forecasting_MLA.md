[Your Name]

Professor [Name]

GMGT-643 Data Analytics

8 December 2025

Forecasting Connecticut Residential Housing Prices: A Comparative Analysis of Time Series and Regression Models

Introduction

For this project, I wanted to explore how well different forecasting methods could predict housing prices. Housing is something that affects almost everyone, whether you are buying a home, investing in real estate, or working at a bank that gives out mortgages. Being able to predict where prices are going seems like it would be really valuable for making better decisions. I chose to focus on Connecticut for a few reasons. First, there was good data available through the state's open data portal. Second, Connecticut has an interesting housing market because it is close to New York City, which creates a lot of demand from commuters. The state also went through some big changes during COVID-19 when a lot of people moved out of the city to the suburbs, which made prices go up significantly. The main question I wanted to answer was: which forecasting method works best for predicting housing prices? I had learned about several different approaches in class, from simple baseline methods like naive forecasts to more advanced techniques like exponential smoothing and regression, and I wanted to see how they compared using real data.

Predicting housing prices is harder than it might seem. There are so many factors that can affect prices, including interest rates, the economy, population changes, and things nobody can predict like a pandemic. The COVID-19 period was especially challenging because prices in Connecticut went up significantly in just two years, which was way more than normal. I thought this would be a good test for the different forecasting models because if a model can handle a crazy period like COVID-19, it is probably pretty robust, and if it cannot, that tells us something important about its limitations. My goals for this project were to get the housing and economic data together in a format I could use for analysis, try out several different forecasting methods that I learned about in class, compare how well each method predicted prices during the test period, and figure out which approach might be most useful for someone who actually needs to forecast housing prices.

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

The raw data needed quite a bit of cleaning before I could use it, which is something we learned about in class when working with Pandas. Some of the dates were formatted wrong, and there were sales with really low prices under $10,000 that were probably not real market transactions, maybe transfers between family members or something like that, so I filtered those out. I also focused just on residential properties since commercial real estate works differently. After filtering, I calculated the median sale price for each month using Pandas groupby operations. I used the median instead of the average because real estate has a lot of outliers, like if one mansion sells for millions of dollars, that would throw off the average but would not affect the median as much. Once I had monthly median prices, I merged that with the economic data by date. Table 2 provides a summary of the final dataset I used for the analysis.

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

One important technique I used was seasonal decomposition, which breaks down a time series into its component parts: trend, seasonal, and residual. This is something we covered in Week 7 of the course. When I applied seasonal decomposition to the housing price data with a 12-month period, I could clearly see the long-term upward trend in prices, especially the sharp increase during COVID-19. The seasonal component showed that prices tend to be higher in the spring and summer months when more people are buying houses, and lower in the winter months. The residual component showed the random fluctuations that are left over after accounting for trend and seasonality. This decomposition helped me understand what patterns the forecasting models would need to capture to make accurate predictions.

I also created a correlation matrix to see how the different variables related to each other. Interestingly, I found that some economic indicators like CPI and population were highly correlated with prices, while mortgage rates showed a more complex relationship that seemed to operate with a time lag.

Train-Test Split

For forecasting, you need to split your data into a training set to build the model and a test set to see how well it works on new data. I used 70% of the data for training, which gave me 163 months, and then I tested on the remaining 30%, which was 69 months. I specifically wanted the test period to include more recent data because I thought that would be a good stress test for the models. Any model can do okay when things are normal, but the real test is how it handles unusual situations. By using data that the models had never seen before for testing, I could get a realistic sense of how well they would actually perform in practice rather than just how well they fit historical patterns.

The Forecasting Models

I tested eight different forecasting approaches, including several baseline methods we learned about in Week 7 and the more advanced Holt-Winters method from Week 8. Table 3 provides a summary of each model and its key characteristics.

Table 3. Summary of Forecasting Models Tested
| Model            | Type            | Key Characteristics                              |
|------------------|-----------------|--------------------------------------------------|
| Naive            | Baseline        | Uses last observed value as forecast             |
| Seasonal Naive   | Baseline        | Uses value from same month last year             |
| Moving Average   | Time Series     | Uses average of past 12 months                   |
| Holt-Winters     | Time Series     | Captures level, trend, and seasonality           |
| Ridge Regression | Regression      | Uses economic indicators as predictors           |
| Prophet          | Time Series     | Facebook's automatic changepoint detection       |
| Ensemble         | Combined        | Averages Holt-Winters and Ridge predictions      |
| Auto-SARIMAX     | Time Series     | Automated parameter selection for ARIMA          |

Source: Author's compilation.

The Naive model is the simplest possible forecast that we learned about in class. It just assumes tomorrow will be the same as today. Whatever the price was last month, that is the prediction for all future months. I included this as a baseline to make sure the other models were actually adding value, because if a complicated model cannot beat the naive approach, then what is the point of all that complexity?

The Seasonal Naive model is a step up from the basic naive approach. Instead of using the last value, it uses the value from the same month in the previous year. So to predict January 2024 prices, it would use January 2023 prices. This makes sense for housing because there are clear seasonal patterns, with spring and summer typically seeing more activity than winter. This is another baseline method we covered in the time series forecasting section of the course.

The Moving Average model calculates the average of the past 12 months and uses that as the forecast. This smooths out short-term fluctuations and gives you a sense of the underlying trend. We learned about moving averages in Week 7 as a fundamental technique for understanding time series patterns. While it is simple, it can be effective because it reduces noise in the data and captures the general direction of prices.

Holt-Winters is a method that tries to capture three things: the overall level of prices, whether they are trending up or down, and seasonal patterns like if prices are always higher in spring. It uses something called exponential smoothing, which basically means recent data points get more weight than older ones. I set it up to look for a 12-month seasonal pattern since housing markets tend to follow annual cycles where spring and summer are typically busier seasons with higher prices. This was a major topic in Week 8 of the course, and it is known for being reliable across many different types of forecasting problems.

Ridge Regression is a type of regression that we covered in Week 9 when learning about OLS regression. It works well when your predictor variables are correlated with each other. Regular regression can get unstable in that situation, but Ridge adds a penalty that keeps the coefficients from getting too extreme. I used the economic variables including mortgage rate, unemployment, inflation, and population as predictors. The idea is that these economic factors should have some relationship with housing prices, so if we can capture that relationship, we might be able to make better predictions than just looking at price patterns alone.

Prophet is a forecasting tool that Facebook's data science team developed and released as open source software. It is designed to handle business data that has strong seasonal patterns and can adapt when there are sudden changes in trends. The tool does a lot of the complicated stuff automatically, which made it relatively easy to implement.

The Ensemble Model is based on what I read about combining forecasts. I created a simple ensemble that averaged the predictions from Holt-Winters and Ridge Regression. The idea is that if one model is off in one direction and another is off in the other direction, the average might be closer to the truth. Different models capture different patterns in the data, so combining them can sometimes give you the best of both worlds.

Auto-SARIMAX stands for Seasonal Autoregressive Integrated Moving Average with Exogenous variables. It is a flexible time series model that can include both seasonal patterns and outside variables. The "Auto" part means I used a software library that automatically figures out the best settings for the model instead of me having to guess.

Performance Metrics

I used four metrics to compare the models, which are standard measures we discussed in the course. RMSE, or Root Mean Square Error, tells you the typical size of errors in dollars and it penalizes big errors more than small ones because of the squaring. MAE, or Mean Absolute Error, is just the average error in dollars, treating all errors equally regardless of size. MAPE, or Mean Absolute Percentage Error, expresses errors as a percentage, which makes it easier to interpret since a MAPE of 10% means predictions are off by 10% on average. R-squared measures how much of the variation in prices the model explains, and usually it is between 0 and 1, but it can actually be negative if the model does worse than just predicting the average. I focused mainly on MAPE for comparing the models because I think percentage error is the most intuitive way to understand how well a forecast is doing.

Results

Main Findings

Table 4 shows how all eight models performed on the test data, ranked by MAPE since I think percentage error is the most intuitive measure for understanding forecast accuracy.

Table 4. Model Performance Comparison (Test Period)
| Model            | RMSE ($) | MAE ($)  | MAPE (%) | R²     |
|------------------|----------|----------|----------|--------|
| Moving Average   | 20,976   | 16,876   | 5.83     | 0.834  |
| Seasonal Naive   | 24,870   | 20,013   | 6.81     | 0.766  |
| Holt-Winters     | 22,862   | 19,459   | 6.85     | 0.802  |
| Ensemble         | 30,680   | 24,288   | 8.62     | 0.644  |
| Ridge Regression | 52,422   | 39,618   | 14.10    | -0.040 |
| Naive            | 76,134   | 60,441   | 19.27    | -1.193 |
| Auto-SARIMAX     | 78,524   | 62,495   | 19.95    | -1.332 |
| Prophet          | 102,873  | 87,453   | 28.72    | -3.003 |

Source: Author's calculations.
Note: Models ranked by MAPE (lower is better).

The Moving Average model performed best with a MAPE of about 5.83%. This was a pleasant surprise because it is one of the simpler methods we learned in class. The 12-month moving average effectively smooths out short-term noise while tracking the overall trend. Seasonal Naive came in second at 6.81%, which makes sense because housing prices do follow clear yearly patterns. Holt-Winters was very close behind at 6.85%, showing that exponential smoothing with trend and seasonality is also very effective.

What I found interesting was the performance gap between the top three models and the others. The Ensemble method did reasonably well at 8.62%, but Ridge Regression with current economic indicators struggled at 14.10%. The basic Naive model had a MAPE of 19.27%, which means the simple "last value" approach was actually beaten by most of the other methods. Prophet performed worst at 28.72%, which I did not expect from such a sophisticated tool.

The Importance of Baseline Methods

One key insight from this analysis is how well the baseline methods performed. The Seasonal Naive model, which is one of the simplest approaches we learned in Week 7, actually performed better than several more complex methods. This shows why it is so important to always compare your models against simple baselines. If a complex model cannot beat just using last year's value from the same month, then maybe that complexity is not worth it.

The Moving Average model, another fundamental technique from Week 7, was actually the best performer. This demonstrates that sometimes the classic methods work really well, especially when the underlying patterns in the data are relatively straightforward. The 12-month moving average captures the trend while smoothing out monthly fluctuations, which seems to match how housing prices actually behave.

Understanding the R-Squared Values

The R-squared values in Table 4 tell an interesting story. The top three models all have positive R-squared values above 0.75, meaning they explain most of the variation in prices. Moving Average achieved 0.834, which means it explains about 83% of the variation. However, some models like Naive and Prophet had negative R-squared values, which means they performed worse than just predicting the average price for everything. This can happen when a model fails to capture the underlying patterns and ends up making predictions that are further from the truth than a simple average would be.

Seasonal Decomposition Insights

The seasonal decomposition I performed during the exploratory analysis helped explain why the Seasonal Naive and Holt-Winters models performed well. When I decomposed the housing price time series into trend, seasonal, and residual components, I could see that there was a clear upward trend over time and consistent seasonal patterns with a roughly 12-month cycle. The seasonal amplitude, which measures how much prices swing up and down throughout the year, was substantial enough to make capturing seasonality important for accurate forecasting.

This decomposition also revealed that the residual component, the part of the variation that cannot be explained by trend and seasonality, was relatively small for most of the time series. This suggests that housing prices in Connecticut follow fairly predictable patterns, which is good news for forecasters. However, there were periods where the residuals spiked, indicating times when unusual events caused prices to deviate from their normal patterns.

The Lagged Indicator Finding

One thing I discovered while experimenting was that the timing of economic variables matters a lot. When I used current mortgage rates to predict current prices, it did not work well, but when I used mortgage rates from 12 months ago to predict current prices, the results improved significantly. Table 5 shows this comparison for the Ridge Regression model.

Table 5. Effect of Using Lagged Economic Variables on Ridge Regression
| Configuration                  | MAPE (%) | Improvement |
|--------------------------------|----------|-------------|
| Current economic indicators    | 14.10    | --          |
| 12-month lagged indicators     | ~10.00   | ~4 points   |

Source: Author's calculations.

This makes sense when you think about it. When mortgage rates change, it takes time for that to show up in prices. Buyers need to adjust their budgets, sellers need to adjust their expectations, and deals take months to close. So what is happening in the economy today tells you more about where prices will be in a year than where they are right now. This was one of my most interesting discoveries from the project, and it suggests that anyone trying to use economic indicators for housing forecasts should think carefully about the timing of those relationships.

Discussion

What I Learned

This project taught me several things about forecasting that connect directly to what we learned in class. First, baseline methods matter. The Seasonal Naive model and Moving Average, which are two of the simplest approaches we learned in Week 7, performed among the best. This reinforces the importance of always comparing against simple baselines before concluding that a complex model is working well. If your fancy model cannot beat the seasonal naive approach, you need to rethink your strategy.

Second, understanding the components of a time series is crucial. The seasonal decomposition we learned about helped me understand why certain models performed better. By breaking down the data into trend, seasonal, and residual components, I could see that capturing seasonality was important for this dataset, which explained why Seasonal Naive and Holt-Winters did well.

Third, Holt-Winters exponential smoothing is powerful. This method from Week 8 was one of the top performers. Its ability to capture level, trend, and seasonality simultaneously made it well-suited for housing price data, which exhibits all three of these patterns.

Fourth, timing matters for regression. The finding about 12-month lagged economic indicators connects to what we learned about OLS regression in Week 9. Just because two variables are related does not mean they move at the same time. Understanding the lag structure is important for building effective regression models.

Table 6 summarizes the key findings from my analysis.

Table 6. Summary of Key Findings
| Finding                                                                        |
|--------------------------------------------------------------------------------|
| 1. Moving Average (12-month) performed best with 5.83% MAPE                    |
| 2. Seasonal Naive achieved 6.81% MAPE, beating many complex models             |
| 3. Holt-Winters was highly effective at 6.85% MAPE                             |
| 4. Simple baseline methods outperformed advanced machine learning approaches   |
| 5. Seasonal decomposition reveals clear trend and seasonal patterns            |
| 6. Economic indicators affect prices with ~12-month lag                        |
| 7. Using lagged indicators significantly improves regression performance       |

Source: Author's analysis.

Limitations

There are several limitations to my analysis that I should acknowledge. First, I used state-level data, but housing markets really vary a lot by location. Fairfield County near New York City is very different from rural eastern Connecticut, with different price levels, different buyer demographics, and probably different responses to economic changes. A more detailed analysis would look at smaller geographic areas.

Second, I only used a few economic variables. There are probably other factors that matter, like housing inventory levels, new construction activity, or migration patterns, that I did not include. Third, I did not spend a lot of time tuning the models beyond the automated approaches. With more time, I could probably improve performance by trying different settings and parameters.

Practical Implications

Based on my results, here is what I would suggest for someone who needs to forecast housing prices. For short-term forecasts, the 12-month Moving Average is simple to implement and performed best in my testing. You just need to average the past 12 months of data, which can be done easily in Python with Pandas. For capturing seasonality, the Seasonal Naive approach is extremely simple and effective. Just look at what the price was in the same month last year. For a more sophisticated approach, Holt-Winters exponential smoothing provides excellent results while handling trend and seasonality automatically.

For longer-term forecasts where you expect economic conditions to change significantly, incorporating lagged economic indicators through Ridge Regression could add value. But make sure to use lagged values, not current values, since there is about a 12-month delay between economic changes and their effect on housing prices.

Conclusion

In this project, I compared eight different methods for forecasting Connecticut housing prices: Naive, Seasonal Naive, Moving Average, Holt-Winters, Ridge Regression, Prophet, Ensemble, and Auto-SARIMAX. Testing on a holdout dataset, I found that the 12-month Moving Average performed best with 5.83% MAPE, followed closely by Seasonal Naive (6.81%) and Holt-Winters (6.85%).

The key takeaways from my analysis connect directly to what we learned in class. First, the baseline methods from Week 7 like Seasonal Naive and Moving Average are not just for comparison, they can actually be among the best performers. Second, the Holt-Winters method from Week 8 is highly effective for data with trend and seasonality like housing prices. Third, regression approaches from Week 9 work better when you account for the time lag between economic changes and their effect on prices. Finally, seasonal decomposition is a valuable tool for understanding time series data before building forecasting models.

For future work, it would be interesting to look at more localized data at the town or county level, include additional variables like housing inventory, and explore how the models perform during different market conditions. Overall, this project gave me a good appreciation for both the potential and the limitations of forecasting, and demonstrated that sometimes the simplest approaches we learn in class turn out to be among the most effective in practice.

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

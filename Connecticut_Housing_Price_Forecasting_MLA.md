[Your Name]

Professor [Name]

MBA Data Analytics

8 December 2025

Forecasting Connecticut Residential Housing Prices: A Comparative Analysis of Time Series and Machine Learning Models

Introduction

For this project, I wanted to explore how well different forecasting methods could predict housing prices. Housing is something that affects almost everyone, whether you are buying a home, investing in real estate, or working at a bank that gives out mortgages. Being able to predict where prices are going seems like it would be really valuable for making better decisions. I chose to focus on Connecticut for a few reasons. First, there was good data available through the state's open data portal. Second, Connecticut has an interesting housing market because it is close to New York City, which creates a lot of demand from commuters. The state also went through some big changes during COVID-19 when a lot of people moved out of the city to the suburbs, which made prices go up significantly. The main question I wanted to answer was: which forecasting method works best for predicting housing prices? I had learned about several different approaches in class, from simple methods to more advanced machine learning techniques, and I wanted to see how they compared using real data.

Predicting housing prices is harder than it might seem. There are so many factors that can affect prices, including interest rates, the economy, population changes, and things nobody can predict like a pandemic. The COVID-19 period was especially challenging because prices in Connecticut went up by about 40% in just two years, which was way more than normal. I thought this would be a good test for the different forecasting models because if a model can handle a crazy period like COVID-19, it is probably pretty robust, and if it cannot, that tells us something important about its limitations. My goals for this project were to get the housing and economic data together in a format I could use for analysis, try out several different forecasting methods that I learned about, compare how well each method predicted prices during the 2019-2024 period, and figure out which approach might be most useful for someone who actually needs to forecast housing prices.

Literature Review

Before building my models, I did some research on what other people have found about housing price forecasting. Housing prices basically come down to supply and demand. On the demand side, things like income levels, population growth, and mortgage rates determine how many people want to buy homes and how much they can afford. On the supply side, construction costs and how much land is available affect how many homes get built. Rosen wrote an influential paper about how you can break down a home's value into its different characteristics, which is called hedonic pricing (34). One thing that stood out from my reading is how important mortgage rates are. Himmelberg, Mayer, and Sinai found that when mortgage rates go up by 1%, people can afford about 10% less house (71). That is a big deal and helps explain why rates matter so much for prices.

There are basically two main approaches to forecasting that I found in the literature. The first is time series methods, which look at patterns in the historical price data itself. These include ARIMA models that Crawford and Fratantoni studied (225), and exponential smoothing methods like Holt-Winters that Hyndman and Athanasopoulos discuss in their textbook (236). The second approach is using regression and machine learning, where you try to predict prices based on other variables like economic indicators. Tibshirani developed Ridge and Lasso regression which help when your predictor variables are correlated with each other, which is common with economic data (267). There is also a newer method called Prophet that Facebook developed, which is supposed to be good at handling seasonal patterns and unusual events, so I thought it might do well with housing data (Taylor and Letham 38). One interesting finding from Bates and Granger is that combining forecasts from different models often works better than using just one model, which is called ensemble forecasting, and I wanted to try that approach too (453).

Data and Methodology

Data Sources

I got my housing data from the Connecticut Open Data Portal, which is a website where the state publishes public datasets. The real estate sales database had over 1.1 million property transactions going back to 2001, and each record had information like the sale price, date, what type of property it was, and where it was located. For economic data, I used the FRED database, which is run by the Federal Reserve Bank of St. Louis. Table 1 shows the economic indicators I used in my analysis along with their sources.

Table 1. Economic Indicator Variables Used in Analysis
┌─────────────────────┬─────────────────┬────────────────────────────────────┐
│ Variable            │ FRED Series ID  │ Description                        │
├─────────────────────┼─────────────────┼────────────────────────────────────┤
│ Mortgage Rate       │ MORTGAGE30US    │ 30-year fixed mortgage rate (%)    │
│ Unemployment Rate   │ CTURN           │ Connecticut unemployment rate (%)  │
│ Consumer Price Index│ CPIAUCSL        │ CPI for All Urban Consumers        │
│ Population          │ CTPOP           │ Connecticut population estimate    │
└─────────────────────┴─────────────────┴────────────────────────────────────┘
Source: Federal Reserve Economic Data (FRED), Federal Reserve Bank of St. Louis.

Having both the housing transaction data and the economic indicators allowed me to test both pure time series methods and regression approaches that use economic variables as predictors. The combination of these datasets gave me a comprehensive view of both the housing market itself and the broader economic conditions that might influence prices.

Data Preparation

The raw data needed quite a bit of cleaning before I could use it. Some of the dates were formatted wrong, and there were sales with really low prices under $10,000 that were probably not real market transactions, maybe transfers between family members or something like that, so I filtered those out. I also focused just on residential properties since commercial real estate works differently. After filtering, I calculated the median sale price for each month. I used the median instead of the average because real estate has a lot of outliers, like if one mansion sells for $10 million, that would throw off the average but would not affect the median as much. Once I had monthly median prices, I merged that with the economic data by date. Table 2 provides a summary of the final dataset I used for the analysis.

Table 2. Summary Statistics for Connecticut Housing Data (2005-2024)
┌────────────────────────────┬────────────────────────┐
│ Statistic                  │ Value                  │
├────────────────────────────┼────────────────────────┤
│ Total Monthly Observations │ 231 months             │
│ Date Range                 │ January 2005 - Sept 2024│
│ Mean Median Price          │ $298,456               │
│ Minimum Median Price       │ $215,000               │
│ Maximum Median Price       │ $435,000               │
│ Standard Deviation         │ $52,847                │
│ Training Period            │ 163 months (2005-2018) │
│ Testing Period             │ 69 months (2019-2024)  │
└────────────────────────────┴────────────────────────┘
Source: Connecticut Open Data Portal, author's calculations.

Figure 1 below illustrates the trend in Connecticut median housing prices over the study period. The chart shows relatively stable prices from 2005-2012, followed by gradual recovery, and then a sharp increase during the COVID-19 pandemic period starting in 2020.

Figure 1. Connecticut Median Housing Prices (2005-2024)

Price ($)
450,000 |                                                    ****
        |                                                 ***
400,000 |                                              ***
        |                                           **
350,000 |                                        ***
        |                                     ***
300,000 |  ***                            ****
        |      ****                   ****
250,000 |          *******************
        |
200,000 |________________________________________________
         2005    2008    2011    2014    2017    2020    2024
                              Year

Source: Connecticut Open Data Portal, author's calculations.
Note: Prices represent monthly median sale prices for residential properties.

Train-Test Split

For forecasting, you need to split your data into a training set to build the model and a test set to see how well it works on new data. I used data from 2005-2018 for training, which gave me 163 months, and then I tested on 2019-2024, which was 69 months. I specifically wanted the test period to include COVID-19 because I thought that would be a good stress test for the models. Any model can do okay when things are normal, but the real test is how it handles unusual situations. By using data that the models had never seen before for testing, I could get a realistic sense of how well they would actually perform in practice rather than just how well they fit historical patterns.

The Forecasting Models

I tested six different forecasting approaches, and Table 3 provides a summary of each model and its key characteristics.

Table 3. Summary of Forecasting Models Tested
┌──────────────────┬─────────────────┬────────────────────────────────────────┐
│ Model            │ Type            │ Key Characteristics                    │
├──────────────────┼─────────────────┼────────────────────────────────────────┤
│ Naive            │ Benchmark       │ Uses last observed value as forecast   │
│ Holt-Winters     │ Time Series     │ Captures level, trend, and seasonality │
│ Ridge Regression │ Machine Learning│ Uses economic indicators as predictors │
│ Facebook Prophet │ Time Series     │ Automatic changepoint detection        │
│ Ensemble         │ Combined        │ Averages Holt-Winters and Ridge        │
│ Auto-SARIMAX     │ Time Series     │ Automated parameter selection          │
└──────────────────┴─────────────────┴────────────────────────────────────────┘
Source: Author's compilation.

The Naive model is the simplest possible forecast where it just assumes tomorrow will be the same as today. Whatever the price was last month, that is the prediction for all future months. I included this as a baseline to make sure the other models were actually adding value, because if a complicated model cannot beat the naive approach, then what is the point of all that complexity?

Holt-Winters is a method that tries to capture three things: the overall level of prices, whether they are trending up or down, and seasonal patterns like if prices are always higher in spring. It uses something called exponential smoothing, which basically means recent data points get more weight than older ones. I set it up to look for a 12-month seasonal pattern since housing markets tend to follow annual cycles where spring and summer are typically busier seasons with higher prices. This method has been around for a while and is known for being reliable across many different types of forecasting problems.

Ridge Regression is a type of regression that works well when your predictor variables are correlated with each other. Regular regression can get unstable in that situation, but Ridge adds a penalty that keeps the coefficients from getting too extreme. I used the economic variables including mortgage rate, unemployment, inflation, and population as predictors. The idea is that these economic factors should have some relationship with housing prices, so if we can capture that relationship, we might be able to make better predictions than just looking at price patterns alone.

Facebook Prophet is a forecasting tool that Facebook's data science team developed and released as open source software. It is designed to handle business data that has strong seasonal patterns and can adapt when there are sudden changes in trends. I thought it might handle the COVID period well since it is supposed to detect these kinds of shifts automatically. The tool is pretty user-friendly and does a lot of the complicated stuff under the hood, which made it relatively easy to implement even though I am not an expert in all the technical details.

The Ensemble Model is based on what I read about combining forecasts. I created a simple ensemble that averaged the predictions from Holt-Winters and Ridge Regression. The idea is that if one model is off in one direction and another is off in the other direction, the average might be closer to the truth. Different models capture different patterns in the data, so combining them can sometimes give you the best of both worlds.

Auto-SARIMAX stands for Seasonal Autoregressive Integrated Moving Average with Exogenous variables, which is a mouthful. Basically, it is a flexible time series model that can include both seasonal patterns and outside variables like economic indicators. The "Auto" part means I used a software library called pmdarima that automatically figures out the best settings for the model instead of me having to guess. This was helpful because manually picking SARIMAX parameters is tricky and getting them wrong can lead to really bad results, which I learned the hard way in some of my earlier attempts.

Performance Metrics

I used four metrics to compare the models. RMSE, or Root Mean Square Error, tells you the typical size of errors in dollars and it penalizes big errors more than small ones because of the squaring. MAE, or Mean Absolute Error, is just the average error in dollars, treating all errors equally regardless of size. MAPE, or Mean Absolute Percentage Error, expresses errors as a percentage, which makes it easier to interpret since a MAPE of 20% means predictions are off by 20% on average. R-squared measures how much of the variation in prices the model explains, and usually it is between 0 and 1, but it can actually be negative if the model does worse than just predicting the average. I focused mainly on MAPE for comparing the models because I think percentage error is the most intuitive way to understand how well a forecast is doing.

Results

Main Findings

Table 4 shows how all six models performed on the test data, ranked by MAPE since I think percentage error is the most intuitive measure for understanding forecast accuracy.

Table 4. Model Performance Comparison (2019-2024 Test Period)
┌──────────────────┬────────────┬────────────┬──────────┬─────────┐
│ Model            │ RMSE ($)   │ MAE ($)    │ MAPE (%) │ R²      │
├──────────────────┼────────────┼────────────┼──────────┼─────────┤
│ Holt-Winters     │ 70,551     │ 60,811     │ 19.05    │ -1.10   │
│ Auto-SARIMAX     │ 88,791     │ 75,952     │ 23.62    │ -2.32   │
│ Naive            │ 91,752     │ 78,967     │ 24.59    │ -2.55   │
│ Prophet          │ 102,966    │ 88,850     │ 28.07    │ -3.46   │
│ Ensemble         │ 116,471    │ 95,117     │ 29.26    │ -4.71   │
│ Ridge Regression │ 163,746    │ 130,906    │ 40.12    │ -10.29  │
└──────────────────┴────────────┴────────────┴──────────┴─────────┘
Source: Author's calculations.
Note: Models ranked by MAPE (lower is better). RMSE = Root Mean Square Error; MAE = Mean Absolute Error; MAPE = Mean Absolute Percentage Error.

The Holt-Winters model performed best with a MAPE of about 19%. That means on average, its predictions were off by about 19% from the actual prices, so if a home sold for $350,000, the prediction might have been somewhere in the range of $285,000 to $415,000. Auto-SARIMAX came in second at around 24% MAPE, which was still pretty decent. What surprised me was that the Naive model, which does not really do anything sophisticated at all, actually beat Prophet and the Ensemble approach. Ridge Regression did the worst by far at 40% error, which I did not expect since it was using all that economic data. This result made me think about why the simpler methods might have done better, which I discuss more in the next section.

Figure 2 provides a visual comparison of the MAPE scores across all six models, making it easy to see the relative performance differences.

Figure 2. Model Performance Comparison by MAPE (%)

MAPE (%)
    |
 45 |                                                          ████
    |                                                          ████
 40 |                                                          ████
    |                                                          ████
 35 |                                                          ████
    |                                        ████    ████      ████
 30 |                                        ████    ████      ████
    |                              ████      ████    ████      ████
 25 |                    ████      ████      ████    ████      ████
    |          ████      ████      ████      ████    ████      ████
 20 |  ████    ████      ████      ████      ████    ████      ████
    |  ████    ████      ████      ████      ████    ████      ████
 15 |  ████    ████      ████      ████      ████    ████      ████
    |  ████    ████      ████      ████      ████    ████      ████
 10 |  ████    ████      ████      ████      ████    ████      ████
    |  ████    ████      ████      ████      ████    ████      ████
  5 |  ████    ████      ████      ████      ████    ████      ████
    |  ████    ████      ████      ████      ████    ████      ████
  0 └──────────────────────────────────────────────────────────────
       Holt-   Auto-    Naive   Prophet  Ensemble  Ridge
       Winters SARIMAX                             Regression

Source: Author's calculations.
Note: Lower MAPE indicates better forecasting performance.

Why the R-squared Values Are Negative

When I first saw the negative R-squared values, I thought I did something wrong. But after looking into it more, I learned this can happen when your test period is very different from your training period. During COVID-19, housing prices went up way faster than anything in the historical data. The models were trained on data where prices grew slowly and steadily, and then suddenly in 2020-2021, prices shot up 40% in two years. None of the models could predict that because nothing like it had happened before in the data they learned from. A negative R-squared basically means the model did worse than if you just predicted the average price for everything. It is not great, but it makes sense given how unusual the test period was, and it actually shows the value of testing on out-of-sample data rather than just looking at how well the model fits historical patterns.

The Lagged Indicator Finding

One thing I discovered while experimenting was that the timing of economic variables matters a lot. When I used current mortgage rates to predict current prices, it did not work well, but when I used mortgage rates from 12 months ago to predict current prices, the results improved dramatically. Table 5 shows this comparison for the Ridge Regression model.

Table 5. Effect of Using Lagged Economic Variables on Ridge Regression
┌────────────────────────────────┬──────────┬─────────┐
│ Configuration                  │ MAPE (%) │ R²      │
├────────────────────────────────┼──────────┼─────────┤
│ Current economic indicators    │ 40.12    │ -10.29  │
│ 12-month lagged indicators     │ 11.53    │ +0.23   │
└────────────────────────────────┴──────────┴─────────┘
Source: Author's calculations.
Note: Both models use Ridge Regression with economic indicators as predictors.

This makes sense when you think about it. When mortgage rates change, it takes time for that to show up in prices. Buyers need to adjust their budgets, sellers need to adjust their expectations, and deals take months to close. So what is happening in the economy today tells you more about where prices will be in a year than where they are right now. This was probably my most interesting discovery from the whole project, and it suggests that anyone trying to use economic indicators for housing forecasts should think carefully about the timing of those relationships rather than just assuming everything happens at the same time.

Figure 3 illustrates the dramatic improvement in model performance when using lagged economic indicators instead of current values.

Figure 3. Impact of Lagged Variables on Ridge Regression Performance

MAPE (%)
    |
 45 |  ████████████████
    |  ████████████████
 40 |  ████████████████
    |  ████████████████
 35 |  ████████████████
    |  ████████████████
 30 |  ████████████████
    |  ████████████████
 25 |  ████████████████
    |  ████████████████
 20 |  ████████████████
    |  ████████████████
 15 |  ████████████████      ██████████████
    |  ████████████████      ██████████████
 10 |  ████████████████      ██████████████
    |  ████████████████      ██████████████
  5 |  ████████████████      ██████████████
    |  ████████████████      ██████████████
  0 └──────────────────────────────────────
        Current               12-Month
       Indicators              Lagged

Source: Author's calculations.
Note: Using 12-month lagged indicators reduces MAPE from 40.12% to 11.53%.

Discussion

What I Learned

This project taught me several things about forecasting that I think are valuable takeaways. First, simple can be better. I expected the more sophisticated methods like Prophet and the Ensemble to do better, but the relatively simple Holt-Winters model won. I think this is because the complicated models tried too hard to fit the historical patterns, and when those patterns broke during COVID, they were way off. Holt-Winters just follows the trend and seasonal pattern without overthinking it, which turned out to be more robust when conditions changed unexpectedly. This reminded me of something called overfitting that we talked about in class, where a model can be too complex for its own good.

Second, automation helps. The Auto-SARIMAX model performed much better than when I tried to manually pick the SARIMAX parameters earlier in my analysis. Getting those parameters wrong led to errors over 80%, which was terrible. The automated approach found settings that worked much better by systematically searching through different combinations and picking the one that performed best. This suggests that using tools that can optimize settings is better than trying to guess, especially for someone like me who does not have years of experience with these models.

Third, timing matters for economic variables. The 12-month lag finding was probably my most interesting discovery. It suggests that if you want to use economic indicators for forecasting, you need to think about when they actually affect prices, not just whether they are related. This makes intuitive sense when you consider how the housing market works in practice, with long transaction times and gradual adjustments in buyer and seller behavior.

Table 6 summarizes the key findings from my analysis.

Table 6. Summary of Key Findings
┌─────────────────────────────────────────────────────────────────────────────┐
│ Finding                                                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. Holt-Winters performed best overall with 19.05% MAPE                     │
│ 2. Simpler models outperformed complex machine learning approaches          │
│ 3. All models struggled during the COVID-19 period (negative R² values)     │
│ 4. Economic indicators affect prices with ~12-month lag                     │
│ 5. Automated parameter selection dramatically improves SARIMAX performance  │
│ 6. Ridge Regression with lagged indicators achieved best single result      │
│    (11.53% MAPE)                                                            │
└─────────────────────────────────────────────────────────────────────────────┘
Source: Author's analysis.

Limitations

There are several limitations to my analysis that I should acknowledge. First, I used state-level data, but housing markets really vary a lot by location. Fairfield County near New York City is very different from rural eastern Connecticut, with different price levels, different buyer demographics, and probably different responses to economic changes. A more detailed analysis would look at smaller geographic areas to see if the patterns hold or if some models work better in certain markets.

Second, I only used a few economic variables. There are probably other factors that matter, like housing inventory levels, new construction activity, or migration patterns, that I did not include. Having more data might have improved the regression models in particular, since they rely on having good predictor variables. Third, I did not spend a lot of time tuning the models beyond the automated approaches. With more time, I could probably improve performance by trying different settings and parameters. Fourth, the COVID-19 period was really unusual, so the models might perform differently during more normal times.

Practical Implications

Based on my results, here is what I would suggest for different users. For investors trying to forecast prices, I would recommend using Holt-Winters or a similar exponential smoothing approach for the near term. It is relatively simple to implement and performed best in my testing. For longer-term forecasts where you expect economic conditions to change significantly, incorporating lagged economic indicators through regression could add value, especially given how much better the Ridge model did with 12-month lags.

For banks and mortgage lenders, understanding that economic changes take about a year to show up in prices is important. If rates are rising today, that will affect collateral values down the road, not immediately. This has implications for how they should think about risk in their loan portfolios and how they evaluate properties for new mortgages.

For anyone doing forecasting, my results show that you should always test your models on recent data that the model has not seen before. A model might look great on historical data but fail badly when conditions change. The negative R-squared values I got are a good example of this, where models that seemed reasonable based on training data performance really struggled with the unusual test period.

Conclusion

In this project, I compared six different methods for forecasting Connecticut housing prices: Naive, Holt-Winters, Ridge Regression, Prophet, Ensemble, and Auto-SARIMAX. Testing on data from 2019-2024, which included the COVID-19 period, I found that Holt-Winters performed best with about 19% average error. The key takeaways from my analysis are that simpler forecasting methods can outperform more complex ones, especially during unusual market conditions; automated parameter selection like Auto-SARIMAX works better than manual guessing for complex models; economic indicators affect housing prices with a delay of about 12 months, so using lagged variables improves predictions significantly; and all models struggled during COVID-19, showing that extreme events are very hard to predict.

For future work, it would be interesting to look at more localized data at the town or county level, include additional variables like housing inventory, and test some newer deep learning approaches that I did not have time to explore. Overall, this project gave me a good appreciation for both the potential and the limitations of forecasting. While we can use data and models to make better predictions than just guessing, there will always be uncertainty, especially when unprecedented events occur. The best approach is probably to use forecasts as one input into decisions while acknowledging their limitations and being prepared for scenarios where the predictions turn out to be wrong.

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

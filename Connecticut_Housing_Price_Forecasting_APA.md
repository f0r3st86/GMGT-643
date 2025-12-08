Running head: FORECASTING CONNECTICUT HOUSING PRICES                                    1










Forecasting Connecticut Residential Housing Prices:

A Comparative Analysis of Time Series and Machine Learning Models

[Your Name]

[University Name]

MBA Data Analytics

December 2025




















FORECASTING CONNECTICUT HOUSING PRICES                                                   2

Abstract

This paper presents my analysis of different forecasting methods to predict housing prices in Connecticut. I used real estate sales data from the Connecticut Open Data Portal, which had over a million property transactions, and combined it with economic data like mortgage rates and unemployment from the Federal Reserve database. I tested six different forecasting models to see which one could best predict housing prices: a simple Naive model, Holt-Winters, Ridge Regression, Facebook Prophet, an Ensemble approach, and an automated SARIMAX model. After running all the models, I found that Holt-Winters performed the best with about 19% error rate, followed by the Auto-SARIMAX at around 24%. My results suggest that sometimes simpler forecasting approaches work better than complicated machine learning methods, especially when the market behaves unexpectedly like it did during COVID-19.

Keywords: housing prices, forecasting, time series, machine learning, Connecticut real estate
















FORECASTING CONNECTICUT HOUSING PRICES                                                   3

Forecasting Connecticut Residential Housing Prices:

A Comparative Analysis of Time Series and Machine Learning Models

For this project, I wanted to explore how well different forecasting methods could predict housing prices. Housing is something that affects almost everyone, whether you are buying a home, investing in real estate, or working at a bank that gives out mortgages. Being able to predict where prices are going seems like it would be really valuable for making better decisions. I chose to focus on Connecticut for a few reasons. First, there was good data available through the state's open data portal. Second, Connecticut has an interesting housing market because it is close to New York City, which creates a lot of demand from commuters. The state also went through some big changes during COVID-19 when a lot of people moved out of the city to the suburbs, which made prices go up significantly. The main question I wanted to answer was: which forecasting method works best for predicting housing prices? I had learned about several different approaches in class, from simple methods to more advanced machine learning techniques, and I wanted to see how they compared using real data.

Predicting housing prices is harder than it might seem. There are so many factors that can affect prices, including interest rates, the economy, population changes, and things nobody can predict like a pandemic. The COVID-19 period was especially challenging because prices in Connecticut went up by about 40% in just two years, which was way more than normal. I thought this would be a good test for the different forecasting models because if a model can handle a crazy period like COVID-19, it is probably pretty robust, and if it cannot, that tells us something important about its limitations. My goals for this project were to get the housing and economic data together in a format I could use for analysis, try out several different forecasting methods that I learned about, compare how well each method predicted prices during the 2019-2024 period, and figure out which approach might be most useful for someone who actually needs to forecast housing prices.


FORECASTING CONNECTICUT HOUSING PRICES                                                   4

Literature Review

Before building my models, I did some research on what other people have found about housing price forecasting. Housing prices basically come down to supply and demand. On the demand side, things like income levels, population growth, and mortgage rates determine how many people want to buy homes and how much they can afford. On the supply side, construction costs and how much land is available affect how many homes get built. Rosen (1974) wrote an influential paper about how you can break down a home's value into its different characteristics, which is called hedonic pricing. One thing that stood out from my reading is how important mortgage rates are. Himmelberg et al. (2005) found that when mortgage rates go up by 1%, people can afford about 10% less house. That is a big deal and helps explain why rates matter so much for prices.

There are basically two main approaches to forecasting that I found in the literature. The first is time series methods, which look at patterns in the historical price data itself. These include ARIMA models that Crawford and Fratantoni (2003) studied, and exponential smoothing methods like Holt-Winters that Hyndman and Athanasopoulos (2021) discuss in their textbook. The second approach is using regression and machine learning, where you try to predict prices based on other variables like economic indicators. Tibshirani (1996) developed Ridge and Lasso regression which help when your predictor variables are correlated with each other, which is common with economic data. There is also a newer method called Prophet that Facebook developed (Taylor & Letham, 2018), which is supposed to be good at handling seasonal patterns and unusual events, so I thought it might do well with housing data. One interesting finding from Bates and Granger (1969) is that combining forecasts from different models often works better than using just one model, which is called ensemble forecasting, and I wanted to try that approach too.


FORECASTING CONNECTICUT HOUSING PRICES                                                   5

Method

Data Collection

I got my housing data from the Connecticut Open Data Portal, which is a website where the state publishes public datasets. The real estate sales database had over 1.1 million property transactions going back to 2001, and each record had information like the sale price, date, what type of property it was, and where it was located. For economic data, I used the FRED database, which is run by the Federal Reserve Bank of St. Louis. I pulled data on the 30-year mortgage rate, Connecticut's unemployment rate, the Consumer Price Index which measures inflation, and population estimates. Having both the housing transaction data and the economic indicators allowed me to test both pure time series methods and regression approaches that use economic variables as predictors.

Data Preparation

The raw data needed quite a bit of cleaning before I could use it. Some of the dates were formatted wrong, and there were sales with really low prices under $10,000 that were probably not real market transactions, maybe transfers between family members or something like that, so I filtered those out. I also focused just on residential properties since commercial real estate works differently. After filtering, I calculated the median sale price for each month. I used the median instead of the average because real estate has a lot of outliers, like if one mansion sells for $10 million, that would throw off the average but would not affect the median as much. Once I had monthly median prices, I merged that with the economic data by date. My final dataset had 231 months of data from 2005 to 2024, which gave me a pretty good amount of historical information to work with for building and testing the models.




FORECASTING CONNECTICUT HOUSING PRICES                                                   6

Train-Test Split

For forecasting, you need to split your data into a training set to build the model and a test set to see how well it works on new data. I used data from 2005-2018 for training, which gave me 163 months, and then I tested on 2019-2024, which was 69 months. I specifically wanted the test period to include COVID-19 because I thought that would be a good stress test for the models. Any model can do okay when things are normal, but the real test is how it handles unusual situations. By using data that the models had never seen before for testing, I could get a realistic sense of how well they would actually perform in practice rather than just how well they fit historical patterns.

The Forecasting Models

I tested six different forecasting approaches, and here is a brief explanation of each one. The Naive model is the simplest possible forecast where it just assumes tomorrow will be the same as today. Whatever the price was last month, that is the prediction for all future months. I included this as a baseline to make sure the other models were actually adding value, because if a complicated model cannot beat the naive approach, then what is the point of all that complexity?

Holt-Winters is a method that tries to capture three things: the overall level of prices, whether they are trending up or down, and seasonal patterns like if prices are always higher in spring. It uses something called exponential smoothing, which basically means recent data points get more weight than older ones. I set it up to look for a 12-month seasonal pattern since housing markets tend to follow annual cycles where spring and summer are typically busier seasons with higher prices. This method has been around for a while and is known for being reliable across many different types of forecasting problems.


FORECASTING CONNECTICUT HOUSING PRICES                                                   7

Ridge Regression is a type of regression that works well when your predictor variables are correlated with each other. Regular regression can get unstable in that situation, but Ridge adds a penalty that keeps the coefficients from getting too extreme. I used the economic variables including mortgage rate, unemployment, inflation, and population as predictors. The idea is that these economic factors should have some relationship with housing prices, so if we can capture that relationship, we might be able to make better predictions than just looking at price patterns alone.

Facebook Prophet is a forecasting tool that Facebook's data science team developed and released as open source software. It is designed to handle business data that has strong seasonal patterns and can adapt when there are sudden changes in trends. I thought it might handle the COVID period well since it is supposed to detect these kinds of shifts automatically. The tool is pretty user-friendly and does a lot of the complicated stuff under the hood, which made it relatively easy to implement even though I am not an expert in all the technical details.

The Ensemble Model is based on what I read about combining forecasts. I created a simple ensemble that averaged the predictions from Holt-Winters and Ridge Regression. The idea is that if one model is off in one direction and another is off in the other direction, the average might be closer to the truth. Different models capture different patterns in the data, so combining them can sometimes give you the best of both worlds.

Auto-SARIMAX stands for Seasonal Autoregressive Integrated Moving Average with Exogenous variables, which is a mouthful. Basically, it is a flexible time series model that can include both seasonal patterns and outside variables like economic indicators. The "Auto" part means I used a software library called pmdarima that automatically figures out the best settings for the model instead of me having to guess. This was helpful because manually picking SARIMAX parameters is tricky and getting them wrong can lead to really bad results, which I learned the hard way in some of my earlier attempts.


FORECASTING CONNECTICUT HOUSING PRICES                                                   8

How I Measured Performance

I used four metrics to compare the models. RMSE, or Root Mean Square Error, tells you the typical size of errors in dollars and it penalizes big errors more than small ones because of the squaring. MAE, or Mean Absolute Error, is just the average error in dollars, treating all errors equally regardless of size. MAPE, or Mean Absolute Percentage Error, expresses errors as a percentage, which makes it easier to interpret since a MAPE of 20% means predictions are off by 20% on average. R-squared measures how much of the variation in prices the model explains, and usually it is between 0 and 1, but it can actually be negative if the model does worse than just predicting the average. I focused mainly on MAPE for comparing the models because I think percentage error is the most intuitive way to understand how well a forecast is doing.

Results

Main Findings

Table 1 shows how all six models performed on the test data, ranked by MAPE since I think percentage error is the most intuitive measure for understanding forecast accuracy.

Table 1

Model Performance Comparison (2019-2024 Test Period)

Model                   RMSE ($)        MAE ($)         MAPE (%)        R²
─────────────────────────────────────────────────────────────────────────────
Holt-Winters            70,551          60,811          19.05           -1.10
Auto-SARIMAX            88,791          75,952          23.62           -2.32
Naive                   91,752          78,967          24.59           -2.55
Prophet                 102,966         88,850          28.07           -3.46
Ensemble                116,471         95,117          29.26           -4.71
Ridge Regression        163,746         130,906         40.12           -10.29
─────────────────────────────────────────────────────────────────────────────

Note. Models ranked by MAPE (lower is better). RMSE = Root Mean Square Error; MAE = Mean Absolute Error.


FORECASTING CONNECTICUT HOUSING PRICES                                                   9

The Holt-Winters model performed best with a MAPE of about 19%. That means on average, its predictions were off by about 19% from the actual prices, so if a home sold for $350,000, the prediction might have been somewhere in the range of $285,000 to $415,000. Auto-SARIMAX came in second at around 24% MAPE, which was still pretty decent. What surprised me was that the Naive model, which does not really do anything sophisticated at all, actually beat Prophet and the Ensemble approach. Ridge Regression did the worst by far at 40% error, which I did not expect since it was using all that economic data. This result made me think about why the simpler methods might have done better, which I discuss more in the next section.

Why the R-squared Values Are Negative

When I first saw the negative R-squared values, I thought I did something wrong. But after looking into it more, I learned this can happen when your test period is very different from your training period. During COVID-19, housing prices went up way faster than anything in the historical data. The models were trained on data where prices grew slowly and steadily, and then suddenly in 2020-2021, prices shot up 40% in two years. None of the models could predict that because nothing like it had happened before in the data they learned from. A negative R-squared basically means the model did worse than if you just predicted the average price for everything. It is not great, but it makes sense given how unusual the test period was, and it actually shows the value of testing on out-of-sample data rather than just looking at how well the model fits historical patterns.

The Lagged Indicator Finding

One thing I discovered while experimenting was that the timing of economic variables matters a lot. When I used current mortgage rates to predict current prices, it did not work well, but when I used mortgage rates from 12 months ago to predict current prices, the results improved dramatically. Table 2 shows this comparison for the Ridge Regression model.


FORECASTING CONNECTICUT HOUSING PRICES                                                   10

Table 2

Effect of Using Lagged Economic Variables

Configuration                           MAPE (%)                R²
─────────────────────────────────────────────────────────────────────────────
Current economic indicators             40.12                   -10.29
12-month lagged indicators              11.53                   +0.23
─────────────────────────────────────────────────────────────────────────────

Note. Results shown for Ridge Regression model.

This makes sense when you think about it. When mortgage rates change, it takes time for that to show up in prices. Buyers need to adjust their budgets, sellers need to adjust their expectations, and deals take months to close. So what is happening in the economy today tells you more about where prices will be in a year than where they are right now. This was probably my most interesting discovery from the whole project, and it suggests that anyone trying to use economic indicators for housing forecasts should think carefully about the timing of those relationships rather than just assuming everything happens at the same time.

Discussion

What I Learned

This project taught me several things about forecasting that I think are valuable takeaways. First, simple can be better. I expected the more sophisticated methods like Prophet and the Ensemble to do better, but the relatively simple Holt-Winters model won. I think this is because the complicated models tried too hard to fit the historical patterns, and when those patterns broke during COVID, they were way off. Holt-Winters just follows the trend and seasonal pattern without overthinking it, which turned out to be more robust when conditions changed unexpectedly. This reminded me of something called overfitting that we talked about in class, where a model can be too complex for its own good.


FORECASTING CONNECTICUT HOUSING PRICES                                                   11

Second, automation helps. The Auto-SARIMAX model performed much better than when I tried to manually pick the SARIMAX parameters earlier in my analysis. Getting those parameters wrong led to errors over 80%, which was terrible. The automated approach found settings that worked much better by systematically searching through different combinations and picking the one that performed best. This suggests that using tools that can optimize settings is better than trying to guess, especially for someone like me who does not have years of experience with these models.

Third, timing matters for economic variables. The 12-month lag finding was probably my most interesting discovery. It suggests that if you want to use economic indicators for forecasting, you need to think about when they actually affect prices, not just whether they are related. This makes intuitive sense when you consider how the housing market works in practice, with long transaction times and gradual adjustments in buyer and seller behavior.

Limitations

There are several limitations to my analysis that I should acknowledge. First, I used state-level data, but housing markets really vary a lot by location. Fairfield County near New York City is very different from rural eastern Connecticut, with different price levels, different buyer demographics, and probably different responses to economic changes. A more detailed analysis would look at smaller geographic areas to see if the patterns hold or if some models work better in certain markets.

Second, I only used a few economic variables. There are probably other factors that matter, like housing inventory levels, new construction activity, or migration patterns, that I did not include. Having more data might have improved the regression models in particular, since they rely on having good predictor variables. Third, I did not spend a lot of time tuning the models beyond the automated approaches. With more time, I could probably improve performance by trying different settings and parameters. Fourth, the COVID-19 period was really unusual, so the models might perform differently during more normal times.


FORECASTING CONNECTICUT HOUSING PRICES                                                   12

Practical Implications

Based on my results, here is what I would suggest for different users. For investors trying to forecast prices, I would recommend using Holt-Winters or a similar exponential smoothing approach for the near term. It is relatively simple to implement and performed best in my testing. For longer-term forecasts where you expect economic conditions to change significantly, incorporating lagged economic indicators through regression could add value, especially given how much better the Ridge model did with 12-month lags.

For banks and mortgage lenders, understanding that economic changes take about a year to show up in prices is important. If rates are rising today, that will affect collateral values down the road, not immediately. This has implications for how they should think about risk in their loan portfolios and how they evaluate properties for new mortgages.

For anyone doing forecasting, my results show that you should always test your models on recent data that the model has not seen before. A model might look great on historical data but fail badly when conditions change. The negative R-squared values I got are a good example of this, where models that seemed reasonable based on training data performance really struggled with the unusual test period.

Conclusion

In this project, I compared six different methods for forecasting Connecticut housing prices: Naive, Holt-Winters, Ridge Regression, Prophet, Ensemble, and Auto-SARIMAX. Testing on data from 2019-2024, which included the COVID-19 period, I found that Holt-Winters performed best with about 19% average error. The key takeaways from my analysis are that simpler forecasting methods can outperform more complex ones, especially during unusual market conditions; automated parameter selection like Auto-SARIMAX works better than manual guessing for complex models; economic indicators affect housing prices with a delay of about 12 months, so using lagged variables improves predictions significantly; and all models struggled during COVID-19, showing that extreme events are very hard to predict.


FORECASTING CONNECTICUT HOUSING PRICES                                                   13

For future work, it would be interesting to look at more localized data at the town or county level, include additional variables like housing inventory, and test some newer deep learning approaches that I did not have time to explore. Overall, this project gave me a good appreciation for both the potential and the limitations of forecasting. While we can use data and models to make better predictions than just guessing, there will always be uncertainty, especially when unprecedented events occur. The best approach is probably to use forecasts as one input into decisions while acknowledging their limitations and being prepared for scenarios where the predictions turn out to be wrong.



















FORECASTING CONNECTICUT HOUSING PRICES                                                   14

References

Bates, J. M., & Granger, C. W. J. (1969). The combination of forecasts. Operations Research

        Quarterly, 20(4), 451-468. https://doi.org/10.1057/jors.1969.103

Crawford, G. W., & Fratantoni, M. C. (2003). Assessing the forecasting performance of regime-

        switching, ARIMA and GARCH models of house prices. Real Estate Economics, 31(2),

        223-243. https://doi.org/10.1111/1540-6229.00064

Himmelberg, C., Mayer, C., & Sinai, T. (2005). Assessing high house prices: Bubbles,

        fundamentals and misperceptions. Journal of Economic Perspectives, 19(4), 67-92.

        https://doi.org/10.1257/089533005775196769

Hyndman, R. J., & Athanasopoulos, G. (2021). Forecasting: Principles and practice (3rd ed.).

        OTexts. https://otexts.com/fpp3/

Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2018). Statistical and machine learning

        forecasting methods: Concerns and ways forward. PLoS ONE, 13(3), Article e0194889.

        https://doi.org/10.1371/journal.pone.0194889

Rosen, S. (1974). Hedonic prices and implicit markets: Product differentiation in pure

        competition. Journal of Political Economy, 82(1), 34-55. https://doi.org/10.1086/260169

Taylor, S. J., & Letham, B. (2018). Forecasting at scale. The American Statistician, 72(1), 37-45.

        https://doi.org/10.1080/00031305.2017.1380080

Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. Journal of the Royal

        Statistical Society: Series B (Methodological), 58(1), 267-288.

        https://doi.org/10.1111/j.2517-6161.1996.tb02080.x

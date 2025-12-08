FORECASTING CONNECTICUT RESIDENTIAL HOUSING PRICES:
A COMPARATIVE ANALYSIS OF TIME SERIES AND MACHINE LEARNING MODELS






A Paper Submitted in Partial Fulfillment
of the Requirements for
MBA Data Analytics









December 2025




ABSTRACT

This study examines the effectiveness of various forecasting methodologies for predicting residential housing prices in Connecticut. Utilizing transaction data from the Connecticut Open Data Portal encompassing over 1.1 million property sales from 2001 to 2024, combined with macroeconomic indicators from the Federal Reserve Economic Data database, six distinct forecasting models were developed and evaluated. The models tested include Naive forecasting, Holt-Winters Exponential Smoothing, Ridge Regression, Facebook Prophet, Ensemble averaging, and Seasonal Autoregressive Integrated Moving Average with Exogenous Variables (SARIMAX) with automated parameter selection. Results indicate that the Holt-Winters model achieved superior predictive performance with a Mean Absolute Percentage Error of 19.05 percent, followed by the Auto-SARIMAX model at 23.62 percent. These findings suggest that traditional time series methods effectively capture housing market dynamics and outperform more complex machine learning approaches during periods of significant market volatility. The implications of these findings for real estate investors, mortgage lenders, and policy makers are discussed.

Keywords: housing price forecasting, time series analysis, machine learning, Connecticut real estate, predictive modeling




TABLE OF CONTENTS

Abstract.....................................................................................................................i
Table of Contents....................................................................................................ii
List of Tables..........................................................................................................iii
1. Introduction.........................................................................................................1
2. Literature Review................................................................................................3
3. Data and Methodology........................................................................................5
4. Results...............................................................................................................10
5. Discussion.........................................................................................................13
6. Conclusion........................................................................................................16
References.............................................................................................................18




LIST OF TABLES

Table 1: Economic Indicator Variables and Sources..............................................6
Table 2: Descriptive Statistics for Housing Prices.................................................7
Table 3: Model Performance Comparison............................................................10
Table 4: Impact of Lagged Economic Indicators.................................................12




1. INTRODUCTION

1.1 Background

The residential real estate market represents one of the most significant asset classes in the United States economy. According to Federal Reserve data, housing wealth constitutes approximately seventy percent of total household wealth for the median American family (Board of Governors of the Federal Reserve System, 2023). The ability to accurately forecast housing prices carries substantial implications for multiple stakeholder groups, including real estate investors evaluating portfolio allocation decisions, mortgage lenders conducting risk assessments, policy makers monitoring economic conditions, and individual homeowners making purchase or sale decisions.

Connecticut presents a particularly compelling case for housing market analysis due to its distinctive market characteristics. The state's proximity to New York City generates substantial commuter demand, particularly in Fairfield County. Additionally, Connecticut possesses an aging housing stock, significant wealth concentration in certain municipalities, and has historically exhibited notable price volatility during economic cycles. These factors combine to create a complex forecasting environment that tests the capabilities of various predictive modeling approaches.

1.2 Problem Statement

Despite advances in statistical modeling and machine learning, accurate housing price prediction remains challenging. The COVID-19 pandemic, which began affecting markets in early 2020, created unprecedented disruptions to housing market dynamics. Median home prices in Connecticut increased by approximately forty percent between 2020 and 2022, driven by historically low mortgage rates, urban-to-suburban migration patterns, and supply chain constraints limiting new construction. This period of extraordinary volatility provides a rigorous test environment for evaluating forecasting model robustness.

1.3 Research Objectives

This study pursues four primary objectives. First, it aims to develop a robust data pipeline integrating Connecticut real estate transactions with relevant macroeconomic indicators. Second, it seeks to implement and compare multiple forecasting methodologies spanning traditional statistical approaches to contemporary machine learning techniques. Third, it evaluates model performance using industry-standard metrics during a challenging out-of-sample period encompassing the COVID-19 pandemic. Fourth, it provides actionable recommendations for practitioners selecting among available forecasting approaches.

1.4 Significance of the Study

This research contributes to the existing literature in several ways. It provides an empirical comparison of forecasting methods using recent data that includes a major market disruption, offering insights into model performance under stress conditions. The study also demonstrates the practical application of automated model selection procedures, which have received limited attention in the housing forecasting literature. Finally, the findings offer guidance to practitioners who must select among competing methodologies for real-world applications.

1.5 Organization of the Paper

The remainder of this paper is organized as follows. Section 2 reviews the relevant literature on housing price forecasting and economic determinants of property values. Section 3 describes the data sources and methodological approach. Section 4 presents the empirical results. Section 5 discusses the findings and their implications. Section 6 concludes with recommendations and directions for future research.


2. LITERATURE REVIEW

2.1 Theoretical Framework

Housing prices are determined by the interaction of supply and demand forces operating within local, regional, and national contexts. On the demand side, factors including household income, population growth, mortgage interest rates, and consumer confidence influence the willingness and ability of buyers to purchase properties. On the supply side, construction costs, land availability, zoning regulations, and the existing housing stock determine the quantity of available units. The hedonic pricing model, introduced by Rosen (1974), provides a theoretical framework for understanding how individual property characteristics contribute to overall value.

2.2 Time Series Forecasting Methods

The application of time series methods to housing price prediction has a substantial history in the academic literature. Box-Jenkins Autoregressive Integrated Moving Average (ARIMA) models have been widely employed to capture the autocorrelation structures inherent in housing price series (Crawford & Fratantoni, 2003). These methods model future values as a function of past values and past forecast errors, with differencing applied to achieve stationarity.

Exponential smoothing methods, including the Holt-Winters approach, have demonstrated strong performance for time series exhibiting both trend and seasonal components (Hyndman & Athanasopoulos, 2021). The Holt-Winters method decomposes a series into level, trend, and seasonal components, with exponential weighting applied to historical observations. This approach offers computational simplicity while providing robust forecasts across diverse applications.

Seasonal ARIMA models extend the basic ARIMA framework to accommodate periodic patterns, making them particularly suitable for housing markets where transaction volumes and prices typically exhibit seasonal fluctuations. The addition of exogenous variables (SARIMAX) allows these models to incorporate external predictors such as economic indicators.

2.3 Machine Learning Approaches

Recent years have witnessed growing interest in applying machine learning techniques to housing price prediction. Regularized regression methods, including Ridge and Lasso regression, address the multicollinearity issues common in housing datasets where economic predictors are often highly correlated (Tibshirani, 1996). These methods impose penalties on coefficient magnitudes, reducing overfitting and improving out-of-sample prediction.

Facebook's Prophet algorithm represents a more recent contribution to the forecasting toolkit (Taylor & Letham, 2018). Designed for business time series with strong seasonal patterns, Prophet employs an additive decomposition model with automatic changepoint detection. The algorithm handles missing data and outliers gracefully, making it attractive for practical applications.

2.4 Ensemble Methods

The combination of forecasts from multiple models, known as ensemble forecasting, has been shown to improve accuracy by reducing model-specific errors (Bates & Granger, 1969). This approach recognizes that different models may capture different aspects of the underlying data generating process, and their combination can yield more robust predictions than any individual model.

2.5 Economic Determinants of Housing Prices

The empirical literature has identified several key macroeconomic factors affecting residential property values. Mortgage interest rates represent perhaps the most significant determinant of housing affordability. Himmelberg, Mayer, and Sinai (2005) estimate that a one percentage point increase in mortgage rates reduces purchasing power by approximately ten percent, with corresponding effects on prices.

Unemployment rates reflect labor market conditions and household income stability, affecting both the ability to qualify for mortgages and the willingness to make major purchase commitments. The Consumer Price Index captures inflationary pressures that affect construction costs, land values, and nominal price levels. Population growth drives housing demand through household formation, while income growth influences the quality and quantity of housing demanded.


3. DATA AND METHODOLOGY

3.1 Data Sources

This study employs two primary data sources. The first is the Connecticut Real Estate Sales database maintained by the Connecticut Office of Policy and Management and accessible through the Connecticut Open Data Portal (data.ct.gov). This comprehensive database, identified as dataset 5mzw-sjtu, contains property transaction records dating from 2001 to the present. Each record includes the sale amount, transaction date, property type classification, municipal location, and assessment information.

The second data source is the Federal Reserve Economic Data (FRED) database maintained by the Federal Reserve Bank of St. Louis. From this source, macroeconomic variables were obtained including the thirty-year fixed mortgage rate, Connecticut unemployment rate, Consumer Price Index for All Urban Consumers, and Connecticut population estimates. Table 1 provides details on the specific series utilized.

Table 1: Economic Indicator Variables and Sources
---------------------------------------------------------------------------
Variable              FRED Series ID    Description
---------------------------------------------------------------------------
Mortgage Rate         MORTGAGE30US      30-year fixed mortgage rate
Unemployment Rate     CTURN             Connecticut unemployment rate
Consumer Price Index  CPIAUCSL          CPI for All Urban Consumers
Population            CTPOP             Connecticut population estimate
---------------------------------------------------------------------------

3.2 Data Preprocessing

The raw housing transaction data required substantial preprocessing before analysis. First, date fields were parsed and standardized, with records containing invalid date values removed from the dataset. Second, non-arm's length transactions were filtered by excluding sales with amounts below ten thousand dollars, which typically represent transfers between family members, foreclosures, or recording errors. Third, the analysis was restricted to residential properties by filtering on property type classifications including single-family homes, condominiums, and multi-family residential buildings.

Following these filtering steps, individual transactions were aggregated to monthly median sale prices. The use of median rather than mean values provides robustness against outliers, which are common in real estate transactions. This aggregation resulted in 231 monthly observations spanning from January 2005 to September 2024.

The housing price series was then merged with monthly economic indicator data by date. Where economic indicators were available only at quarterly or annual frequency, values were interpolated to monthly observations using forward-filling procedures. Table 2 presents descriptive statistics for the final analysis dataset.

Table 2: Descriptive Statistics for Housing Prices
---------------------------------------------------------------------------
Statistic                          Value
---------------------------------------------------------------------------
Number of Observations             231 months
Date Range                         January 2005 - September 2024
Mean Median Price                  $298,456
Standard Deviation                 $52,847
Minimum                            $215,000
Maximum                            $435,000
---------------------------------------------------------------------------

3.3 Train-Test Split

Following established practices in time series forecasting, the data was divided into training and testing sets using a temporal split. The training period encompasses January 2005 through December 2018, providing 163 monthly observations for model estimation. The testing period spans January 2019 through September 2024, comprising 69 months reserved for out-of-sample evaluation.

This split was selected to ensure adequate historical data for model training while providing a substantial out-of-sample period that includes both normal market conditions and the COVID-19 disruption. The inclusion of the pandemic period in the test set provides a rigorous assessment of model robustness under extraordinary circumstances.

3.4 Forecasting Models

Six forecasting models were implemented, representing different methodological approaches to time series prediction.

3.4.1 Naive Model

The naive model serves as a benchmark, using the last observed value as the forecast for all future periods. Formally, the forecast for period t+h is defined as the observed value at period t. This simple approach establishes a minimum performance threshold that more sophisticated models should exceed to demonstrate value.

3.4.2 Holt-Winters Exponential Smoothing

The Holt-Winters method extends simple exponential smoothing to accommodate both trend and seasonal components. The model decomposes the time series into three elements: a level component representing the baseline value, a trend component capturing systematic increase or decrease over time, and a seasonal component modeling periodic fluctuations. For this analysis, additive specifications were employed for both trend and seasonality, with a seasonal period of twelve months corresponding to annual patterns.

3.4.3 Ridge Regression

Ridge regression addresses multicollinearity among predictor variables by augmenting the ordinary least squares objective function with an L2 penalty on coefficient magnitudes. The regularization parameter, set to 100 for this analysis, controls the trade-off between fitting the training data and maintaining small coefficient values. Predictor variables included the contemporaneous values of mortgage rate, unemployment rate, Consumer Price Index, and population.

3.4.4 Facebook Prophet

The Prophet algorithm decomposes time series into trend, seasonality, and holiday components using an additive model. For this application, Prophet was configured with yearly seasonality enabled, multiplicative seasonality mode to accommodate the growing price series, and automatic changepoint detection to identify shifts in the underlying trend.

3.4.5 Ensemble Model

The ensemble model combines forecasts from the Holt-Winters and Ridge regression models using simple averaging. This approach aims to leverage the complementary strengths of the two methods: Holt-Winters captures trend and seasonal patterns in the price series itself, while Ridge regression incorporates information from economic predictors.

3.4.6 Auto-SARIMAX

The Seasonal Autoregressive Integrated Moving Average with Exogenous Variables model extends the ARIMA framework to include seasonal components and external predictors. Rather than manually specifying the model orders, the pmdarima library's automatic model selection procedure was employed. This algorithm searches over candidate specifications, evaluating each using the Akaike Information Criterion, to identify the optimal combination of autoregressive, differencing, and moving average terms for both non-seasonal and seasonal components.

3.5 Evaluation Metrics

Model performance was assessed using four complementary metrics. Root Mean Square Error (RMSE) calculates the square root of the average squared difference between predicted and actual values, providing a measure in dollar terms that penalizes large errors heavily. Mean Absolute Error (MAE) computes the average absolute difference between predictions and actuals, also expressed in dollars but with linear rather than quadratic penalty for errors.

Mean Absolute Percentage Error (MAPE) expresses forecast errors as a percentage of actual values, facilitating interpretation across different price levels and comparison with other studies. The Coefficient of Determination (R-squared) measures the proportion of variance in actual values explained by the model predictions, with values potentially negative when predictions perform worse than a simple historical mean.


4. RESULTS

4.1 Model Performance Comparison

Table 3 presents the comparative performance of all six models on the test set spanning January 2019 through September 2024. Models are ranked by Mean Absolute Percentage Error, with lower values indicating superior predictive accuracy.

Table 3: Model Performance Comparison
---------------------------------------------------------------------------
Rank  Model                 RMSE ($)    MAE ($)    MAPE (%)   R-squared
---------------------------------------------------------------------------
1     Holt-Winters          70,551      60,811     19.05      -1.10
2     Auto-SARIMAX          88,791      75,952     23.62      -2.32
3     Naive                 91,752      78,967     24.59      -2.55
4     Prophet               102,966     88,850     28.07      -3.46
5     Ensemble              116,471     95,117     29.26      -4.71
6     Ridge Regression      163,746     130,906    40.12      -10.29
---------------------------------------------------------------------------

The Holt-Winters exponential smoothing model achieved the best performance across all four evaluation metrics. With a MAPE of 19.05 percent, the model's forecasts deviated from actual prices by an average of approximately sixty-one thousand dollars over the test period. The Auto-SARIMAX model ranked second with a MAPE of 23.62 percent, representing a meaningful improvement over the Naive benchmark.

Notably, the Naive model outperformed several more sophisticated approaches, including Prophet, the Ensemble, and Ridge regression. This counterintuitive result reflects the extraordinary nature of the test period, during which housing prices deviated substantially from patterns observed in the training data.

4.2 Interpretation of Negative R-squared Values

All models exhibit negative R-squared values on the test set, which may initially appear concerning. However, this outcome is expected when test period dynamics differ substantially from training period patterns. A negative R-squared indicates that model predictions explain less variance than a simple historical mean, which can occur when the test period represents a structural break from historical norms.

The COVID-19 pandemic triggered an unprecedented housing market surge driven by multiple factors operating simultaneously. Mortgage interest rates declined to historic lows, with the thirty-year fixed rate falling below three percent in 2020 and 2021. Urban-to-suburban migration accelerated as remote work arrangements enabled households to relocate away from city centers. Supply chain disruptions limited new construction, constraining supply at a time of elevated demand. Finally, massive fiscal stimulus programs increased household savings available for down payments.

These extraordinary conditions produced price increases of approximately forty percent over a two-year period, a trajectory that no model trained on pre-2019 data could reasonably have anticipated.

4.3 Analysis of Individual Model Performance

The success of the Holt-Winters model can be attributed to several factors. First, the method effectively captures the underlying upward trend in Connecticut housing prices that has persisted across the full sample period. Second, it models the seasonal pattern in housing transactions, where spring and summer months typically exhibit higher prices and transaction volumes. Third, the exponential smoothing approach adapts gradually to changing conditions without overfitting to noise in the training data.

The Auto-SARIMAX model's performance merits particular attention. The automated parameter selection procedure identified ARIMA(0,1,1)(0,0,0)[12] as the optimal specification. This relatively simple model indicates that first-order differencing adequately captures the trend component, a single moving average term models short-term dependencies, and explicit seasonal differencing was not required. The MAPE of 23.62 percent represents a substantial improvement over manually-specified SARIMAX models tested in preliminary analysis, which exhibited MAPE values exceeding eighty percent due to suboptimal parameter choices.

4.4 Impact of Lagged Economic Indicators

Supplementary analysis examined whether incorporating lagged economic indicators improves regression model performance. The rationale for this investigation stems from the recognition that economic conditions affect housing prices with a delay. Changes in mortgage rates, for example, require time to filter through the market as prospective buyers adjust their search behavior and sellers respond to changing demand conditions.

Table 4 presents Ridge regression results using both contemporaneous and twelve-month lagged economic indicators.

Table 4: Impact of Lagged Economic Indicators on Ridge Regression
---------------------------------------------------------------------------
Configuration                     MAPE (%)        R-squared
---------------------------------------------------------------------------
Contemporaneous indicators        40.12           -10.29
12-month lagged indicators        11.53           +0.23
---------------------------------------------------------------------------

The incorporation of lagged indicators produces dramatic improvement. The MAPE decreases from 40.12 percent to 11.53 percent, and the R-squared shifts from negative to positive, indicating that the model now explains meaningful variance in housing prices. This finding has important implications for model specification, suggesting that the relationship between economic conditions and housing prices operates with significant temporal delay.


5. DISCUSSION

5.1 Interpretation of Findings

The results of this study yield several important insights for housing price forecasting. First, and perhaps most notably, simpler models demonstrated competitive or superior performance relative to more complex alternatives during a period of significant market disruption. The Holt-Winters method, which relies solely on historical price patterns without incorporating external economic information, outperformed regression models and machine learning approaches that utilize multiple predictor variables.

This finding aligns with the broader forecasting literature, which has documented the "forecasting paradox" whereby simple methods often match or exceed the accuracy of sophisticated techniques (Makridakis et al., 2018). Complex models with numerous parameters are susceptible to overfitting historical patterns that may not persist into the future. During periods of structural change, such as the COVID-19 pandemic, this overfitting can result in particularly poor performance.

Second, the value of automated model selection procedures was clearly demonstrated. The Auto-SARIMAX approach, using systematic search over candidate specifications, achieved substantially better results than manually-specified alternatives. This finding suggests that practitioners should leverage available computational tools rather than relying on judgment-based parameter selection, which may introduce suboptimal choices.

Third, the analysis of lagged economic indicators revealed an important aspect of housing market dynamics. The twelve-month lag structure implies that current economic conditions provide information about housing prices one year hence, rather than contemporaneously. This finding has both theoretical and practical implications. Theoretically, it suggests that housing markets adjust slowly to changing economic conditions, perhaps due to search frictions, contracting delays, and expectation formation processes. Practically, it implies that models incorporating lagged predictors may offer superior forecasting performance.

5.2 Implications for Real Estate Investors

The findings of this study carry several implications for real estate investment practice. First, the approximately nineteen percent MAPE achieved by the best-performing model implies substantial forecast uncertainty that should be incorporated into investment decision-making. Point forecasts should be supplemented with confidence intervals, and sensitivity analysis should examine investment performance under alternative price scenarios.

Second, the trend and seasonal components identified by the Holt-Winters model provide actionable insights. The trend component offers guidance regarding long-term price direction, while seasonal patterns suggest optimal transaction timing. Historical data indicate that spring and summer listings typically achieve higher sale prices, a pattern that investors may exploit when timing dispositions.

Third, the extended forecast horizon enabled by lagged economic relationships suggests opportunities for forward-looking analysis. If current economic conditions predict prices twelve months hence, investors can incorporate this leading information into acquisition and disposition decisions.

5.3 Implications for Mortgage Lenders

Mortgage lenders may apply the findings of this study in several ways. First, price forecasts can inform stress testing of loan portfolios, allowing lenders to assess potential losses under adverse scenarios. The negative R-squared values observed during the test period highlight the importance of considering tail risks that exceed historical experience.

Second, the lagged relationship between mortgage rates and housing prices has direct implications for collateral valuation. Rate changes implemented today will affect property values with a delay, suggesting that loan-to-value calculations should incorporate forward-looking price projections rather than relying solely on current appraisals.

Third, geographic diversification benefits may be assessed by applying similar forecasting approaches across multiple markets, identifying regions with differing risk profiles.

5.4 Implications for Policy Makers

Policy makers may draw several insights from this analysis. First, the difficulty of forecasting housing prices during the COVID-19 period underscores the challenge of anticipating market behavior during extraordinary circumstances. Policy responses must account for fundamental uncertainty about market trajectories.

Second, the lagged structure of economic relationships implies that policy interventions require extended time horizons to achieve full effect. Interest rate changes, for example, appear to affect housing prices with approximately twelve-month delay, suggesting that monetary policy actions must be initiated well in advance of desired market outcomes.

Third, systematic forecast errors—situations where actual prices consistently exceed or fall short of predictions—may serve as early warning indicators of market imbalances. Monitoring forecast performance could complement other surveillance tools employed to assess housing market conditions.

5.5 Limitations

Several limitations of this study should be acknowledged. First, the analysis employs state-level median prices, which aggregate across Connecticut's diverse submarkets. Significant variation exists between high-cost areas such as Fairfield County and more affordable regions in eastern Connecticut. Substate analysis would provide more granular insights.

Second, the set of economic predictor variables, while covering major macroeconomic factors, does not exhaust potentially relevant information. Variables such as housing inventory levels, new construction permits, migration flows, and local employment composition could enhance model performance.

Third, the models were implemented with default or lightly-tuned hyperparameters. More extensive cross-validation and hyperparameter optimization procedures could potentially improve results.

Fourth, the COVID-19 pandemic represents a structural break of unusual magnitude. While including this period provides a rigorous test of model robustness, it may not be representative of typical forecasting challenges.


6. CONCLUSION

6.1 Summary of Findings

This study conducted a comprehensive evaluation of forecasting methods for Connecticut residential housing prices. Analysis of six models spanning traditional time series approaches to contemporary machine learning techniques yields several key conclusions.

The Holt-Winters exponential smoothing model achieved the best predictive performance, with a Mean Absolute Percentage Error of 19.05 percent over the 2019-2024 test period. This result demonstrates that traditional time series methods remain highly competitive for housing price forecasting, particularly when markets experience structural changes that invalidate historical relationships between prices and economic predictors.

Automated model selection procedures provide substantial value. The Auto-SARIMAX approach achieved 23.62 percent MAPE compared to over eighty percent for manually-specified alternatives, highlighting the importance of systematic parameter optimization.

Economic relationships in housing markets operate with significant temporal lags. Models incorporating twelve-month lagged economic indicators achieved dramatically better performance than those using contemporaneous values, with MAPE improving from 40.12 percent to 11.53 percent.

Forecast uncertainty must be explicitly acknowledged in practical applications. Even the best-performing model exhibits nearly twenty percent average error, indicating that point forecasts should be supplemented with uncertainty quantification and scenario analysis.

6.2 Recommendations for Practice

Based on the findings of this study, several recommendations can be offered to practitioners. For short-term forecasting horizons where capturing trend and seasonal patterns is paramount, the Holt-Winters or Auto-SARIMAX methods are recommended. These approaches provide robust performance without requiring specification of external predictor variables.

For longer-horizon forecasts where economic conditions are expected to evolve, regression models incorporating lagged indicators merit consideration. The twelve-month lag structure identified in this analysis suggests that current economic conditions provide meaningful information about future prices.

Forecast uncertainty should be quantified and communicated alongside point predictions. Confidence intervals derived from historical forecast errors provide users with appropriate context for decision-making.

Model performance should be monitored over time. Systematic deterioration in forecast accuracy may indicate structural changes requiring model re-specification.

6.3 Directions for Future Research

Several directions for future research emerge from this study. First, geographic disaggregation to the town or county level would capture local market dynamics that state-level analysis obscures. Second, incorporation of alternative data sources such as online listing information, search trends, or social media sentiment could enhance predictive performance. Third, application of deep learning methods designed for sequential data, such as Long Short-Term Memory networks, represents a promising avenue. Fourth, development of probabilistic forecasting methods that provide full predictive distributions rather than point estimates would enable more sophisticated risk assessment.

The residential housing market's inherent complexity and susceptibility to external shocks ensures that perfect forecasting accuracy will remain elusive. However, rigorous model development, systematic evaluation, and appropriate uncertainty quantification can meaningfully improve decision-making for the diverse stakeholders who depend on housing market insights.


REFERENCES

Bates, J. M., & Granger, C. W. J. (1969). The combination of forecasts. Operations Research Quarterly, 20(4), 451-468.

Board of Governors of the Federal Reserve System. (2023). Survey of Consumer Finances. Washington, DC: Federal Reserve Board.

Crawford, G. W., & Fratantoni, M. C. (2003). Assessing the forecasting performance of regime-switching, ARIMA and GARCH models of house prices. Real Estate Economics, 31(2), 223-243.

Himmelberg, C., Mayer, C., & Sinai, T. (2005). Assessing high house prices: Bubbles, fundamentals and misperceptions. Journal of Economic Perspectives, 19(4), 67-92.

Hyndman, R. J., & Athanasopoulos, G. (2021). Forecasting: Principles and Practice (3rd ed.). Melbourne, Australia: OTexts.

Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2018). Statistical and machine learning forecasting methods: Concerns and ways forward. PLoS One, 13(3), e0194889.

Rosen, S. (1974). Hedonic prices and implicit markets: Product differentiation in pure competition. Journal of Political Economy, 82(1), 34-55.

Taylor, S. J., & Letham, B. (2018). Forecasting at scale. The American Statistician, 72(1), 37-45.

Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. Journal of the Royal Statistical Society: Series B (Methodological), 58(1), 267-288.

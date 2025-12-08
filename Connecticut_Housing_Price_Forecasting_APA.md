Running head: FORECASTING CONNECTICUT HOUSING PRICES                                    1










Forecasting Connecticut Residential Housing Prices:

A Comparative Analysis of Time Series and Machine Learning Models

[Your Name]

[University Name]

MBA Data Analytics

December 2025




















FORECASTING CONNECTICUT HOUSING PRICES                                                   2

Abstract

This study examines the effectiveness of various forecasting methodologies for predicting residential housing prices in Connecticut. Utilizing transaction data from the Connecticut Open Data Portal encompassing over 1.1 million property sales from 2001 to 2024, combined with macroeconomic indicators from the Federal Reserve Economic Data database, six distinct forecasting models were developed and evaluated. The models tested include Naive forecasting, Holt-Winters Exponential Smoothing, Ridge Regression, Facebook Prophet, Ensemble averaging, and Seasonal Autoregressive Integrated Moving Average with Exogenous Variables with automated parameter selection. Results indicate that the Holt-Winters model achieved superior predictive performance with a Mean Absolute Percentage Error of 19.05%, followed by the Auto-SARIMAX model at 23.62%. These findings suggest that traditional time series methods effectively capture housing market dynamics and outperform more complex machine learning approaches during periods of significant market volatility.

Keywords: housing price forecasting, time series analysis, machine learning, Connecticut real estate, predictive modeling
















FORECASTING CONNECTICUT HOUSING PRICES                                                   3

Forecasting Connecticut Residential Housing Prices:

A Comparative Analysis of Time Series and Machine Learning Models

The residential real estate market represents one of the most significant asset classes in the United States economy. According to Federal Reserve data, housing wealth constitutes approximately 70% of total household wealth for the median American family (Board of Governors of the Federal Reserve System, 2023). The ability to accurately forecast housing prices carries substantial implications for multiple stakeholder groups, including real estate investors evaluating portfolio allocation decisions, mortgage lenders conducting risk assessments, policy makers monitoring economic conditions, and individual homeowners making purchase or sale decisions.

Connecticut presents a particularly compelling case for housing market analysis due to its distinctive market characteristics. The state's proximity to New York City generates substantial commuter demand, particularly in Fairfield County. Additionally, Connecticut possesses an aging housing stock, significant wealth concentration in certain municipalities, and has historically exhibited notable price volatility during economic cycles. These factors combine to create a complex forecasting environment that tests the capabilities of various predictive modeling approaches.

Problem Statement

Despite advances in statistical modeling and machine learning, accurate housing price prediction remains challenging. The COVID-19 pandemic, which began affecting markets in early 2020, created unprecedented disruptions to housing market dynamics. Median home prices in Connecticut increased by approximately 40% between 2020 and 2022, driven by historically low mortgage rates, urban-to-suburban migration patterns, and supply chain constraints limiting new construction (National Association of Realtors, 2023). This period of extraordinary volatility provides a rigorous test environment for evaluating forecasting model robustness.



FORECASTING CONNECTICUT HOUSING PRICES                                                   4

Research Objectives

This study pursues four primary objectives: (a) develop a robust data pipeline integrating Connecticut real estate transactions with relevant macroeconomic indicators, (b) implement and compare multiple forecasting methodologies spanning traditional statistical approaches to contemporary machine learning techniques, (c) evaluate model performance using industry-standard metrics during a challenging out-of-sample period encompassing the COVID-19 pandemic, and (d) provide actionable recommendations for practitioners selecting among available forecasting approaches.

Literature Review

Theoretical Framework

Housing prices are determined by the interaction of supply and demand forces operating within local, regional, and national contexts. On the demand side, factors including household income, population growth, mortgage interest rates, and consumer confidence influence the willingness and ability of buyers to purchase properties. On the supply side, construction costs, land availability, zoning regulations, and the existing housing stock determine the quantity of available units. The hedonic pricing model, introduced by Rosen (1974), provides a theoretical framework for understanding how individual property characteristics contribute to overall value.

Time Series Forecasting Methods

The application of time series methods to housing price prediction has a substantial history in the academic literature. Box-Jenkins Autoregressive Integrated Moving Average (ARIMA) models have been widely employed to capture the autocorrelation structures inherent in housing price series (Crawford & Fratantoni, 2003). These methods model future values as a function of past values and past forecast errors, with differencing applied to achieve stationarity.



FORECASTING CONNECTICUT HOUSING PRICES                                                   5

Exponential smoothing methods, including the Holt-Winters approach, have demonstrated strong performance for time series exhibiting both trend and seasonal components (Hyndman & Athanasopoulos, 2021). The Holt-Winters method decomposes a series into level, trend, and seasonal components, with exponential weighting applied to historical observations. This approach offers computational simplicity while providing robust forecasts across diverse applications.

Seasonal ARIMA models extend the basic ARIMA framework to accommodate periodic patterns, making them particularly suitable for housing markets where transaction volumes and prices typically exhibit seasonal fluctuations. The addition of exogenous variables (SARIMAX) allows these models to incorporate external predictors such as economic indicators.

Machine Learning Approaches

Recent years have witnessed growing interest in applying machine learning techniques to housing price prediction. Regularized regression methods, including Ridge and Lasso regression, address the multicollinearity issues common in housing datasets where economic predictors are often highly correlated (Tibshirani, 1996). These methods impose penalties on coefficient magnitudes, reducing overfitting and improving out-of-sample prediction.

Facebook's Prophet algorithm represents a more recent contribution to the forecasting toolkit (Taylor & Letham, 2018). Designed for business time series with strong seasonal patterns, Prophet employs an additive decomposition model with automatic changepoint detection. The algorithm handles missing data and outliers gracefully, making it attractive for practical applications.

Ensemble Methods

The combination of forecasts from multiple models, known as ensemble forecasting, has been shown to improve accuracy by reducing model-specific errors (Bates & Granger, 1969). This


FORECASTING CONNECTICUT HOUSING PRICES                                                   6

approach recognizes that different models may capture different aspects of the underlying data generating process, and their combination can yield more robust predictions than any individual model.

Economic Determinants of Housing Prices

The empirical literature has identified several key macroeconomic factors affecting residential property values. Mortgage interest rates represent perhaps the most significant determinant of housing affordability. Himmelberg et al. (2005) estimate that a one percentage point increase in mortgage rates reduces purchasing power by approximately 10%, with corresponding effects on prices. Unemployment rates reflect labor market conditions and household income stability, affecting both the ability to qualify for mortgages and the willingness to make major purchase commitments. The Consumer Price Index captures inflationary pressures that affect construction costs, land values, and nominal price levels.

Method

Data Sources

This study employs two primary data sources. The first is the Connecticut Real Estate Sales database maintained by the Connecticut Office of Policy and Management and accessible through the Connecticut Open Data Portal. This comprehensive database contains property transaction records dating from 2001 to the present. Each record includes the sale amount, transaction date, property type classification, municipal location, and assessment information.

The second data source is the Federal Reserve Economic Data (FRED) database maintained by the Federal Reserve Bank of St. Louis. From this source, macroeconomic variables were obtained including the 30-year fixed mortgage rate (MORTGAGE30US), Connecticut unemployment rate (CTURN), Consumer Price Index for All Urban Consumers (CPIAUCSL), and Connecticut population estimates (CTPOP).


FORECASTING CONNECTICUT HOUSING PRICES                                                   7

Data Preprocessing

The raw housing transaction data required substantial preprocessing before analysis. First, date fields were parsed and standardized, with records containing invalid date values removed from the dataset. Second, non-arm's length transactions were filtered by excluding sales with amounts below $10,000, which typically represent transfers between family members, foreclosures, or recording errors. Third, the analysis was restricted to residential properties by filtering on property type classifications including single-family homes, condominiums, and multi-family residential buildings.

Following these filtering steps, individual transactions were aggregated to monthly median sale prices. The use of median rather than mean values provides robustness against outliers, which are common in real estate transactions. This aggregation resulted in 231 monthly observations spanning from January 2005 to September 2024. The housing price series was then merged with monthly economic indicator data by date.

Train-Test Split

Following established practices in time series forecasting, the data was divided into training and testing sets using a temporal split. The training period encompasses January 2005 through December 2018, providing 163 monthly observations for model estimation. The testing period spans January 2019 through September 2024, comprising 69 months reserved for out-of-sample evaluation. This split provides adequate historical data for model training while reserving a substantial out-of-sample period that includes both normal market conditions and the COVID-19 disruption.





FORECASTING CONNECTICUT HOUSING PRICES                                                   8

Forecasting Models

Six forecasting models were implemented, representing different methodological approaches to time series prediction.

Naive Model. The naive model serves as a benchmark, using the last observed value as the forecast for all future periods. This simple approach establishes a minimum performance threshold that more sophisticated models should exceed to demonstrate value.

Holt-Winters Exponential Smoothing. The Holt-Winters method extends simple exponential smoothing to accommodate both trend and seasonal components. The model decomposes the time series into three elements: a level component representing the baseline value, a trend component capturing systematic increase or decrease over time, and a seasonal component modeling periodic fluctuations. For this analysis, additive specifications were employed for both trend and seasonality, with a seasonal period of 12 months.

Ridge Regression. Ridge regression addresses multicollinearity among predictor variables by augmenting the ordinary least squares objective function with an L2 penalty on coefficient magnitudes. The regularization parameter was set to 100 for this analysis. Predictor variables included the contemporaneous values of mortgage rate, unemployment rate, Consumer Price Index, and population.

Facebook Prophet. The Prophet algorithm decomposes time series into trend, seasonality, and holiday components using an additive model. For this application, Prophet was configured with yearly seasonality enabled, multiplicative seasonality mode to accommodate the growing price series, and automatic changepoint detection to identify shifts in the underlying trend.



FORECASTING CONNECTICUT HOUSING PRICES                                                   9

Ensemble Model. The ensemble model combines forecasts from the Holt-Winters and Ridge regression models using simple averaging. This approach aims to leverage the complementary strengths of the two methods.

Auto-SARIMAX. The Seasonal Autoregressive Integrated Moving Average with Exogenous Variables model extends the ARIMA framework to include seasonal components and external predictors. The pmdarima library's automatic model selection procedure was employed, searching over candidate specifications and evaluating each using the Akaike Information Criterion.

Evaluation Metrics

Model performance was assessed using four complementary metrics: Root Mean Square Error (RMSE), which provides a measure in dollar terms that penalizes large errors heavily; Mean Absolute Error (MAE), expressed in dollars with linear penalty for errors; Mean Absolute Percentage Error (MAPE), which expresses forecast errors as a percentage of actual values; and Coefficient of Determination (R²), which measures the proportion of variance explained by the model predictions.

Results

Model Performance Comparison

Table 1 presents the comparative performance of all six models on the test set spanning January 2019 through September 2024. Models are ranked by Mean Absolute Percentage Error, with lower values indicating superior predictive accuracy.





FORECASTING CONNECTICUT HOUSING PRICES                                                   10

Table 1

Model Performance Comparison on Test Set (2019-2024)

Model                   RMSE ($)        MAE ($)         MAPE (%)        R²
─────────────────────────────────────────────────────────────────────────────
Holt-Winters            70,551          60,811          19.05           -1.10
Auto-SARIMAX            88,791          75,952          23.62           -2.32
Naive                   91,752          78,967          24.59           -2.55
Prophet                 102,966         88,850          28.07           -3.46
Ensemble                116,471         95,117          29.26           -4.71
Ridge Regression        163,746         130,906         40.12           -10.29
─────────────────────────────────────────────────────────────────────────────

Note. RMSE = Root Mean Square Error; MAE = Mean Absolute Error; MAPE = Mean Absolute Percentage Error. Models ranked by MAPE (lower is better).


The Holt-Winters exponential smoothing model achieved the best performance across all four evaluation metrics. With a MAPE of 19.05%, the model's forecasts deviated from actual prices by an average of approximately $61,000 over the test period. The Auto-SARIMAX model ranked second with a MAPE of 23.62%, representing a meaningful improvement over the Naive benchmark.

Notably, the Naive model outperformed several more sophisticated approaches, including Prophet, the Ensemble, and Ridge regression. This counterintuitive result reflects the extraordinary nature of the test period, during which housing prices deviated substantially from patterns observed in the training data.

Interpretation of Negative R-squared Values

All models exhibit negative R² values on the test set. This outcome is expected when test period dynamics differ substantially from training period patterns. A negative R² indicates that model predictions explain less variance than a simple historical mean, which can occur when the test period represents a structural break from historical norms.


FORECASTING CONNECTICUT HOUSING PRICES                                                   11

The COVID-19 pandemic triggered an unprecedented housing market surge driven by multiple factors operating simultaneously: mortgage interest rates declined to historic lows below 3% in 2020-2021; urban-to-suburban migration accelerated as remote work arrangements enabled household relocation; supply chain disruptions limited new construction; and fiscal stimulus programs increased household savings. These extraordinary conditions produced price increases of approximately 40% over a two-year period, a trajectory no model trained on pre-2019 data could reasonably have anticipated.

Analysis of Individual Model Performance

The success of the Holt-Winters model can be attributed to several factors. First, the method effectively captures the underlying upward trend in Connecticut housing prices that has persisted across the full sample period. Second, it models the seasonal pattern in housing transactions, where spring and summer months typically exhibit higher prices. Third, the exponential smoothing approach adapts gradually to changing conditions without overfitting to noise.

The Auto-SARIMAX model's performance merits attention. The automated parameter selection procedure identified ARIMA(0,1,1)(0,0,0)[12] as the optimal specification, indicating that first-order differencing adequately captures the trend component and a single moving average term models short-term dependencies. The MAPE of 23.62% represents substantial improvement over manually-specified SARIMAX models tested in preliminary analysis, which exhibited MAPE values exceeding 80%.

Impact of Lagged Economic Indicators

Supplementary analysis examined whether incorporating lagged economic indicators improves regression model performance. The rationale stems from recognition that economic conditions affect housing prices with delay—changes in mortgage rates require time to filter


FORECASTING CONNECTICUT HOUSING PRICES                                                   12

through the market as prospective buyers adjust behavior and sellers respond to changing demand. Table 2 presents results.

Table 2

Impact of Lagged Economic Indicators on Ridge Regression Performance

Configuration                           MAPE (%)                R²
─────────────────────────────────────────────────────────────────────────────
Contemporaneous indicators              40.12                   -10.29
12-month lagged indicators              11.53                   +0.23
─────────────────────────────────────────────────────────────────────────────

Note. MAPE = Mean Absolute Percentage Error.


The incorporation of lagged indicators produces dramatic improvement. The MAPE decreases from 40.12% to 11.53%, and R² shifts from negative to positive, indicating the model explains meaningful variance in housing prices. This finding suggests that relationships between economic conditions and housing prices operate with significant temporal delay.

Discussion

The results yield several important insights for housing price forecasting. First, simpler models demonstrated competitive or superior performance relative to more complex alternatives during significant market disruption. The Holt-Winters method, relying solely on historical price patterns without incorporating external economic information, outperformed regression models and machine learning approaches utilizing multiple predictor variables.

This finding aligns with the broader forecasting literature documenting the "forecasting paradox" whereby simple methods often match or exceed sophisticated techniques (Makridakis et al., 2018). Complex models with numerous parameters are susceptible to overfitting historical patterns that may not persist. During structural change periods such as the COVID-19 pandemic, this overfitting can result in particularly poor performance.


FORECASTING CONNECTICUT HOUSING PRICES                                                   13

Second, automated model selection procedures demonstrated clear value. The Auto-SARIMAX approach achieved substantially better results than manually-specified alternatives, suggesting practitioners should leverage available computational tools rather than relying on judgment-based parameter selection.

Third, analysis of lagged economic indicators revealed important housing market dynamics. The 12-month lag structure implies current economic conditions provide information about housing prices one year hence. Theoretically, this suggests housing markets adjust slowly to changing economic conditions due to search frictions, contracting delays, and expectation formation processes. Practically, models incorporating lagged predictors may offer superior forecasting performance.

Implications for Practice

Real Estate Investors. The approximately 19% MAPE achieved by the best-performing model implies substantial forecast uncertainty requiring incorporation into investment decisions. Point forecasts should be supplemented with confidence intervals, and sensitivity analysis should examine investment performance under alternative price scenarios. The trend and seasonal components identified by Holt-Winters provide actionable insights regarding long-term price direction and optimal transaction timing.

Mortgage Lenders. Price forecasts can inform stress testing of loan portfolios, allowing assessment of potential losses under adverse scenarios. The lagged relationship between mortgage rates and housing prices has implications for collateral valuation—rate changes implemented today will affect property values with delay, suggesting loan-to-value calculations should incorporate forward-looking price projections.



FORECASTING CONNECTICUT HOUSING PRICES                                                   14

Policy Makers. The difficulty forecasting housing prices during COVID-19 underscores challenges anticipating market behavior during extraordinary circumstances. The lagged structure of economic relationships implies policy interventions require extended time horizons to achieve full effect—interest rate changes appear to affect housing prices with approximately 12-month delay.

Limitations

Several limitations should be acknowledged. First, the analysis employs state-level median prices aggregating across Connecticut's diverse submarkets. Second, the economic predictor variables, while covering major factors, do not exhaust potentially relevant information such as housing inventory levels or migration flows. Third, models were implemented with default or lightly-tuned hyperparameters; extensive cross-validation could potentially improve results. Fourth, the COVID-19 pandemic represents a structural break of unusual magnitude that may not represent typical forecasting challenges.

Conclusion

This study conducted comprehensive evaluation of forecasting methods for Connecticut residential housing prices. Analysis of six models spanning traditional time series approaches to machine learning techniques yields several conclusions.

The Holt-Winters exponential smoothing model achieved best predictive performance with MAPE of 19.05% over the 2019-2024 test period, demonstrating that traditional time series methods remain highly competitive for housing price forecasting, particularly when markets experience structural changes invalidating historical relationships between prices and economic predictors.




FORECASTING CONNECTICUT HOUSING PRICES                                                   15

Automated model selection procedures provide substantial value—Auto-SARIMAX achieved 23.62% MAPE compared to over 80% for manually-specified alternatives. Economic relationships in housing markets operate with significant temporal lags, and models incorporating 12-month lagged economic indicators achieved dramatically better performance.

For practitioners, a pragmatic approach is recommended: use Holt-Winters or Auto-SARIMAX for short-term forecasting where capturing trend and seasonality is paramount, while incorporating lagged economic indicators in regression frameworks for longer-horizon projections. Forecast uncertainty must be explicitly acknowledged—even the best model exhibits nearly 20% average error, indicating point forecasts should be supplemented with uncertainty quantification and scenario analysis.

Future research directions include geographic disaggregation to town or county levels, incorporation of alternative data sources, application of deep learning methods, and development of probabilistic forecasting approaches providing full predictive distributions.

















FORECASTING CONNECTICUT HOUSING PRICES                                                   16

References

Bates, J. M., & Granger, C. W. J. (1969). The combination of forecasts. Operations Research

        Quarterly, 20(4), 451-468. https://doi.org/10.1057/jors.1969.103

Board of Governors of the Federal Reserve System. (2023). Survey of consumer finances.

        Federal Reserve Board.

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

National Association of Realtors. (2023). Housing statistics. https://www.nar.realtor/research-

        and-statistics/housing-statistics

Rosen, S. (1974). Hedonic prices and implicit markets: Product differentiation in pure

        competition. Journal of Political Economy, 82(1), 34-55. https://doi.org/10.1086/260169




FORECASTING CONNECTICUT HOUSING PRICES                                                   17

Taylor, S. J., & Letham, B. (2018). Forecasting at scale. The American Statistician, 72(1), 37-45.

        https://doi.org/10.1080/00031305.2017.1380080

Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. Journal of the Royal

        Statistical Society: Series B (Methodological), 58(1), 267-288.

        https://doi.org/10.1111/j.2517-6161.1996.tb02080.x

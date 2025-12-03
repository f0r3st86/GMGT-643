"""
Forecasting Models Module
Implements five forecasting approaches:
1. Naïve Model
2. Seasonal Naïve Model
3. Moving Average Model
4. Holt-Winters Exponential Smoothing
5. OLS Regression with Time Trend and Monthly Dummies
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.api import SimpleExpSmoothing
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')


class ForecastingModel:
    """
    Base class for forecasting models.
    """

    def __init__(self, name):
        """
        Initialize the model.

        Parameters:
        -----------
        name : str
            Model name
        """
        self.name = name
        self.fitted = False
        self.train_data = None
        self.test_data = None

    def fit(self, train_data):
        """
        Fit the model to training data.

        Parameters:
        -----------
        train_data : pd.Series or pd.DataFrame
            Training data
        """
        raise NotImplementedError("Subclasses must implement fit()")

    def predict(self, steps):
        """
        Generate forecasts.

        Parameters:
        -----------
        steps : int
            Number of steps ahead to forecast

        Returns:
        --------
        np.array
            Forecasted values
        """
        raise NotImplementedError("Subclasses must implement predict()")

    def get_params(self):
        """
        Get model parameters.

        Returns:
        --------
        dict
            Model parameters
        """
        return {'name': self.name}


class NaiveModel(ForecastingModel):
    """
    Naïve Forecast Model.

    Uses the most recent observed value as the forecast for all future periods.
    """

    def __init__(self):
        super().__init__("Naïve")
        self.last_value = None

    def fit(self, train_data):
        """
        Fit the model by storing the last observed value.

        Parameters:
        -----------
        train_data : pd.Series
            Training data
        """
        self.train_data = train_data
        self.last_value = train_data.iloc[-1]
        self.fitted = True

    def predict(self, steps):
        """
        Generate forecasts (all equal to last value).

        Parameters:
        -----------
        steps : int
            Number of steps ahead

        Returns:
        --------
        np.array
            Forecasted values
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")

        return np.full(steps, self.last_value)

    def get_params(self):
        return {
            'name': self.name,
            'last_value': self.last_value
        }


class SeasonalNaiveModel(ForecastingModel):
    """
    Seasonal Naïve Forecast Model.

    Uses the value from the same season in the previous year.
    For monthly data, uses the value from 12 months ago.
    """

    def __init__(self, seasonal_period=12):
        super().__init__("Seasonal Naïve")
        self.seasonal_period = seasonal_period
        self.seasonal_values = None

    def fit(self, train_data):
        """
        Fit the model by storing the last seasonal period of values.

        Parameters:
        -----------
        train_data : pd.Series
            Training data
        """
        self.train_data = train_data
        # Store the last seasonal_period values
        self.seasonal_values = train_data.iloc[-self.seasonal_period:].values
        self.fitted = True

    def predict(self, steps):
        """
        Generate forecasts using seasonal pattern.

        Parameters:
        -----------
        steps : int
            Number of steps ahead

        Returns:
        --------
        np.array
            Forecasted values
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")

        # Repeat the seasonal pattern
        forecasts = []
        for i in range(steps):
            season_index = i % self.seasonal_period
            forecasts.append(self.seasonal_values[season_index])

        return np.array(forecasts)

    def get_params(self):
        return {
            'name': self.name,
            'seasonal_period': self.seasonal_period
        }


class MovingAverageModel(ForecastingModel):
    """
    Moving Average Forecast Model.

    Uses the average of the last k observations as the forecast.
    """

    def __init__(self, window=12):
        super().__init__("Moving Average")
        self.window = window
        self.ma_value = None

    def fit(self, train_data):
        """
        Fit the model by calculating the moving average.

        Parameters:
        -----------
        train_data : pd.Series
            Training data
        """
        self.train_data = train_data
        # Calculate the average of the last 'window' observations
        self.ma_value = train_data.iloc[-self.window:].mean()
        self.fitted = True

    def predict(self, steps):
        """
        Generate forecasts (all equal to moving average).

        Parameters:
        -----------
        steps : int
            Number of steps ahead

        Returns:
        --------
        np.array
            Forecasted values
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")

        return np.full(steps, self.ma_value)

    def get_params(self):
        return {
            'name': self.name,
            'window': self.window,
            'ma_value': self.ma_value
        }


class HoltWintersModel(ForecastingModel):
    """
    Holt-Winters Exponential Smoothing Model.

    Captures level, trend, and seasonal components with adaptive weighting.
    """

    def __init__(self, seasonal='add', seasonal_periods=12, trend='add'):
        super().__init__("Holt-Winters")
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.trend = trend
        self.model = None
        self.fitted_model = None

    def fit(self, train_data):
        """
        Fit the Holt-Winters model.

        Parameters:
        -----------
        train_data : pd.Series
            Training data
        """
        self.train_data = train_data

        try:
            # Fit Holt-Winters model
            self.model = ExponentialSmoothing(
                train_data,
                seasonal_periods=self.seasonal_periods,
                trend=self.trend,
                seasonal=self.seasonal,
                initialization_method='estimated'
            )

            self.fitted_model = self.model.fit(optimized=True)
            self.fitted = True

        except Exception as e:
            print(f"Error fitting Holt-Winters model: {e}")
            print("Trying with different parameters...")

            # Try with multiplicative seasonal
            try:
                self.seasonal = 'mul'
                self.model = ExponentialSmoothing(
                    train_data,
                    seasonal_periods=self.seasonal_periods,
                    trend=self.trend,
                    seasonal=self.seasonal,
                    initialization_method='estimated'
                )
                self.fitted_model = self.model.fit(optimized=True)
                self.fitted = True
            except:
                raise ValueError("Could not fit Holt-Winters model")

    def predict(self, steps):
        """
        Generate forecasts using the fitted model.

        Parameters:
        -----------
        steps : int
            Number of steps ahead

        Returns:
        --------
        np.array
            Forecasted values
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")

        forecasts = self.fitted_model.forecast(steps=steps)
        return forecasts.values

    def get_params(self):
        params = {
            'name': self.name,
            'seasonal': self.seasonal,
            'seasonal_periods': self.seasonal_periods,
            'trend': self.trend
        }

        if self.fitted_model is not None:
            params.update({
                'alpha': self.fitted_model.params['smoothing_level'],
                'beta': self.fitted_model.params.get('smoothing_trend', None),
                'gamma': self.fitted_model.params.get('smoothing_seasonal', None)
            })

        return params


class OLSRegressionModel(ForecastingModel):
    """
    OLS Regression Model with Time Trend and Monthly Dummies.

    Uses a linear time trend and monthly dummy variables to capture
    trend and seasonality.
    """

    def __init__(self):
        super().__init__("OLS Regression")
        self.model = None
        self.coefficients = None
        self.intercept = None
        self.train_length = None

    def fit(self, train_data):
        """
        Fit OLS regression model.

        Parameters:
        -----------
        train_data : pd.Series or pd.DataFrame
            Training data (should have datetime index or be a series)
        """
        self.train_data = train_data
        n = len(train_data)
        self.train_length = n

        # Create time trend variable
        time_trend = np.arange(1, n + 1)

        # Create monthly dummy variables
        if isinstance(train_data.index, pd.DatetimeIndex):
            months = list(train_data.index.month)
        else:
            # Assume monthly frequency starting from first observation
            months = [(i % 12) + 1 for i in range(n)]

        # Create dummy variables (exclude one month to avoid multicollinearity)
        month_dummies = pd.get_dummies(months, prefix='month', drop_first=True)
        # Ensure numeric dtype
        for col in month_dummies.columns:
            month_dummies[col] = month_dummies[col].astype(int)

        # Combine features
        X = pd.DataFrame({
            'time_trend': time_trend
        })
        X = pd.concat([X, month_dummies], axis=1)

        # Target variable
        y = train_data.values

        # Fit OLS model using statsmodels
        X_with_const = sm.add_constant(X)
        self.model = sm.OLS(y, X_with_const).fit()

        self.coefficients = self.model.params
        self.fitted = True

    def predict(self, steps):
        """
        Generate forecasts using the fitted regression model.

        Parameters:
        -----------
        steps : int
            Number of steps ahead

        Returns:
        --------
        np.array
            Forecasted values
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")

        # Create future time trend
        time_trend = np.arange(self.train_length + 1, self.train_length + steps + 1)

        # Create future monthly dummies
        # Determine the last month of training data
        last_month = self.train_length % 12
        future_months = [((last_month + i) % 12) + 1 for i in range(steps)]

        month_dummies = pd.get_dummies(future_months, prefix='month', drop_first=True)
        # Ensure numeric dtype
        for col in month_dummies.columns:
            month_dummies[col] = month_dummies[col].astype(int)

        # Combine features
        X_future = pd.DataFrame({
            'time_trend': time_trend
        })
        X_future = pd.concat([X_future, month_dummies], axis=1)

        # Ensure all columns from training are present
        for col in self.model.params.index:
            if col != 'const' and col not in X_future.columns:
                X_future[col] = 0

        # Reorder columns to match training
        X_future = X_future[self.model.params.index[1:]]  # Exclude 'const'

        # Add constant
        X_future_with_const = sm.add_constant(X_future)

        # Generate predictions
        forecasts = self.model.predict(X_future_with_const)

        return forecasts.values

    def get_params(self):
        params = {
            'name': self.name,
            'train_length': self.train_length
        }

        if self.model is not None:
            params.update({
                'r_squared': self.model.rsquared,
                'adj_r_squared': self.model.rsquared_adj,
                'aic': self.model.aic,
                'bic': self.model.bic
            })

        return params


class OLSWithExogenousModel(ForecastingModel):
    """
    OLS Regression Model with Exogenous Variables.

    Extends the basic OLS model to include external economic indicators
    (e.g., FRED data) as additional predictors.
    """

    def __init__(self, exog_columns=None, include_trend=True, include_seasonality=True):
        """
        Initialize the model.

        Parameters:
        -----------
        exog_columns : list of str, optional
            List of column names to use as exogenous variables.
            If None, all numeric columns except 'median_price' will be used.
        include_trend : bool
            Whether to include a time trend variable
        include_seasonality : bool
            Whether to include monthly dummy variables
        """
        super().__init__("OLS with Exogenous")
        self.exog_columns = exog_columns
        self.include_trend = include_trend
        self.include_seasonality = include_seasonality
        self.model = None
        self.train_length = None
        self.feature_columns = None
        self.exog_means = None  # For filling missing future values

    def fit(self, train_data, exog_data=None):
        """
        Fit OLS regression model with exogenous variables.

        Parameters:
        -----------
        train_data : pd.Series
            Target variable (housing prices)
        exog_data : pd.DataFrame, optional
            Exogenous variables (economic indicators)
        """
        self.train_data = train_data
        n = len(train_data)
        self.train_length = n

        # Start building feature matrix
        X = pd.DataFrame(index=range(n))

        # Add time trend
        if self.include_trend:
            X['time_trend'] = np.arange(1, n + 1)

        # Add monthly dummies
        if self.include_seasonality:
            if isinstance(train_data.index, pd.DatetimeIndex):
                months = list(train_data.index.month)
            else:
                months = [(i % 12) + 1 for i in range(n)]
            month_dummies = pd.get_dummies(months, prefix='month', drop_first=True)
            month_dummies.index = X.index
            # Ensure all columns are numeric
            for col in month_dummies.columns:
                month_dummies[col] = month_dummies[col].astype(int)
            X = pd.concat([X, month_dummies], axis=1)

        # Add exogenous variables
        if exog_data is not None:
            # Determine which columns to use
            if self.exog_columns is None:
                # Use all numeric columns
                numeric_cols = exog_data.select_dtypes(include=[np.number]).columns.tolist()
                # Exclude price-related columns
                exclude_cols = ['median_price', 'mean_price', 'count', 'std_price']
                self.exog_columns = [c for c in numeric_cols if c not in exclude_cols]

            # Align and add exogenous data
            exog_subset = exog_data[self.exog_columns].copy()
            exog_subset.index = X.index

            # Store means for filling future missing values
            self.exog_means = exog_subset.mean()

            # Fill any missing values with column means
            exog_subset = exog_subset.fillna(self.exog_means)

            X = pd.concat([X, exog_subset], axis=1)

        self.feature_columns = X.columns.tolist()

        # Target variable
        y = train_data.values

        # Drop any rows with NaN (from lagged features)
        valid_idx = ~X.isna().any(axis=1)
        X_clean = X[valid_idx]
        y_clean = y[valid_idx]

        # Fit OLS model
        X_with_const = sm.add_constant(X_clean)
        self.model = sm.OLS(y_clean, X_with_const).fit()

        self.fitted = True

        # Print model summary
        print(f"\n{self.name} Model Summary:")
        print(f"  R-squared: {self.model.rsquared:.4f}")
        print(f"  Adj R-squared: {self.model.rsquared_adj:.4f}")
        print(f"  Features used: {len(self.feature_columns)}")
        if self.exog_columns:
            print(f"  Exogenous variables: {len(self.exog_columns)}")

    def predict(self, steps, future_exog=None):
        """
        Generate forecasts using the fitted regression model.

        Parameters:
        -----------
        steps : int
            Number of steps ahead
        future_exog : pd.DataFrame, optional
            Future values of exogenous variables.
            If None, uses last known values or means.

        Returns:
        --------
        np.array
            Forecasted values
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")

        # Build future feature matrix
        X_future = pd.DataFrame(index=range(steps))

        # Add time trend
        if self.include_trend:
            X_future['time_trend'] = np.arange(
                self.train_length + 1,
                self.train_length + steps + 1
            )

        # Add monthly dummies
        if self.include_seasonality:
            last_month = self.train_length % 12
            future_months = [((last_month + i) % 12) + 1 for i in range(steps)]
            month_dummies = pd.get_dummies(future_months, prefix='month', drop_first=True)
            month_dummies.index = X_future.index
            # Ensure all columns are numeric
            for col in month_dummies.columns:
                month_dummies[col] = month_dummies[col].astype(int)
            X_future = pd.concat([X_future, month_dummies], axis=1)

        # Add exogenous variables
        if self.exog_columns:
            if future_exog is not None and len(future_exog) >= steps:
                # Use provided future values
                exog_future = future_exog[self.exog_columns].iloc[:steps].copy()
                exog_future.index = X_future.index
                exog_future = exog_future.fillna(self.exog_means)
            else:
                # Use means as placeholder
                print("Warning: Using mean values for exogenous variables in forecast")
                exog_future = pd.DataFrame(
                    {col: [self.exog_means[col]] * steps for col in self.exog_columns},
                    index=X_future.index
                )
            X_future = pd.concat([X_future, exog_future], axis=1)

        # Ensure all columns from training are present
        for col in self.feature_columns:
            if col not in X_future.columns:
                X_future[col] = 0

        # Reorder columns to match training
        X_future = X_future[self.feature_columns]

        # Add constant
        X_future_with_const = sm.add_constant(X_future, has_constant='add')

        # Ensure constant column exists
        if 'const' not in X_future_with_const.columns:
            X_future_with_const.insert(0, 'const', 1)

        # Generate predictions
        forecasts = self.model.predict(X_future_with_const)

        return np.array(forecasts)

    def get_params(self):
        params = {
            'name': self.name,
            'train_length': self.train_length,
            'include_trend': self.include_trend,
            'include_seasonality': self.include_seasonality,
            'exog_columns': self.exog_columns
        }

        if self.model is not None:
            params.update({
                'r_squared': self.model.rsquared,
                'adj_r_squared': self.model.rsquared_adj,
                'aic': self.model.aic,
                'bic': self.model.bic,
                'n_features': len(self.feature_columns)
            })

            # Add top 5 most significant coefficients
            if hasattr(self.model, 'pvalues'):
                sig_features = self.model.pvalues.sort_values().head(5)
                params['top_significant_features'] = sig_features.to_dict()

        return params

    def get_feature_importance(self):
        """
        Get feature importance based on t-statistics.

        Returns:
        --------
        pd.DataFrame
            DataFrame with coefficients, t-stats, and p-values
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")

        importance = pd.DataFrame({
            'coefficient': self.model.params,
            't_stat': self.model.tvalues,
            'p_value': self.model.pvalues,
            'abs_t_stat': np.abs(self.model.tvalues)
        }).sort_values('abs_t_stat', ascending=False)

        return importance


class ModelFactory:
    """
    Factory class to create forecasting models.
    """

    @staticmethod
    def create_model(model_name, **kwargs):
        """
        Create a forecasting model by name.

        Parameters:
        -----------
        model_name : str
            Model name: 'naive', 'seasonal_naive', 'ma', 'holt_winters', 'ols'
        **kwargs : dict
            Model-specific parameters

        Returns:
        --------
        ForecastingModel
            Forecasting model instance
        """
        if model_name.lower() == 'naive':
            return NaiveModel()

        elif model_name.lower() == 'seasonal_naive':
            seasonal_period = kwargs.get('seasonal_period', 12)
            return SeasonalNaiveModel(seasonal_period=seasonal_period)

        elif model_name.lower() == 'ma' or model_name.lower() == 'moving_average':
            window = kwargs.get('window', 12)
            return MovingAverageModel(window=window)

        elif model_name.lower() == 'holt_winters' or model_name.lower() == 'hw':
            seasonal = kwargs.get('seasonal', 'add')
            seasonal_periods = kwargs.get('seasonal_periods', 12)
            trend = kwargs.get('trend', 'add')
            return HoltWintersModel(seasonal=seasonal, seasonal_periods=seasonal_periods, trend=trend)

        elif model_name.lower() == 'ols' or model_name.lower() == 'regression':
            return OLSRegressionModel()

        elif model_name.lower() == 'ols_exog' or model_name.lower() == 'ols_with_exogenous':
            exog_columns = kwargs.get('exog_columns', None)
            include_trend = kwargs.get('include_trend', True)
            include_seasonality = kwargs.get('include_seasonality', True)
            return OLSWithExogenousModel(
                exog_columns=exog_columns,
                include_trend=include_trend,
                include_seasonality=include_seasonality
            )

        else:
            raise ValueError(f"Unknown model: {model_name}")

    @staticmethod
    def create_all_models():
        """
        Create all five forecasting models.

        Returns:
        --------
        dict
            Dictionary of model name -> model instance
        """
        return {
            'naive': NaiveModel(),
            'seasonal_naive': SeasonalNaiveModel(seasonal_period=12),
            'moving_average': MovingAverageModel(window=12),
            'holt_winters': HoltWintersModel(seasonal='add', seasonal_periods=12, trend='add'),
            'ols': OLSRegressionModel()
        }


def fit_and_forecast_all_models(train_data, forecast_steps):
    """
    Fit all models and generate forecasts.

    Parameters:
    -----------
    train_data : pd.Series
        Training data
    forecast_steps : int
        Number of steps to forecast

    Returns:
    --------
    dict
        Dictionary of model name -> forecasts
    """
    models = ModelFactory.create_all_models()
    forecasts = {}

    for name, model in models.items():
        print(f"Fitting {name} model...")
        try:
            model.fit(train_data)
            forecasts[name] = model.predict(forecast_steps)
            print(f"  ✓ {name} fitted successfully")
        except Exception as e:
            print(f"  ✗ Error fitting {name}: {e}")
            forecasts[name] = None

    return forecasts, models


def fit_and_forecast_with_exogenous(train_data, exog_train, forecast_steps,
                                     exog_test=None, exog_columns=None):
    """
    Fit models including OLS with exogenous variables and generate forecasts.

    Parameters:
    -----------
    train_data : pd.Series
        Training target data (housing prices)
    exog_train : pd.DataFrame
        Training exogenous data (FRED indicators)
    forecast_steps : int
        Number of steps to forecast
    exog_test : pd.DataFrame, optional
        Test period exogenous data for prediction
    exog_columns : list of str, optional
        Specific columns to use as exogenous variables

    Returns:
    --------
    tuple
        (forecasts dict, models dict)
    """
    print("\n" + "=" * 60)
    print("Fitting Models with Exogenous Variables")
    print("=" * 60)

    # Create all base models
    models = ModelFactory.create_all_models()

    # Add OLS with exogenous model
    models['ols_with_fred'] = OLSWithExogenousModel(exog_columns=exog_columns)

    forecasts = {}

    for name, model in models.items():
        print(f"\nFitting {name} model...")
        try:
            if name == 'ols_with_fred':
                # Fit with exogenous data
                model.fit(train_data, exog_data=exog_train)
                forecasts[name] = model.predict(forecast_steps, future_exog=exog_test)
            else:
                # Fit regular models
                model.fit(train_data)
                forecasts[name] = model.predict(forecast_steps)
            print(f"  {name} fitted successfully")
        except Exception as e:
            print(f"  Error fitting {name}: {e}")
            forecasts[name] = None

    return forecasts, models


def compare_exogenous_impact(train_data, exog_train, test_data, exog_test=None,
                              exog_columns=None):
    """
    Compare OLS model with and without exogenous variables.

    Parameters:
    -----------
    train_data : pd.Series
        Training target data
    exog_train : pd.DataFrame
        Training exogenous data
    test_data : pd.Series
        Test target data for evaluation
    exog_test : pd.DataFrame, optional
        Test period exogenous data
    exog_columns : list of str, optional
        Specific columns to use

    Returns:
    --------
    pd.DataFrame
        Comparison of model performance
    """
    from .evaluation import calculate_rmse, calculate_mae, calculate_mape

    print("\n" + "=" * 60)
    print("Comparing Impact of Exogenous Variables")
    print("=" * 60)

    steps = len(test_data)

    # Fit OLS without exogenous
    ols_basic = OLSRegressionModel()
    ols_basic.fit(train_data)
    pred_basic = ols_basic.predict(steps)

    # Fit OLS with exogenous
    ols_exog = OLSWithExogenousModel(exog_columns=exog_columns)
    ols_exog.fit(train_data, exog_data=exog_train)
    pred_exog = ols_exog.predict(steps, future_exog=exog_test)

    # Calculate metrics
    actual = test_data.values

    results = pd.DataFrame({
        'Model': ['OLS (Basic)', 'OLS (with FRED)'],
        'RMSE': [
            calculate_rmse(actual, pred_basic),
            calculate_rmse(actual, pred_exog)
        ],
        'MAE': [
            calculate_mae(actual, pred_basic),
            calculate_mae(actual, pred_exog)
        ],
        'MAPE': [
            calculate_mape(actual, pred_basic),
            calculate_mape(actual, pred_exog)
        ],
        'R-squared': [
            ols_basic.get_params().get('r_squared', None),
            ols_exog.get_params().get('r_squared', None)
        ]
    })

    # Calculate improvement
    rmse_improvement = (results.loc[0, 'RMSE'] - results.loc[1, 'RMSE']) / results.loc[0, 'RMSE'] * 100
    mape_improvement = (results.loc[0, 'MAPE'] - results.loc[1, 'MAPE']) / results.loc[0, 'MAPE'] * 100

    print("\nModel Comparison:")
    print(results.to_string(index=False))
    print(f"\nRMSE Improvement with FRED data: {rmse_improvement:.1f}%")
    print(f"MAPE Improvement with FRED data: {mape_improvement:.1f}%")

    # Print feature importance
    print("\nTop 10 Most Important Features (by t-statistic):")
    importance = ols_exog.get_feature_importance()
    print(importance.head(10).to_string())

    return results, ols_exog

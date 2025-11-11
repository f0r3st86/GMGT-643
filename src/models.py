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
            months = train_data.index.month
        else:
            # Assume monthly frequency starting from first observation
            months = [(i % 12) + 1 for i in range(n)]

        # Create dummy variables (exclude one month to avoid multicollinearity)
        month_dummies = pd.get_dummies(months, prefix='month', drop_first=True)

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

"""
Rolling Forecast Module
Implements rolling window forecasting for more realistic evaluation.

Instead of generating all forecasts at once, this approach:
1. Trains on historical data
2. Forecasts one period ahead
3. Adds the actual value to training data
4. Re-trains and forecasts the next period
5. Repeats through the entire test period

This simulates real-world forecasting where models are updated as new data arrives.
"""

import pandas as pd
import numpy as np
from models import ModelFactory
from evaluation import calculate_all_metrics
import warnings
warnings.filterwarnings('ignore')


def rolling_forecast_single_model(train_series, test_series, model_name,
                                   horizon=1, update_frequency=1, **model_kwargs):
    """
    Generate rolling forecasts for a single model.

    Parameters:
    -----------
    train_series : pd.Series
        Initial training data
    test_series : pd.Series
        Test data (actual values)
    model_name : str
        Name of the model to use
    horizon : int
        Forecast horizon (1 for one-step-ahead, 12 for one-year-ahead, etc.)
    update_frequency : int
        How often to retrain the model (1 = retrain every period)
    **model_kwargs : dict
        Additional arguments for model initialization

    Returns:
    --------
    dict
        Dictionary with forecasts, actual values, and metadata
    """
    forecasts = []
    train_data = train_series.copy()

    print(f"\nGenerating rolling forecasts for {model_name}...")
    print(f"  Horizon: {horizon} step(s) ahead")
    print(f"  Update frequency: every {update_frequency} period(s)")

    # Iterate through test period
    for i in range(len(test_series)):
        # Only retrain if it's time to update
        if i % update_frequency == 0:
            # Create and fit model
            model = ModelFactory.create_model(model_name, **model_kwargs)
            model.fit(train_data)

        # Generate forecast
        forecast = model.predict(horizon)

        # Take the forecast for the desired horizon
        # For 1-step ahead: forecast[0]
        # For h-step ahead: forecast[horizon-1] if len(forecast) >= horizon
        if horizon == 1:
            forecasts.append(forecast[0])
        else:
            if len(forecast) >= horizon:
                forecasts.append(forecast[horizon-1])
            else:
                forecasts.append(forecast[-1])

        # Add actual value to training data for next iteration
        next_actual = test_series.iloc[i]
        train_data = pd.concat([train_data, pd.Series([next_actual])])

        # Progress indicator
        if (i + 1) % 12 == 0:
            print(f"  Progress: {i+1}/{len(test_series)} forecasts generated")

    print(f"  ✓ Complete: {len(forecasts)} forecasts generated")

    return {
        'forecasts': np.array(forecasts),
        'actual': test_series.values,
        'model_name': model_name,
        'horizon': horizon,
        'update_frequency': update_frequency
    }


def rolling_forecast_all_models(train_series, test_series, horizon=1,
                                update_frequency=1):
    """
    Generate rolling forecasts for all models.

    Parameters:
    -----------
    train_series : pd.Series
        Initial training data
    test_series : pd.Series
        Test data
    horizon : int
        Forecast horizon
    update_frequency : int
        How often to retrain

    Returns:
    --------
    dict
        Dictionary of model_name -> forecast results
    """
    print("="*80)
    print("ROLLING FORECAST ANALYSIS")
    print("="*80)
    print(f"\nForecast configuration:")
    print(f"  Training period start: {train_series.index[0] if hasattr(train_series, 'index') else 'N/A'}")
    print(f"  Training period length: {len(train_series)} observations")
    print(f"  Test period length: {len(test_series)} observations")
    print(f"  Forecast horizon: {horizon} step(s) ahead")
    print(f"  Model update frequency: every {update_frequency} period(s)")

    models_config = {
        'naive': {},
        'seasonal_naive': {'seasonal_period': 12},
        'moving_average': {'window': 12},
        'holt_winters': {'seasonal': 'add', 'seasonal_periods': 12, 'trend': 'add'},
        'ols': {}
    }

    results = {}

    for model_name, model_kwargs in models_config.items():
        try:
            result = rolling_forecast_single_model(
                train_series,
                test_series,
                model_name,
                horizon=horizon,
                update_frequency=update_frequency,
                **model_kwargs
            )
            results[model_name] = result
        except Exception as e:
            print(f"  ✗ Error with {model_name}: {e}")
            results[model_name] = None

    return results


def evaluate_rolling_forecasts(rolling_results):
    """
    Evaluate rolling forecast results.

    Parameters:
    -----------
    rolling_results : dict
        Dictionary of rolling forecast results

    Returns:
    --------
    pd.DataFrame
        Results dataframe with metrics
    """
    print("\n" + "="*80)
    print("ROLLING FORECAST EVALUATION")
    print("="*80)

    results_list = []

    for model_name, result in rolling_results.items():
        if result is None:
            continue

        forecasts = result['forecasts']
        actual = result['actual']

        metrics = calculate_all_metrics(actual, forecasts)
        metrics['model'] = model_name
        metrics['horizon'] = result['horizon']
        metrics['update_frequency'] = result['update_frequency']

        results_list.append(metrics)

    results_df = pd.DataFrame(results_list)

    if len(results_df) > 0:
        results_df = results_df[['model', 'horizon', 'update_frequency', 'rmse', 'mae', 'mape']]
        results_df = results_df.sort_values('rmse')

    return results_df


def compare_static_vs_rolling(train_series, test_series, static_forecasts):
    """
    Compare static vs rolling forecast performance.

    Parameters:
    -----------
    train_series : pd.Series
        Training data
    test_series : pd.Series
        Test data
    static_forecasts : dict
        Static forecast results (from original method)

    Returns:
    --------
    pd.DataFrame
        Comparison dataframe
    """
    print("\n" + "="*80)
    print("STATIC vs ROLLING FORECAST COMPARISON")
    print("="*80)

    # Generate rolling forecasts (1-step-ahead, update every period)
    rolling_results = rolling_forecast_all_models(
        train_series,
        test_series,
        horizon=1,
        update_frequency=1
    )

    # Evaluate rolling forecasts
    rolling_df = evaluate_rolling_forecasts(rolling_results)
    rolling_df['method'] = 'Rolling (1-step)'

    # Evaluate static forecasts
    from evaluation import evaluate_multiple_models
    static_df = evaluate_multiple_models(test_series, static_forecasts)
    static_df['method'] = 'Static (60-step)'
    static_df['horizon'] = 60
    static_df['update_frequency'] = 0

    # Combine results
    comparison_df = pd.concat([static_df, rolling_df], ignore_index=True)
    comparison_df = comparison_df.sort_values(['model', 'method'])

    return comparison_df, rolling_results


def analyze_forecast_improvement(comparison_df):
    """
    Analyze improvement from static to rolling forecasts.

    Parameters:
    -----------
    comparison_df : pd.DataFrame
        Comparison dataframe
    """
    print("\n" + "="*80)
    print("FORECAST IMPROVEMENT ANALYSIS")
    print("="*80)

    models = comparison_df['model'].unique()

    for model in models:
        model_data = comparison_df[comparison_df['model'] == model]

        if len(model_data) < 2:
            continue

        static = model_data[model_data['method'] == 'Static (60-step)'].iloc[0]
        rolling = model_data[model_data['method'] == 'Rolling (1-step)'].iloc[0]

        rmse_improvement = (static['rmse'] - rolling['rmse']) / static['rmse'] * 100
        mae_improvement = (static['mae'] - rolling['mae']) / static['mae'] * 100
        mape_improvement = (static['mape'] - rolling['mape']) / static['mape'] * 100

        print(f"\n{model}:")
        print(f"  Static (60-step):")
        print(f"    RMSE: ${static['rmse']:,.2f}")
        print(f"    MAE:  ${static['mae']:,.2f}")
        print(f"    MAPE: {static['mape']:.2f}%")

        print(f"  Rolling (1-step):")
        print(f"    RMSE: ${rolling['rmse']:,.2f}")
        print(f"    MAE:  ${rolling['mae']:,.2f}")
        print(f"    MAPE: {rolling['mape']:.2f}%")

        print(f"  Improvement:")
        print(f"    RMSE: {rmse_improvement:+.1f}%")
        print(f"    MAE:  {mae_improvement:+.1f}%")
        print(f"    MAPE: {mape_improvement:+.1f}%")


def main():
    """
    Example usage of rolling forecast functionality.
    """
    print("Rolling Forecast Module")
    print("This module provides more realistic forecast evaluation through rolling windows.")
    print("\nTo use this module:")
    print("1. Import: from rolling_forecast import rolling_forecast_all_models")
    print("2. Generate: results = rolling_forecast_all_models(train_series, test_series)")
    print("3. Evaluate: results_df = evaluate_rolling_forecasts(results)")


if __name__ == '__main__':
    main()

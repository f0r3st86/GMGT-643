"""
Evaluation Metrics Module
Calculates forecast accuracy metrics:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
"""

import pandas as pd
import numpy as np


def rmse(actual, predicted):
    """
    Calculate Root Mean Squared Error (RMSE).

    RMSE penalizes large errors more heavily than small errors.
    Lower values indicate better fit.

    Parameters:
    -----------
    actual : array-like
        Actual observed values
    predicted : array-like
        Predicted values

    Returns:
    --------
    float
        RMSE value
    """
    actual = np.array(actual)
    predicted = np.array(predicted)

    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted arrays must have the same length")

    squared_errors = (actual - predicted) ** 2
    mean_squared_error = np.mean(squared_errors)
    return np.sqrt(mean_squared_error)


def mae(actual, predicted):
    """
    Calculate Mean Absolute Error (MAE).

    MAE shows the average error in dollar terms.
    Lower values indicate better fit.

    Parameters:
    -----------
    actual : array-like
        Actual observed values
    predicted : array-like
        Predicted values

    Returns:
    --------
    float
        MAE value
    """
    actual = np.array(actual)
    predicted = np.array(predicted)

    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted arrays must have the same length")

    absolute_errors = np.abs(actual - predicted)
    return np.mean(absolute_errors)


def mape(actual, predicted):
    """
    Calculate Mean Absolute Percentage Error (MAPE).

    MAPE shows the average error as a percentage of the actual value.
    Lower values indicate better fit.

    Parameters:
    -----------
    actual : array-like
        Actual observed values
    predicted : array-like
        Predicted values

    Returns:
    --------
    float
        MAPE value (as a percentage, e.g., 5.0 means 5%)
    """
    actual = np.array(actual)
    predicted = np.array(predicted)

    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted arrays must have the same length")

    # Avoid division by zero
    mask = actual != 0
    if not mask.any():
        raise ValueError("All actual values are zero, cannot calculate MAPE")

    actual = actual[mask]
    predicted = predicted[mask]

    absolute_percentage_errors = np.abs((actual - predicted) / actual) * 100
    return np.mean(absolute_percentage_errors)


def calculate_all_metrics(actual, predicted):
    """
    Calculate all three metrics at once.

    Parameters:
    -----------
    actual : array-like
        Actual observed values
    predicted : array-like
        Predicted values

    Returns:
    --------
    dict
        Dictionary with 'rmse', 'mae', and 'mape' keys
    """
    return {
        'rmse': rmse(actual, predicted),
        'mae': mae(actual, predicted),
        'mape': mape(actual, predicted)
    }


def evaluate_model(actual, predicted, model_name='Model'):
    """
    Evaluate a model and print results.

    Parameters:
    -----------
    actual : array-like
        Actual observed values
    predicted : array-like
        Predicted values
    model_name : str
        Name of the model

    Returns:
    --------
    dict
        Dictionary with evaluation metrics
    """
    metrics = calculate_all_metrics(actual, predicted)

    print(f"\n{model_name} Evaluation:")
    print(f"  RMSE: ${metrics['rmse']:,.2f}")
    print(f"  MAE:  ${metrics['mae']:,.2f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")

    return metrics


def evaluate_multiple_models(actual, forecasts_dict):
    """
    Evaluate multiple models and return results as a DataFrame.

    Parameters:
    -----------
    actual : array-like
        Actual observed values
    forecasts_dict : dict
        Dictionary of model_name -> predicted values

    Returns:
    --------
    pd.DataFrame
        DataFrame with models as rows and metrics as columns
    """
    results = []

    for model_name, predicted in forecasts_dict.items():
        if predicted is None:
            continue

        try:
            metrics = calculate_all_metrics(actual, predicted)
            metrics['model'] = model_name
            results.append(metrics)
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")

    df = pd.DataFrame(results)

    if len(df) > 0:
        # Reorder columns
        df = df[['model', 'rmse', 'mae', 'mape']]

        # Sort by RMSE (best first)
        df = df.sort_values('rmse')

    return df


def compare_models(results_df):
    """
    Compare models and identify the best performer.

    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame with model evaluation results

    Returns:
    --------
    dict
        Dictionary with comparison statistics
    """
    if len(results_df) == 0:
        return {}

    best_model = results_df.iloc[0]
    worst_model = results_df.iloc[-1]

    comparison = {
        'best_model': best_model['model'],
        'best_rmse': best_model['rmse'],
        'best_mae': best_model['mae'],
        'best_mape': best_model['mape'],
        'worst_model': worst_model['model'],
        'worst_rmse': worst_model['rmse'],
        'worst_mae': worst_model['mae'],
        'worst_mape': worst_model['mape']
    }

    # Calculate improvement percentages
    if worst_model['rmse'] > 0:
        comparison['rmse_improvement_pct'] = (
            (worst_model['rmse'] - best_model['rmse']) / worst_model['rmse'] * 100
        )

    if worst_model['mae'] > 0:
        comparison['mae_improvement_pct'] = (
            (worst_model['mae'] - best_model['mae']) / worst_model['mae'] * 100
        )

    if worst_model['mape'] > 0:
        comparison['mape_improvement_pct'] = (
            (worst_model['mape'] - best_model['mape']) / worst_model['mape'] * 100
        )

    return comparison


def print_comparison(results_df):
    """
    Print a formatted comparison of model results.

    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame with model evaluation results
    """
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    print("\nRanked by RMSE (lower is better):")
    print(results_df.to_string(index=False))

    comparison = compare_models(results_df)

    if comparison:
        print(f"\n{'=' * 80}")
        print("KEY FINDINGS")
        print("=" * 80)

        print(f"\nBest Model: {comparison['best_model']}")
        print(f"  RMSE: ${comparison['best_rmse']:,.2f}")
        print(f"  MAE:  ${comparison['best_mae']:,.2f}")
        print(f"  MAPE: {comparison['best_mape']:.2f}%")

        print(f"\nWorst Model: {comparison['worst_model']}")
        print(f"  RMSE: ${comparison['worst_rmse']:,.2f}")
        print(f"  MAE:  ${comparison['worst_mae']:,.2f}")
        print(f"  MAPE: {comparison['worst_mape']:.2f}%")

        if 'rmse_improvement_pct' in comparison:
            print(f"\nImprovement (Best vs. Worst):")
            print(f"  RMSE: {comparison['rmse_improvement_pct']:.1f}%")
            print(f"  MAE:  {comparison['mae_improvement_pct']:.1f}%")
            print(f"  MAPE: {comparison['mape_improvement_pct']:.1f}%")


def calculate_rolling_metrics(actual, predicted, window=12):
    """
    Calculate metrics over rolling windows.

    Useful for analyzing how forecast accuracy changes over time.

    Parameters:
    -----------
    actual : array-like
        Actual observed values
    predicted : array-like
        Predicted values
    window : int
        Rolling window size

    Returns:
    --------
    pd.DataFrame
        DataFrame with rolling metrics
    """
    actual = np.array(actual)
    predicted = np.array(predicted)

    if len(actual) < window:
        raise ValueError(f"Not enough data for rolling window of {window}")

    n = len(actual)
    rolling_metrics = []

    for i in range(window, n + 1):
        actual_window = actual[i - window:i]
        predicted_window = predicted[i - window:i]

        metrics = calculate_all_metrics(actual_window, predicted_window)
        metrics['window_end'] = i
        rolling_metrics.append(metrics)

    return pd.DataFrame(rolling_metrics)


def analyze_crisis_periods(actual, predicted, dates, crisis_periods):
    """
    Analyze forecast accuracy during crisis periods.

    Parameters:
    -----------
    actual : array-like
        Actual observed values
    predicted : array-like
        Predicted values
    dates : pd.DatetimeIndex or array-like
        Dates corresponding to the values
    crisis_periods : list of tuples
        List of (start_date, end_date, name) tuples

    Returns:
    --------
    pd.DataFrame
        DataFrame with metrics for each crisis period
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    dates = pd.to_datetime(dates)

    crisis_results = []

    for start_date, end_date, name in crisis_periods:
        # Find indices for this crisis period
        mask = (dates >= pd.to_datetime(start_date)) & (dates <= pd.to_datetime(end_date))

        if not mask.any():
            print(f"Warning: No data found for {name} ({start_date} to {end_date})")
            continue

        actual_crisis = actual[mask]
        predicted_crisis = predicted[mask]

        metrics = calculate_all_metrics(actual_crisis, predicted_crisis)
        metrics['period'] = name
        metrics['start_date'] = start_date
        metrics['end_date'] = end_date
        metrics['n_observations'] = len(actual_crisis)

        crisis_results.append(metrics)

    return pd.DataFrame(crisis_results)


def save_results(results_df, filepath='results/tables/model_comparison.csv'):
    """
    Save evaluation results to CSV.

    Parameters:
    -----------
    results_df : pd.DataFrame
        Results dataframe
    filepath : str
        Output file path
    """
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    results_df.to_csv(filepath, index=False)
    print(f"\nResults saved to: {filepath}")


if __name__ == '__main__':
    # Example usage
    print("Evaluation Metrics Module")
    print("=" * 80)

    # Example data
    actual = np.array([100, 105, 110, 108, 115])
    predicted = np.array([98, 107, 109, 110, 114])

    # Calculate individual metrics
    print("\nExample: Forecast Evaluation")
    print(f"Actual:    {actual}")
    print(f"Predicted: {predicted}")

    metrics = evaluate_model(actual, predicted, "Example Model")

    # Example with multiple models
    forecasts = {
        'Model A': np.array([98, 107, 109, 110, 114]),
        'Model B': np.array([100, 105, 111, 107, 116]),
        'Model C': np.array([99, 106, 110, 109, 115])
    }

    results = evaluate_multiple_models(actual, forecasts)
    print_comparison(results)

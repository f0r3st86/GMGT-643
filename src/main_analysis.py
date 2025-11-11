"""
Main Analysis Script
Runs the complete Connecticut Housing Price Forecasting Analysis

This script:
1. Loads preprocessed data
2. Splits into training (2001-2018) and testing (2019-2023) periods
3. Fits all 5 forecasting models
4. Generates forecasts
5. Evaluates model performance
6. Creates visualizations
7. Analyzes crisis periods
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules
from models import ModelFactory, fit_and_forecast_all_models
from evaluation import (evaluate_multiple_models, print_comparison,
                       analyze_crisis_periods, save_results)
from visualization import create_all_visualizations, plot_crisis_analysis


def load_data(filepath='data/processed/ct_housing_monthly.csv'):
    """
    Load the processed monthly data.

    Parameters:
    -----------
    filepath : str
        Path to processed data file

    Returns:
    --------
    pd.DataFrame
        Monthly time series data
    """
    print("Loading processed data...")
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    print(f"Loaded {len(df)} monthly observations")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    return df


def split_train_test(df, train_end='2018-12-31', test_start='2019-01-01'):
    """
    Split data into training and testing periods.

    Parameters:
    -----------
    df : pd.DataFrame
        Full dataset
    train_end : str
        End date for training period
    test_start : str
        Start date for testing period

    Returns:
    --------
    tuple
        (train_df, test_df)
    """
    print("\nSplitting data into train and test sets...")

    train_df = df[df['date'] <= pd.to_datetime(train_end)].copy()
    test_df = df[df['date'] >= pd.to_datetime(test_start)].copy()

    print(f"Training set: {len(train_df)} observations ({train_df['date'].min()} to {train_df['date'].max()})")
    print(f"Testing set:  {len(test_df)} observations ({test_df['date'].min()} to {test_df['date'].max()})")

    return train_df, test_df


def run_analysis(data_path='data/processed/ct_housing_monthly.csv',
                output_dir='results'):
    """
    Run the complete forecasting analysis.

    Parameters:
    -----------
    data_path : str
        Path to processed data
    output_dir : str
        Output directory for results
    """
    print("=" * 80)
    print("CONNECTICUT HOUSING PRICE FORECASTING ANALYSIS")
    print("=" * 80)
    print()

    # Create output directories
    os.makedirs(f'{output_dir}/figures', exist_ok=True)
    os.makedirs(f'{output_dir}/tables', exist_ok=True)

    # 1. Load data
    df = load_data(data_path)

    # 2. Split data
    train_df, test_df = split_train_test(df)

    # 3. Prepare data for modeling
    train_series = train_df['median_price']
    test_series = test_df['median_price']
    forecast_steps = len(test_df)

    print(f"\nTraining data statistics:")
    print(train_series.describe())

    # 4. Fit models and generate forecasts
    print("\n" + "=" * 80)
    print("FITTING MODELS AND GENERATING FORECASTS")
    print("=" * 80)

    forecasts_dict, models_dict = fit_and_forecast_all_models(train_series, forecast_steps)

    # Display model parameters
    print("\nModel Parameters:")
    for name, model in models_dict.items():
        print(f"\n{name}:")
        params = model.get_params()
        for key, value in params.items():
            if key != 'name' and value is not None:
                print(f"  {key}: {value}")

    # 5. Evaluate models
    print("\n" + "=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)

    results_df = evaluate_multiple_models(test_series, forecasts_dict)
    print_comparison(results_df)

    # Save results
    save_results(results_df, f'{output_dir}/tables/model_comparison.csv')

    # 6. Create visualizations
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)

    create_all_visualizations(train_df, test_df, forecasts_dict, results_df,
                             output_dir=f'{output_dir}/figures')

    # 7. Crisis period analysis
    print("\n" + "=" * 80)
    print("CRISIS PERIOD ANALYSIS")
    print("=" * 80)

    # Define crisis periods in the test set
    crisis_periods = [
        ('2020-03-01', '2020-12-31', 'COVID-19 Pandemic (2020)'),
    ]

    # For the best model (Holt-Winters expected)
    best_model_name = results_df.iloc[0]['model']
    best_forecasts = forecasts_dict[best_model_name]

    print(f"\nAnalyzing {best_model_name} performance during crisis periods...")

    crisis_results = analyze_crisis_periods(
        test_series.values,
        best_forecasts,
        test_df['date'],
        crisis_periods
    )

    if len(crisis_results) > 0:
        print("\nCrisis Period Performance:")
        print(crisis_results.to_string(index=False))

        crisis_results.to_csv(f'{output_dir}/tables/crisis_analysis.csv', index=False)
        print(f"\nCrisis analysis saved to: {output_dir}/tables/crisis_analysis.csv")

        # Visualize crisis periods
        crisis_periods_vis = [
            ('2020-03-01', '2020-12-31', 'COVID-19', 'orange')
        ]

        plot_crisis_analysis(
            test_df['date'],
            test_series.values,
            best_forecasts,
            crisis_periods_vis,
            model_name=best_model_name,
            save_path=f'{output_dir}/figures/07_crisis_analysis.png'
        )

    # 8. Summary report
    print("\n" + "=" * 80)
    print("SUMMARY REPORT")
    print("=" * 80)

    print(f"\nBest Model: {results_df.iloc[0]['model']}")
    print(f"  RMSE: ${results_df.iloc[0]['rmse']:,.2f}")
    print(f"  MAE:  ${results_df.iloc[0]['mae']:,.2f}")
    print(f"  MAPE: {results_df.iloc[0]['mape']:.2f}%")

    # Check if expectations are met
    best_mape = results_df.iloc[0]['mape']
    print(f"\nExpected MAPE: 4-8%")
    print(f"Actual MAPE: {best_mape:.2f}%")

    if 4 <= best_mape <= 8:
        print("✓ MAPE within expected range")
    elif best_mape < 4:
        print("✓ MAPE better than expected!")
    else:
        print("✗ MAPE higher than expected")

    # Compare best vs worst
    worst_model = results_df.iloc[-1]
    improvement_pct = (worst_model['rmse'] - results_df.iloc[0]['rmse']) / worst_model['rmse'] * 100

    print(f"\nImprovement over worst model ({worst_model['model']}):")
    print(f"  RMSE improvement: {improvement_pct:.1f}%")
    print(f"\nExpected improvement: 20-40%")

    if 20 <= improvement_pct <= 40:
        print("✓ Improvement within expected range")
    elif improvement_pct > 40:
        print("✓ Improvement better than expected!")
    else:
        print("✗ Improvement lower than expected")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}/")
    print(f"  - Figures: {output_dir}/figures/")
    print(f"  - Tables: {output_dir}/tables/")


def main():
    """
    Main entry point.
    """
    # Check if processed data exists
    data_path = 'data/processed/ct_housing_monthly.csv'

    if not os.path.exists(data_path):
        print("ERROR: Processed data not found!")
        print(f"Expected file: {data_path}")
        print("\nPlease run the preprocessing pipeline first:")
        print("  python src/data_preprocessing.py")
        print("\nOr if raw data doesn't exist:")
        print("  python src/data_acquisition.py")
        print("  python src/data_preprocessing.py")
        sys.exit(1)

    # Run analysis
    run_analysis(data_path)


if __name__ == '__main__':
    main()

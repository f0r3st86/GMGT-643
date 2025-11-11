"""
Improved Main Analysis Script
Compares static vs rolling forecast methods

This script demonstrates the improvement in forecast accuracy when using
rolling (walk-forward) forecasts instead of static long-horizon forecasts.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import fit_and_forecast_all_models
from rolling_forecast import (rolling_forecast_all_models, evaluate_rolling_forecasts,
                              compare_static_vs_rolling, analyze_forecast_improvement)
from evaluation import evaluate_multiple_models, print_comparison, save_results
from visualization import create_all_visualizations
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(filepath='data/processed/ct_housing_monthly.csv'):
    """Load processed data."""
    print("Loading processed data...")
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    print(f"Loaded {len(df)} monthly observations")
    return df


def split_train_test(df, train_end='2018-12-31', test_start='2019-01-01'):
    """Split into train and test sets."""
    print("\nSplitting data...")
    train_df = df[df['date'] <= pd.to_datetime(train_end)].copy()
    test_df = df[df['date'] >= pd.to_datetime(test_start)].copy()

    print(f"Training: {len(train_df)} obs ({train_df['date'].min()} to {train_df['date'].max()})")
    print(f"Testing:  {len(test_df)} obs ({test_df['date'].min()} to {test_df['date'].max()})")

    return train_df, test_df


def plot_static_vs_rolling_comparison(test_data, static_forecasts, rolling_results,
                                      model_name='holt_winters', save_path=None):
    """
    Plot comparison of static vs rolling forecasts for a specific model.
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # Get forecasts
    static = static_forecasts.get(model_name)
    rolling = rolling_results.get(model_name)

    if static is None or rolling is None:
        print(f"Cannot plot comparison for {model_name}")
        return

    dates = test_data['date']
    actual = test_data['median_price']

    # Plot 1: Static forecast
    axes[0].plot(dates, actual, linewidth=3, color='black',
                marker='o', markersize=4, label='Actual', zorder=10)
    axes[0].plot(dates, static, linewidth=2, color='#E63946',
                linestyle='--', marker='s', markersize=3,
                label='Static Forecast (60-step ahead)', alpha=0.8)

    axes[0].set_xlabel('Date', fontsize=12)
    axes[0].set_ylabel('Median Price ($)', fontsize=12)
    axes[0].set_title('Static Forecast: All 60 months predicted at once',
                     fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

    # Plot 2: Rolling forecast
    axes[1].plot(dates, actual, linewidth=3, color='black',
                marker='o', markersize=4, label='Actual', zorder=10)
    axes[1].plot(dates, rolling['forecasts'], linewidth=2, color='#06D6A0',
                linestyle='--', marker='s', markersize=3,
                label='Rolling Forecast (1-step ahead, updated monthly)', alpha=0.8)

    axes[1].set_xlabel('Date', fontsize=12)
    axes[1].set_ylabel('Median Price ($)', fontsize=12)
    axes[1].set_title('Rolling Forecast: Model updated with each new observation',
                     fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

    plt.suptitle(f'Static vs Rolling Forecast Comparison - {model_name.replace("_", " ").title()}',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()


def plot_improvement_heatmap(comparison_df, save_path=None):
    """
    Plot heatmap showing improvement from static to rolling forecasts.
    """
    # Calculate improvements
    models = comparison_df['model'].unique()
    improvements = []

    for model in models:
        model_data = comparison_df[comparison_df['model'] == model]

        if len(model_data) < 2:
            continue

        static = model_data[model_data['method'] == 'Static (60-step)'].iloc[0]
        rolling = model_data[model_data['method'] == 'Rolling (1-step)'].iloc[0]

        improvements.append({
            'Model': model.replace('_', ' ').title(),
            'RMSE': (static['rmse'] - rolling['rmse']) / static['rmse'] * 100,
            'MAE': (static['mae'] - rolling['mae']) / static['mae'] * 100,
            'MAPE': (static['mape'] - rolling['mape']) / static['mape'] * 100
        })

    if not improvements:
        print("No improvement data available")
        return

    imp_df = pd.DataFrame(improvements)
    imp_df = imp_df.set_index('Model')

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(imp_df, annot=True, fmt='.1f', cmap='RdYlGn',
               center=0, cbar_kws={'label': 'Improvement (%)'},
               linewidths=1, ax=ax)

    ax.set_title('Forecast Improvement: Rolling vs Static Method\n(Positive = Better Performance)',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()


def run_improved_analysis(data_path='data/processed/ct_housing_monthly.csv',
                         output_dir='results'):
    """
    Run complete analysis comparing static and rolling forecasts.
    """
    print("="*80)
    print("IMPROVED CONNECTICUT HOUSING PRICE FORECASTING ANALYSIS")
    print("Comparing Static vs Rolling Forecast Methods")
    print("="*80)
    print()

    # Create output directories
    os.makedirs(f'{output_dir}/figures', exist_ok=True)
    os.makedirs(f'{output_dir}/tables', exist_ok=True)

    # 1. Load and split data
    df = load_data(data_path)
    train_df, test_df = split_train_test(df)

    train_series = train_df['median_price']
    test_series = test_df['median_price']
    forecast_steps = len(test_df)

    # 2. STATIC FORECASTS (Original Method)
    print("\n" + "="*80)
    print("METHOD 1: STATIC FORECASTS (60-step ahead)")
    print("="*80)
    print("Training once, forecasting all 60 months at once...")

    static_forecasts, static_models = fit_and_forecast_all_models(train_series, forecast_steps)

    # Evaluate static
    static_results = evaluate_multiple_models(test_series, static_forecasts)
    print("\nStatic Forecast Results:")
    print_comparison(static_results)

    # 3. ROLLING FORECASTS (Improved Method)
    print("\n" + "="*80)
    print("METHOD 2: ROLLING FORECASTS (1-step ahead, updated monthly)")
    print("="*80)
    print("Re-training model with each new observation...")

    rolling_results = rolling_forecast_all_models(
        train_series,
        test_series,
        horizon=1,
        update_frequency=1
    )

    # Evaluate rolling
    rolling_results_df = evaluate_rolling_forecasts(rolling_results)
    print("\nRolling Forecast Results:")
    print(rolling_results_df.to_string(index=False))

    # 4. COMPARISON
    comparison_df, _ = compare_static_vs_rolling(
        train_series,
        test_series,
        static_forecasts
    )

    analyze_forecast_improvement(comparison_df)

    # Save comparison results
    save_results(comparison_df, f'{output_dir}/tables/static_vs_rolling_comparison.csv')

    # 5. VISUALIZATIONS
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)

    # Original visualizations
    create_all_visualizations(train_df, test_df, static_forecasts, static_results,
                             output_dir=f'{output_dir}/figures')

    # Static vs Rolling comparison for best model
    best_model = static_results.iloc[0]['model']
    best_model_key = [k for k, v in static_models.items() if v.name == best_model][0]

    print(f"\n  8. Static vs Rolling comparison ({best_model})...")
    plot_static_vs_rolling_comparison(
        test_df,
        static_forecasts,
        rolling_results,
        model_name=best_model_key,
        save_path=f'{output_dir}/figures/08_static_vs_rolling_comparison.png'
    )

    # Improvement heatmap
    print(f"  9. Improvement heatmap...")
    plot_improvement_heatmap(
        comparison_df,
        save_path=f'{output_dir}/figures/09_improvement_heatmap.png'
    )

    # 6. SUMMARY
    print("\n" + "="*80)
    print("SUMMARY: WHY ROLLING FORECASTS ARE BETTER")
    print("="*80)

    print("""
Rolling forecasts (walk-forward validation) are more realistic because:

1. **Mirrors Real-World Practice**: In reality, you update your model as new
   data becomes available, not predict 5 years ahead at once.

2. **Reduces Error Accumulation**: 1-step ahead forecasts are inherently more
   accurate than 60-step ahead forecasts.

3. **Adapts to Changes**: The model learns from recent trends, economic shifts,
   and market changes as they occur.

4. **Better Evaluation**: Provides a fair assessment of how the model would
   actually perform in production.

For your research:
- Use STATIC forecasts to demonstrate model capabilities over long horizons
- Use ROLLING forecasts to show realistic forecast accuracy
- Report both methods to provide complete analysis
    """)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}/")
    print(f"  - Static forecast results: {output_dir}/tables/model_comparison.csv")
    print(f"  - Static vs Rolling comparison: {output_dir}/tables/static_vs_rolling_comparison.csv")
    print(f"  - Visualizations: {output_dir}/figures/")


def main():
    """Main entry point."""
    data_path = 'data/processed/ct_housing_monthly.csv'

    if not os.path.exists(data_path):
        print("ERROR: Processed data not found!")
        print(f"Expected file: {data_path}")
        print("\nPlease run preprocessing first:")
        print("  python src/data_preprocessing.py")
        sys.exit(1)

    run_improved_analysis(data_path)


if __name__ == '__main__':
    main()

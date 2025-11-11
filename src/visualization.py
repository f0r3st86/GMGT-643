"""
Visualization Module
Creates plots for time series analysis and forecasting results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


def plot_time_series(df, date_col='date', price_col='median_price',
                     title='Connecticut Median Housing Prices',
                     ylabel='Median Price ($)',
                     save_path=None):
    """
    Plot the complete time series.

    Parameters:
    -----------
    df : pd.DataFrame
        Data with date and price columns
    date_col : str
        Name of date column
    price_col : str
        Name of price column
    title : str
        Plot title
    ylabel : str
        Y-axis label
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(df[date_col], df[price_col], linewidth=2, color='#2E86AB')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()


def plot_train_test_split(train_df, test_df, date_col='date', price_col='median_price',
                          title='Train-Test Split', save_path=None):
    """
    Plot training and testing periods.

    Parameters:
    -----------
    train_df : pd.DataFrame
        Training data
    test_df : pd.DataFrame
        Testing data
    date_col : str
        Name of date column
    price_col : str
        Name of price column
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot training data
    ax.plot(train_df[date_col], train_df[price_col],
            linewidth=2, color='#2E86AB', label='Training Data (2001-2018)')

    # Plot testing data
    ax.plot(test_df[date_col], test_df[price_col],
            linewidth=2, color='#A23B72', label='Testing Data (2019-2023)')

    # Add vertical line at split point
    split_date = test_df[date_col].iloc[0]
    ax.axvline(x=split_date, color='red', linestyle='--', linewidth=2,
               label='Train-Test Split', alpha=0.7)

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Median Price ($)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()


def plot_forecasts(actual_dates, actual_values, forecast_dates, forecasts_dict,
                  train_dates=None, train_values=None,
                  title='Forecast Comparison', save_path=None):
    """
    Plot actual values vs. forecasts from multiple models.

    Parameters:
    -----------
    actual_dates : array-like
        Dates for actual values
    actual_values : array-like
        Actual observed values
    forecast_dates : array-like
        Dates for forecasts
    forecasts_dict : dict
        Dictionary of model_name -> forecast values
    train_dates : array-like, optional
        Training period dates
    train_values : array-like, optional
        Training period values
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(16, 8))

    # Plot training data if provided
    if train_dates is not None and train_values is not None:
        ax.plot(train_dates, train_values, linewidth=2, color='gray',
                alpha=0.5, label='Training Data')

    # Plot actual values
    ax.plot(actual_dates, actual_values, linewidth=3, color='black',
            marker='o', markersize=4, label='Actual', zorder=10)

    # Color palette for models
    colors = ['#E63946', '#F77F00', '#06D6A0', '#118AB2', '#073B4C']

    # Plot forecasts
    for i, (model_name, forecasts) in enumerate(forecasts_dict.items()):
        if forecasts is None:
            continue

        color = colors[i % len(colors)]
        ax.plot(forecast_dates, forecasts, linewidth=2, color=color,
                linestyle='--', marker='s', markersize=3,
                label=f'{model_name.replace("_", " ").title()}', alpha=0.8)

    # Add vertical line at forecast start
    ax.axvline(x=forecast_dates[0], color='red', linestyle=':', linewidth=2,
               alpha=0.5, label='Forecast Start')

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Median Price ($)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)

    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()


def plot_model_comparison(results_df, title='Model Performance Comparison',
                          save_path=None):
    """
    Plot bar charts comparing model performance.

    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame with model results (columns: model, rmse, mae, mape)
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Sort by RMSE
    results_df = results_df.sort_values('rmse')

    # Color palette
    colors = sns.color_palette('viridis', len(results_df))

    # Plot RMSE
    axes[0].barh(results_df['model'], results_df['rmse'], color=colors)
    axes[0].set_xlabel('RMSE ($)', fontsize=11)
    axes[0].set_title('Root Mean Squared Error', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')

    # Plot MAE
    axes[1].barh(results_df['model'], results_df['mae'], color=colors)
    axes[1].set_xlabel('MAE ($)', fontsize=11)
    axes[1].set_title('Mean Absolute Error', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')

    # Plot MAPE
    axes[2].barh(results_df['model'], results_df['mape'], color=colors)
    axes[2].set_xlabel('MAPE (%)', fontsize=11)
    axes[2].set_title('Mean Absolute Percentage Error', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='x')

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()


def plot_forecast_errors(actual_values, forecasts_dict, forecast_dates,
                        title='Forecast Errors Over Time', save_path=None):
    """
    Plot forecast errors over time for each model.

    Parameters:
    -----------
    actual_values : array-like
        Actual observed values
    forecasts_dict : dict
        Dictionary of model_name -> forecast values
    forecast_dates : array-like
        Dates for forecasts
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    colors = ['#E63946', '#F77F00', '#06D6A0', '#118AB2', '#073B4C']

    for i, (model_name, forecasts) in enumerate(forecasts_dict.items()):
        if forecasts is None:
            continue

        errors = np.array(actual_values) - np.array(forecasts)
        color = colors[i % len(colors)]

        ax.plot(forecast_dates, errors, linewidth=2, color=color,
                marker='o', markersize=3,
                label=f'{model_name.replace("_", " ").title()}', alpha=0.7)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Forecast Error ($)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)

    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()


def plot_crisis_analysis(dates, actual_values, forecasts, crisis_periods,
                        model_name='Model', title='Crisis Period Analysis',
                        save_path=None):
    """
    Plot forecasts with crisis periods highlighted.

    Parameters:
    -----------
    dates : array-like
        Dates
    actual_values : array-like
        Actual values
    forecasts : array-like
        Forecast values
    crisis_periods : list of tuples
        List of (start_date, end_date, name, color) tuples
    model_name : str
        Model name
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(16, 8))

    # Plot actual and forecast
    ax.plot(dates, actual_values, linewidth=3, color='black',
            marker='o', markersize=4, label='Actual', zorder=10)
    ax.plot(dates, forecasts, linewidth=2, color='#E63946',
            linestyle='--', marker='s', markersize=3,
            label=f'{model_name} Forecast', alpha=0.8)

    # Highlight crisis periods
    for start_date, end_date, name, color in crisis_periods:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        ax.axvspan(start, end, alpha=0.2, color=color, label=name)

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Median Price ($)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()


def plot_seasonal_pattern(df, date_col='date', price_col='median_price',
                         title='Seasonal Pattern in Housing Prices',
                         save_path=None):
    """
    Plot seasonal pattern by month.

    Parameters:
    -----------
    df : pd.DataFrame
        Data with date and price columns
    date_col : str
        Name of date column
    price_col : str
        Name of price column
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    """
    # Extract month from dates
    df_copy = df.copy()
    df_copy['month'] = pd.to_datetime(df_copy[date_col]).dt.month
    df_copy['month_name'] = pd.to_datetime(df_copy[date_col]).dt.strftime('%b')

    # Calculate average price by month
    monthly_avg = df_copy.groupby(['month', 'month_name'])[price_col].mean().reset_index()
    monthly_avg = monthly_avg.sort_values('month')

    fig, ax = plt.subplots(figsize=(12, 6))

    # Create bar plot
    colors = sns.color_palette('coolwarm', 12)
    ax.bar(monthly_avg['month_name'], monthly_avg[price_col], color=colors)

    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Average Median Price ($)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

    # Add value labels on bars
    for i, v in enumerate(monthly_avg[price_col]):
        ax.text(i, v, f'${v/1000:.0f}K', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()


def plot_residual_analysis(actual_values, forecasts, model_name='Model',
                          save_path=None):
    """
    Plot residual analysis for a model.

    Parameters:
    -----------
    actual_values : array-like
        Actual observed values
    forecasts : array-like
        Forecast values
    model_name : str
        Model name
    save_path : str, optional
        Path to save figure
    """
    residuals = np.array(actual_values) - np.array(forecasts)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Residuals over time
    axes[0].plot(residuals, marker='o', linestyle='-', color='#2E86AB')
    axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    axes[0].set_xlabel('Observation', fontsize=11)
    axes[0].set_ylabel('Residual ($)', fontsize=11)
    axes[0].set_title('Residuals Over Time', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Histogram of residuals
    axes[1].hist(residuals, bins=20, color='#2E86AB', edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    axes[1].set_xlabel('Residual ($)', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title('Distribution of Residuals', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[2])
    axes[2].set_title('Q-Q Plot', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(f'Residual Analysis - {model_name}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()


def create_all_visualizations(train_df, test_df, forecasts_dict, results_df,
                              output_dir='results/figures'):
    """
    Create all visualizations for the analysis.

    Parameters:
    -----------
    train_df : pd.DataFrame
        Training data
    test_df : pd.DataFrame
        Testing data
    forecasts_dict : dict
        Dictionary of model forecasts
    results_df : pd.DataFrame
        Model comparison results
    output_dir : str
        Output directory for figures
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Creating visualizations...")

    # Combine train and test
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    # 1. Full time series
    print("  1. Time series plot...")
    plot_time_series(full_df, save_path=f'{output_dir}/01_time_series.png')

    # 2. Train-test split
    print("  2. Train-test split plot...")
    plot_train_test_split(train_df, test_df, save_path=f'{output_dir}/02_train_test_split.png')

    # 3. Forecast comparison
    print("  3. Forecast comparison plot...")
    plot_forecasts(
        test_df['date'], test_df['median_price'],
        test_df['date'], forecasts_dict,
        train_df['date'], train_df['median_price'],
        save_path=f'{output_dir}/03_forecast_comparison.png'
    )

    # 4. Model comparison
    print("  4. Model performance comparison...")
    plot_model_comparison(results_df, save_path=f'{output_dir}/04_model_comparison.png')

    # 5. Forecast errors
    print("  5. Forecast errors plot...")
    plot_forecast_errors(
        test_df['median_price'], forecasts_dict, test_df['date'],
        save_path=f'{output_dir}/05_forecast_errors.png'
    )

    # 6. Seasonal pattern
    print("  6. Seasonal pattern plot...")
    plot_seasonal_pattern(full_df, save_path=f'{output_dir}/06_seasonal_pattern.png')

    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == '__main__':
    print("Visualization Module")
    print("This module contains functions for creating plots and visualizations.")

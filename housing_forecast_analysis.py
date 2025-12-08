"""
Connecticut Housing Price Forecasting Analysis
===============================================
MBA Data Analytics Project

This script performs a comprehensive analysis of forecasting methods
for Connecticut residential housing prices with full visualization.

Author: [Your Name]
Date: December 2025
"""

# =============================================================================
# SECTION 1: IMPORTS AND SETUP
# =============================================================================

import pandas as pd
import numpy as np
import requests
import os
import warnings
from datetime import datetime

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Plot settings
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
pd.set_option('display.max_columns', None)

# Statistical models
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import Ridge

# For Auto-SARIMAX
from pmdarima import auto_arima

# For Prophet (optional)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Note: Prophet not installed. Skipping Prophet model.")


# =============================================================================
# SECTION 2: CONFIGURATION
# =============================================================================

# Set to True to pull fresh data from APIs, False to use cached data
FORCE_REFRESH = False

# Set to True to save charts to files, False to display only
SAVE_CHARTS = True
CHART_DIR = 'charts'

# Cache directory and file paths
CACHE_DIR = 'data/raw'
HOUSING_CACHE_FILE = os.path.join(CACHE_DIR, 'ct_housing_raw.csv')
ECON_CACHE_FILE = os.path.join(CACHE_DIR, 'economic_indicators.csv')

# API Configuration
CT_DATA_PORTAL = 'data.ct.gov'
CT_DATASET_ID = '5mzw-sjtu'

# FRED API key (get free at fred.stlouisfed.org)
FRED_API_KEY = 'c550ff1c5cd3f18f5ef725059a054d88'

# FRED series IDs
FRED_SERIES = {
    'mortgage_rate': 'MORTGAGE30US',
    'building_permits': 'CTBPPRIVSA',
    'unemployment_rate': 'CTURN',
    'cpi': 'CPIAUCSL',
    'population': 'CTPOP',
    'per_capita_income': 'CTPCPI'
}


# =============================================================================
# SECTION 3: DATA FETCHING FUNCTIONS
# =============================================================================

def fetch_housing_data(use_cache=True):
    """Fetch CT real estate data from Open Data Portal."""
    if use_cache and not FORCE_REFRESH and os.path.exists(HOUSING_CACHE_FILE):
        print(f"  Loading housing data from cache...")
        df = pd.read_csv(HOUSING_CACHE_FILE)
        print(f"  ✓ Loaded {len(df):,} records")
        return df

    print("  Fetching from Connecticut Open Data Portal...")
    base_url = f"https://{CT_DATA_PORTAL}/resource/{CT_DATASET_ID}.json"
    all_records = []
    offset = 0
    limit = 50000

    while True:
        params = {'$limit': limit, '$offset': offset, '$order': 'daterecorded DESC'}
        try:
            response = requests.get(base_url, params=params, timeout=120)
            response.raise_for_status()
            data = response.json()
            if not data:
                break
            all_records.extend(data)
            offset += limit
            print(f"    Fetched {len(all_records):,} records...")
            if offset >= 1500000:
                break
        except Exception as e:
            print(f"  ✗ Error: {e}")
            break

    df = pd.DataFrame(all_records)
    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_csv(HOUSING_CACHE_FILE, index=False)
    print(f"  ✓ Saved {len(df):,} records")
    return df


def fetch_fred_series(series_id, start_date='2000-01-01'):
    """Fetch a single series from FRED."""
    base_url = 'https://api.stlouisfed.org/fred/series/observations'
    params = {
        'series_id': series_id,
        'observation_start': start_date,
        'file_type': 'json',
        'api_key': FRED_API_KEY if FRED_API_KEY else 'DEMO_KEY'
    }
    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        if 'observations' not in data:
            return pd.DataFrame()
        df = pd.DataFrame(data['observations'])
        df['date'] = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        return df[['date', 'value']].dropna()
    except Exception as e:
        print(f"    Warning: {series_id} - {e}")
        return pd.DataFrame()


def fetch_economic_data(use_cache=True):
    """Fetch economic indicators from FRED."""
    if use_cache and not FORCE_REFRESH and os.path.exists(ECON_CACHE_FILE):
        print(f"  Loading economic data from cache...")
        df = pd.read_csv(ECON_CACHE_FILE)
        df['date'] = pd.to_datetime(df['date'])
        print(f"  ✓ Loaded {len(df):,} records")
        return df

    print("  Fetching from FRED API...")
    series_data = {}
    for name, series_id in FRED_SERIES.items():
        print(f"    Fetching {name}...")
        df = fetch_fred_series(series_id)
        if not df.empty:
            series_data[name] = df.set_index('date')['value']

    combined = pd.DataFrame(series_data).reset_index()
    combined.columns = ['date'] + list(series_data.keys())
    combined = combined.set_index('date').resample('MS').last().ffill().bfill().reset_index()

    os.makedirs(CACHE_DIR, exist_ok=True)
    combined.to_csv(ECON_CACHE_FILE, index=False)
    print(f"  ✓ Saved {len(combined):,} records")
    return combined


# =============================================================================
# SECTION 4: VISUALIZATION FUNCTIONS
# =============================================================================

def save_or_show(filename):
    """Save chart to file or display it."""
    if SAVE_CHARTS:
        os.makedirs(CHART_DIR, exist_ok=True)
        filepath = os.path.join(CHART_DIR, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"    Saved: {filepath}")
    plt.close()


def plot_price_trend(df):
    """Plot housing price trend over time."""
    print("\n  Creating price trend chart...")
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(df['date'], df['median_price'], color='#2E86AB', linewidth=2)
    ax.fill_between(df['date'], df['median_price'], alpha=0.3, color='#2E86AB')

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Median Sale Price ($)', fontsize=12)
    ax.set_title('Connecticut Median Housing Prices (2001-2024)', fontsize=14, fontweight='bold')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

    # Add COVID marker
    ax.axvline(x=pd.Timestamp('2020-03-01'), color='red', linestyle='--', alpha=0.7, label='COVID-19')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_or_show('01_price_trend.png')


def plot_price_vs_mortgage(df):
    """Plot housing prices vs mortgage rates (dual axis)."""
    print("  Creating price vs mortgage rate chart...")
    fig, ax1 = plt.subplots(figsize=(14, 6))

    color1 = '#2E86AB'
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Median Housing Price ($)', color=color1, fontsize=12)
    ax1.plot(df['date'], df['median_price'], color=color1, linewidth=2, label='Housing Price')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

    ax2 = ax1.twinx()
    color2 = '#E94F37'
    ax2.set_ylabel('30-Year Mortgage Rate (%)', color=color2, fontsize=12)
    ax2.plot(df['date'], df['mortgage_rate'], color=color2, linestyle='--', linewidth=2, label='Mortgage Rate')
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title('CT Housing Prices vs. Mortgage Rates', fontsize=14, fontweight='bold')

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    save_or_show('02_price_vs_mortgage.png')


def plot_correlation_matrix(df, features, title_suffix=''):
    """Plot correlation heatmap."""
    print("  Creating correlation matrix...")
    cols = ['median_price'] + [f for f in features if f in df.columns]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    sns.heatmap(corr, mask=mask, annot=True, cmap='RdBu_r', center=0,
                fmt='.2f', linewidths=0.5, ax=ax, annot_kws={'size': 9},
                cbar_kws={'shrink': 0.8})

    ax.set_title(f'Correlation Matrix: Housing Prices & Economic Indicators{title_suffix}',
                 fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    save_or_show('03_correlation_matrix.png')


def plot_seasonal_pattern(df):
    """Plot seasonal patterns in housing prices."""
    print("  Creating seasonal pattern chart...")
    df_temp = df.copy()
    df_temp['month'] = df_temp['date'].dt.month
    df_temp['year'] = df_temp['date'].dt.year

    monthly_avg = df_temp.groupby('month')['median_price'].mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(monthly_avg.index, monthly_avg.values, color='#2E86AB', edgecolor='white')

    # Highlight peak months
    max_month = monthly_avg.idxmax()
    bars[max_month - 1].set_color('#E94F37')

    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Average Median Price ($)', fontsize=12)
    ax.set_title('Seasonal Pattern: Average Housing Prices by Month', fontsize=14, fontweight='bold')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_or_show('04_seasonal_pattern.png')


def plot_model_comparison(results):
    """Plot model performance comparison bar chart."""
    print("  Creating model comparison chart...")

    models = list(results.keys())
    mapes = [results[m]['metrics']['MAPE'] for m in models]

    # Sort by MAPE
    sorted_idx = np.argsort(mapes)
    models = [models[i] for i in sorted_idx]
    mapes = [mapes[i] for i in sorted_idx]

    colors = ['#2E86AB' if i > 0 else '#E94F37' for i in range(len(models))]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(models, mapes, color=colors, edgecolor='white')

    # Add value labels
    for bar, mape in zip(bars, mapes):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{mape:.1f}%', va='center', fontsize=10)

    ax.set_xlabel('Mean Absolute Percentage Error (MAPE %)', fontsize=12)
    ax.set_title('Model Performance Comparison (Lower is Better)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, max(mapes) * 1.15)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    save_or_show('05_model_comparison.png')


def plot_forecast_vs_actual(test_y, results, title=''):
    """Plot forecasts vs actual values."""
    print("  Creating forecast vs actual chart...")

    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot actual values
    ax.plot(test_y.index, test_y.values, color='black', linewidth=2.5,
            label='Actual', marker='o', markersize=3)

    # Color palette for models
    colors = {'Holt-Winters': '#2E86AB', 'Auto-SARIMAX': '#E94F37',
              'Ridge Regression': '#F6BD60', 'Prophet': '#84A98C',
              'Ensemble': '#9B5DE5', 'Naive': '#888888'}

    for name, data in results.items():
        pred = data['predictions'][:len(test_y)]
        color = colors.get(name, '#888888')
        ax.plot(test_y.index[:len(pred)], pred, linestyle='--',
                linewidth=1.5, label=f"{name} ({data['metrics']['MAPE']:.1f}%)",
                color=color, alpha=0.8)

    ax.axvline(x=pd.Timestamp('2020-03-01'), color='red', linestyle=':',
               alpha=0.5, label='COVID-19 Start')

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Median Sale Price ($)', fontsize=12)
    ax.set_title(f'Forecast vs Actual Housing Prices (2019-2024){title}',
                 fontsize=14, fontweight='bold')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_or_show('06_forecast_vs_actual.png')


def plot_residuals(test_y, results, model_name='Holt-Winters'):
    """Plot residuals for the best model."""
    print(f"  Creating residuals chart for {model_name}...")

    if model_name not in results:
        return

    pred = results[model_name]['predictions'][:len(test_y)]
    residuals = test_y.values[:len(pred)] - pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residuals over time
    axes[0].plot(test_y.index[:len(pred)], residuals, color='#2E86AB', linewidth=1)
    axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[0].fill_between(test_y.index[:len(pred)], residuals, alpha=0.3)
    axes[0].set_xlabel('Date', fontsize=11)
    axes[0].set_ylabel('Residual ($)', fontsize=11)
    axes[0].set_title(f'{model_name} Residuals Over Time', fontsize=12, fontweight='bold')
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    axes[0].grid(True, alpha=0.3)

    # Residuals histogram
    axes[1].hist(residuals, bins=20, color='#2E86AB', edgecolor='white', alpha=0.7)
    axes[1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
    axes[1].set_xlabel('Residual ($)', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title('Residual Distribution', fontsize=12, fontweight='bold')
    axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

    plt.tight_layout()
    save_or_show('07_residuals.png')


def plot_lagged_comparison(current_mape, lagged_mape):
    """Plot current vs lagged indicator performance."""
    print("  Creating lagged indicator comparison chart...")

    fig, ax = plt.subplots(figsize=(8, 6))

    categories = ['Current\nIndicators', '12-Month\nLagged']
    values = [current_mape, lagged_mape]
    colors = ['#E94F37', '#2E86AB']

    bars = ax.bar(categories, values, color=colors, edgecolor='white', width=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', fontsize=12, fontweight='bold')

    ax.set_ylabel('MAPE (%)', fontsize=12)
    ax.set_title('Impact of Using Lagged Economic Indicators\n(Ridge Regression)',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(values) * 1.2)
    ax.grid(True, alpha=0.3, axis='y')

    # Add improvement annotation
    improvement = current_mape - lagged_mape
    ax.annotate(f'↓ {improvement:.1f}% improvement',
                xy=(1, lagged_mape), xytext=(1.3, (current_mape + lagged_mape)/2),
                fontsize=11, color='green', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='green'))

    plt.tight_layout()
    save_or_show('08_lagged_comparison.png')


def print_data_summary_table(df):
    """Print formatted data summary table."""
    print("\n" + "=" * 70)
    print("DATA SUMMARY")
    print("=" * 70)

    print(f"""
┌────────────────────────────────┬────────────────────────────┐
│ Metric                         │ Value                      │
├────────────────────────────────┼────────────────────────────┤
│ Total Monthly Observations     │ {len(df):>26,} │
│ Date Range                     │ {df['date'].min().strftime('%Y-%m')} to {df['date'].max().strftime('%Y-%m'):>15} │
│ Mean Median Price              │ ${df['median_price'].mean():>24,.0f} │
│ Min Median Price               │ ${df['median_price'].min():>24,.0f} │
│ Max Median Price               │ ${df['median_price'].max():>24,.0f} │
│ Standard Deviation             │ ${df['median_price'].std():>24,.0f} │
└────────────────────────────────┴────────────────────────────┘
""")


def print_model_results_table(results):
    """Print formatted model results table."""
    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE RESULTS")
    print("=" * 70)

    sorted_results = sorted(results.items(), key=lambda x: x[1]['metrics']['MAPE'])

    print("""
┌──────────────────────┬──────────────┬──────────────┬──────────┬─────────┐
│ Model                │ RMSE ($)     │ MAE ($)      │ MAPE (%) │ R²      │
├──────────────────────┼──────────────┼──────────────┼──────────┼─────────┤""")

    for name, data in sorted_results:
        m = data['metrics']
        print(f"│ {name:<20} │ {m['RMSE']:>12,.0f} │ {m['MAE']:>12,.0f} │ {m['MAPE']:>8.2f} │ {m['R2']:>7.3f} │")

    print("└──────────────────────┴──────────────┴──────────────┴──────────┴─────────┘")

    best = sorted_results[0]
    print(f"\n🏆 Best Model: {best[0]} (MAPE: {best[1]['metrics']['MAPE']:.2f}%)")


# =============================================================================
# SECTION 5: HELPER FUNCTIONS
# =============================================================================

def calculate_metrics(actual, predicted):
    """Calculate RMSE, MAE, MAPE, R²."""
    actual = np.array(actual)
    predicted = np.array(predicted)
    min_len = min(len(actual), len(predicted))
    actual, predicted = actual[:min_len], predicted[:min_len]

    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mae = np.mean(np.abs(actual - predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2}


# =============================================================================
# SECTION 6: MAIN ANALYSIS
# =============================================================================

def main():
    """Main analysis function."""

    print("=" * 70)
    print("CONNECTICUT HOUSING PRICE FORECASTING ANALYSIS")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Step 1: Load Data
    # -------------------------------------------------------------------------
    print("\n[1/7] LOADING DATA...")
    housing_df = fetch_housing_data(use_cache=True)
    econ_df = fetch_economic_data(use_cache=True)

    # -------------------------------------------------------------------------
    # Step 2: Preprocess Data
    # -------------------------------------------------------------------------
    print("\n[2/7] PREPROCESSING DATA...")

    housing_df['date'] = pd.to_datetime(housing_df['daterecorded'], errors='coerce')
    housing_df = housing_df.dropna(subset=['date'])
    housing_df = housing_df[housing_df['saleamount'] > 10000]
    housing_df = housing_df[
        housing_df['propertytype'].str.lower().str.contains(
            'residential|condo|single|family', na=False
        )
    ]
    housing_df = housing_df[
        (housing_df['date'] >= '2001-01-01') &
        (housing_df['date'] <= '2024-12-31')
    ]

    monthly = housing_df.groupby(
        housing_df['date'].dt.to_period('M')
    )['saleamount'].median().reset_index()
    monthly.columns = ['date', 'median_price']
    monthly['date'] = monthly['date'].dt.to_timestamp()

    econ_df['date'] = pd.to_datetime(econ_df['date'])
    final_df = monthly.merge(econ_df, on='date', how='inner')
    final_df = final_df.sort_values('date').reset_index(drop=True)

    print(f"  ✓ Final dataset: {len(final_df)} months")

    # Print data summary
    print_data_summary_table(final_df)

    # -------------------------------------------------------------------------
    # Step 3: Create Visualizations - Data Exploration
    # -------------------------------------------------------------------------
    print("\n[3/7] CREATING DATA EXPLORATION CHARTS...")

    plot_price_trend(final_df)
    plot_price_vs_mortgage(final_df)

    features = ['mortgage_rate', 'unemployment_rate', 'cpi', 'population']
    available_features = [f for f in features if f in final_df.columns]
    plot_correlation_matrix(final_df, available_features)
    plot_seasonal_pattern(final_df)

    # -------------------------------------------------------------------------
    # Step 4: Train/Test Split
    # -------------------------------------------------------------------------
    print("\n[4/7] SPLITTING DATA...")

    SPLIT_DATE = '2019-01-01'
    TARGET = 'median_price'

    train_df = final_df[final_df['date'] < SPLIT_DATE].copy().set_index('date')
    test_df = final_df[final_df['date'] >= SPLIT_DATE].copy().set_index('date')

    train_y, test_y = train_df[TARGET], test_df[TARGET]
    train_X, test_X = train_df[available_features], test_df[available_features]
    steps = len(test_y)

    print(f"  Training: {len(train_df)} months | Testing: {len(test_df)} months")

    # -------------------------------------------------------------------------
    # Step 5: Train Models
    # -------------------------------------------------------------------------
    print("\n[5/7] TRAINING MODELS...")

    results = {}

    # Naive
    print("  Training Naive...")
    naive_pred = np.full(steps, train_y.iloc[-1])
    results['Naive'] = {'predictions': naive_pred, 'metrics': calculate_metrics(test_y.values, naive_pred)}

    # Holt-Winters
    print("  Training Holt-Winters...")
    try:
        hw = ExponentialSmoothing(train_y, seasonal_periods=12, trend='add', seasonal='add')
        hw_fit = hw.fit(optimized=True)
        hw_pred = hw_fit.forecast(steps)
        results['Holt-Winters'] = {'predictions': hw_pred.values, 'metrics': calculate_metrics(test_y.values, hw_pred.values)}
    except Exception as e:
        print(f"    ✗ Failed: {e}")

    # Ridge
    print("  Training Ridge Regression...")
    try:
        ridge = Ridge(alpha=100.0)
        ridge.fit(train_X, train_y)
        ridge_pred = ridge.predict(test_X)
        results['Ridge Regression'] = {'predictions': ridge_pred, 'metrics': calculate_metrics(test_y.values, ridge_pred)}
    except Exception as e:
        print(f"    ✗ Failed: {e}")

    # Prophet
    if PROPHET_AVAILABLE:
        print("  Training Prophet...")
        try:
            import logging
            logging.getLogger('prophet').setLevel(logging.WARNING)
            logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
            prophet_df = pd.DataFrame({'ds': train_y.index, 'y': train_y.values})
            prophet = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                            daily_seasonality=False, seasonality_mode='multiplicative')
            prophet.fit(prophet_df)
            future = prophet.make_future_dataframe(periods=steps, freq='MS')
            forecast = prophet.predict(future)
            prophet_pred = forecast['yhat'].iloc[-steps:].values
            results['Prophet'] = {'predictions': prophet_pred, 'metrics': calculate_metrics(test_y.values, prophet_pred)}
        except Exception as e:
            print(f"    ✗ Failed: {e}")

    # Ensemble
    print("  Creating Ensemble...")
    if 'Holt-Winters' in results and 'Ridge Regression' in results:
        ensemble_pred = (results['Holt-Winters']['predictions'] + results['Ridge Regression']['predictions']) / 2
        results['Ensemble'] = {'predictions': ensemble_pred, 'metrics': calculate_metrics(test_y.values, ensemble_pred)}

    # Auto-SARIMAX
    print("  Training Auto-SARIMAX...")
    try:
        auto = auto_arima(train_y, exogenous=train_X, m=12, seasonal=True,
                         stepwise=True, suppress_warnings=True, error_action='ignore')
        sarimax_pred = auto.predict(n_periods=steps, exogenous=test_X)
        results['Auto-SARIMAX'] = {'predictions': sarimax_pred, 'metrics': calculate_metrics(test_y.values, sarimax_pred)}
    except Exception as e:
        print(f"    ✗ Failed: {e}")

    # -------------------------------------------------------------------------
    # Step 6: Results & Visualizations
    # -------------------------------------------------------------------------
    print("\n[6/7] CREATING RESULTS CHARTS...")

    print_model_results_table(results)
    plot_model_comparison(results)
    plot_forecast_vs_actual(test_y, results)
    plot_residuals(test_y, results, 'Holt-Winters')

    # -------------------------------------------------------------------------
    # Step 7: Lagged Analysis
    # -------------------------------------------------------------------------
    print("\n[7/7] LAGGED INDICATOR ANALYSIS...")

    lagged_df = final_df.copy()
    for col in available_features:
        lagged_df[f'{col}_lag12'] = lagged_df[col].shift(12)

    lagged_df = lagged_df.dropna().set_index('date')
    train_lag = lagged_df[lagged_df.index < SPLIT_DATE]
    test_lag = lagged_df[lagged_df.index >= SPLIT_DATE]
    lag_features = [f'{c}_lag12' for c in available_features]

    ridge_lag = Ridge(alpha=100.0)
    ridge_lag.fit(train_lag[lag_features], train_lag[TARGET])
    lag_pred = ridge_lag.predict(test_lag[lag_features])
    lag_metrics = calculate_metrics(test_lag[TARGET].values, lag_pred)

    current_mape = results['Ridge Regression']['metrics']['MAPE']
    lagged_mape = lag_metrics['MAPE']

    print(f"\n  Current indicators MAPE:  {current_mape:.2f}%")
    print(f"  Lagged indicators MAPE:   {lagged_mape:.2f}%")
    print(f"  Improvement:              {current_mape - lagged_mape:.2f} percentage points")

    plot_lagged_comparison(current_mape, lagged_mape)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    if SAVE_CHARTS:
        print(f"\nCharts saved to: {CHART_DIR}/")
        print("  01_price_trend.png")
        print("  02_price_vs_mortgage.png")
        print("  03_correlation_matrix.png")
        print("  04_seasonal_pattern.png")
        print("  05_model_comparison.png")
        print("  06_forecast_vs_actual.png")
        print("  07_residuals.png")
        print("  08_lagged_comparison.png")

    return results, final_df


# =============================================================================
# RUN
# =============================================================================

if __name__ == '__main__':
    results, data = main()

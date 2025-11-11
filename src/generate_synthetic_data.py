"""
Generate Synthetic Connecticut Housing Data
Creates realistic monthly median housing prices based on known CT market trends
"""

import pandas as pd
import numpy as np
from datetime import datetime

def generate_ct_housing_data():
    """
    Generate synthetic Connecticut housing price data from 2001-2023.

    Based on actual Connecticut housing market trends:
    - 2001-2006: Steady growth from ~$200k to ~$280k
    - 2007-2012: Financial crisis decline to ~$240k
    - 2013-2019: Recovery and growth to ~$275k
    - 2020-2021: COVID spike to ~$310k
    - 2022-2023: Stabilization around $300k

    Includes:
    - Seasonal patterns (spring/summer peaks)
    - Realistic volatility
    - Trend components
    """

    # Create date range
    dates = pd.date_range(start='2001-01-01', end='2023-12-31', freq='MS')
    n = len(dates)

    # Time index for trend
    t = np.arange(n)

    # Base trend with different growth rates for different periods
    trend = np.zeros(n)

    # 2001-2006: Strong growth (indices 0-72)
    growth_period_1 = min(72, n)
    trend[:growth_period_1] = 200000 + (t[:growth_period_1] / 72) * 80000

    # 2007-2012: Decline/stagnation (indices 72-144)
    if n > 72:
        decline_period = min(144, n) - 72
        trend[72:min(144, n)] = 280000 - (t[:decline_period] / 72) * 40000

    # 2013-2019: Recovery (indices 144-228)
    if n > 144:
        recovery_period = min(228, n) - 144
        trend[144:min(228, n)] = 240000 + (t[:recovery_period] / 84) * 35000

    # 2020-2021: COVID spike (indices 228-252)
    if n > 228:
        covid_period = min(252, n) - 228
        trend[228:min(252, n)] = 275000 + (t[:covid_period] / 24) * 35000

    # 2022-2023: Stabilization (indices 252-276)
    if n > 252:
        stable_period = n - 252
        trend[252:] = 310000 - (t[:stable_period] / 24) * 10000

    # Seasonal pattern (12-month cycle)
    # Higher prices in spring/summer (April-August)
    month = np.array([d.month for d in dates])
    seasonal = np.zeros(n)

    # Peak in June (month 6), trough in January (month 1)
    for i in range(n):
        if month[i] in [4, 5, 6, 7, 8]:  # Spring/Summer
            seasonal[i] = 8000 * np.sin(2 * np.pi * (month[i] - 1) / 12)
        else:  # Fall/Winter
            seasonal[i] = -5000 * np.cos(2 * np.pi * (month[i] - 1) / 12)

    # Add realistic noise
    np.random.seed(42)
    noise = np.random.normal(0, 3000, n)  # $3k standard deviation

    # Add some autocorrelation to noise (housing prices don't jump randomly)
    for i in range(1, n):
        noise[i] = 0.7 * noise[i-1] + 0.3 * noise[i]

    # Combine components
    median_price = trend + seasonal + noise

    # Create transaction count (varies seasonally and over time)
    base_count = 800
    count_seasonal = 200 * np.sin(2 * np.pi * (month - 1) / 12)
    count_trend = 100 * (t / n)  # Slight increase over time
    count_noise = np.random.normal(0, 50, n)
    count = (base_count + count_seasonal + count_trend + count_noise).astype(int)
    count = np.maximum(count, 100)  # Minimum 100 transactions

    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'median_price': median_price,
        'mean_price': median_price * 1.05,  # Mean slightly higher than median
        'count': count,
        'std_price': 40000 + noise * 2  # Price standard deviation
    })

    return df


if __name__ == '__main__':
    print("Generating synthetic Connecticut housing data...")
    print("(Based on actual CT market trends from 2001-2023)")

    df = generate_ct_housing_data()

    # Save data
    import os
    os.makedirs('data/processed', exist_ok=True)
    output_path = 'data/processed/ct_housing_monthly.csv'
    df.to_csv(output_path, index=False)

    print(f"\n✓ Generated {len(df)} monthly observations")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Price range: ${df['median_price'].min():,.0f} to ${df['median_price'].max():,.0f}")
    print(f"\nData saved to: {output_path}")

    print("\nFirst 12 months:")
    print(df.head(12)[['date', 'median_price', 'count']])

    print("\nLast 12 months:")
    print(df.tail(12)[['date', 'median_price', 'count']])

    print("\nSummary statistics:")
    print(df[['median_price', 'count']].describe())

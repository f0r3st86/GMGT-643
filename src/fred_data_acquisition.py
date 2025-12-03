"""
FRED Data Acquisition Module
Downloads economic indicators from Federal Reserve Economic Data (FRED)
to enhance housing price forecasting models.

Relevant indicators for housing price forecasting:
- Mortgage rates (financing costs)
- Unemployment rates (economic health)
- Inflation measures (purchasing power)
- Interest rates (monetary policy)
- Housing market indicators (supply/demand)
- Consumer sentiment (buyer confidence)
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple


class FREDDataFetcher:
    """
    Fetches economic data from FRED (Federal Reserve Economic Data).

    Provides macroeconomic indicators that can improve housing price
    forecasting accuracy by capturing economic conditions.

    Note: Requires a free FRED API key from https://fred.stlouisfed.org/docs/api/api_key.html
    """

    # Economic indicators relevant for housing price forecasting
    HOUSING_INDICATORS = {
        # Mortgage and Interest Rates
        'MORTGAGE30US': {
            'name': '30-Year Fixed Mortgage Rate',
            'frequency': 'weekly',
            'description': 'Average 30-year fixed rate mortgage rate',
            'category': 'financing'
        },
        'MORTGAGE15US': {
            'name': '15-Year Fixed Mortgage Rate',
            'frequency': 'weekly',
            'description': 'Average 15-year fixed rate mortgage rate',
            'category': 'financing'
        },
        'FEDFUNDS': {
            'name': 'Federal Funds Rate',
            'frequency': 'monthly',
            'description': 'Federal funds effective rate',
            'category': 'monetary_policy'
        },

        # Employment
        'UNRATE': {
            'name': 'National Unemployment Rate',
            'frequency': 'monthly',
            'description': 'Civilian unemployment rate',
            'category': 'employment'
        },
        'CTURN': {
            'name': 'Connecticut Unemployment Rate',
            'frequency': 'monthly',
            'description': 'Connecticut state unemployment rate',
            'category': 'employment'
        },

        # Inflation
        'CPIAUCSL': {
            'name': 'Consumer Price Index',
            'frequency': 'monthly',
            'description': 'CPI for all urban consumers',
            'category': 'inflation'
        },

        # Housing Market
        'HOUST': {
            'name': 'Housing Starts',
            'frequency': 'monthly',
            'description': 'New privately-owned housing units started',
            'category': 'housing_supply'
        },
        'PERMIT': {
            'name': 'Building Permits',
            'frequency': 'monthly',
            'description': 'New private housing units authorized by permits',
            'category': 'housing_supply'
        },
        'CSUSHPINSA': {
            'name': 'Case-Shiller Home Price Index',
            'frequency': 'monthly',
            'description': 'S&P/Case-Shiller U.S. National Home Price Index',
            'category': 'housing_prices'
        },

        # Consumer Confidence
        'UMCSENT': {
            'name': 'Consumer Sentiment',
            'frequency': 'monthly',
            'description': 'University of Michigan Consumer Sentiment Index',
            'category': 'sentiment'
        },

        # Income
        'DSPIC96': {
            'name': 'Real Disposable Personal Income',
            'frequency': 'monthly',
            'description': 'Real disposable personal income',
            'category': 'income'
        },
    }

    # Default series for housing analysis
    DEFAULT_SERIES = [
        'MORTGAGE30US',  # Primary mortgage rate
        'CTURN',         # CT-specific unemployment
        'UNRATE',        # National unemployment (backup)
        'CPIAUCSL',      # Inflation
        'FEDFUNDS',      # Fed policy
        'CSUSHPINSA',    # National housing prices
        'UMCSENT',       # Consumer confidence
    ]

    def __init__(self, api_key: Optional[str] = None, data_dir: str = 'data/raw'):
        """
        Initialize the FRED data fetcher.

        Parameters:
        -----------
        api_key : str, optional
            FRED API key. If None, will try to read from FRED_API_KEY
            environment variable or use fallback CSV download method.
        data_dir : str
            Directory to save downloaded data files
        """
        self.data_dir = data_dir
        self.api_key = api_key or os.environ.get('FRED_API_KEY')
        self.fred = None

        # Create data directory if needed
        os.makedirs(self.data_dir, exist_ok=True)

        # Initialize FRED API client if key is available
        if self.api_key:
            try:
                from fredapi import Fred
                self.fred = Fred(api_key=self.api_key)
                print("FRED API client initialized successfully")
            except ImportError:
                print("Warning: fredapi package not installed. Run: pip install fredapi")
            except Exception as e:
                print(f"Warning: Could not initialize FRED API: {e}")

    def fetch_series(self, series_id: str, start_date: str = '2000-01-01',
                     end_date: Optional[str] = None) -> pd.Series:
        """
        Fetch a single FRED data series.

        Parameters:
        -----------
        series_id : str
            FRED series identifier (e.g., 'MORTGAGE30US')
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str, optional
            End date. If None, fetches to present.

        Returns:
        --------
        pd.Series
            Time series data with datetime index
        """
        if self.fred is None:
            raise RuntimeError(
                "FRED API not initialized. Please provide an API key.\n"
                "Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html"
            )

        print(f"Fetching {series_id}...", end=" ")

        try:
            series = self.fred.get_series(
                series_id,
                observation_start=start_date,
                observation_end=end_date
            )
            print(f"OK ({len(series)} observations)")
            return series

        except Exception as e:
            print(f"FAILED: {e}")
            raise

    def fetch_multiple_series(self, series_ids: Optional[List[str]] = None,
                              start_date: str = '2000-01-01',
                              end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch multiple FRED series and combine into a DataFrame.

        Parameters:
        -----------
        series_ids : list of str, optional
            List of FRED series IDs. If None, uses DEFAULT_SERIES.
        start_date : str
            Start date for all series
        end_date : str, optional
            End date for all series

        Returns:
        --------
        pd.DataFrame
            DataFrame with each series as a column
        """
        if series_ids is None:
            series_ids = self.DEFAULT_SERIES

        print(f"\nFetching {len(series_ids)} FRED series...")
        print("-" * 50)

        data = {}
        failed = []

        for series_id in series_ids:
            try:
                data[series_id] = self.fetch_series(series_id, start_date, end_date)
            except Exception as e:
                failed.append((series_id, str(e)))
                continue

        if failed:
            print(f"\nWarning: Failed to fetch {len(failed)} series:")
            for series_id, error in failed:
                print(f"  - {series_id}: {error}")

        if not data:
            raise RuntimeError("No data series could be fetched")

        # Combine into DataFrame
        df = pd.DataFrame(data)
        df.index = pd.to_datetime(df.index)
        df.index.name = 'date'

        print(f"\nSuccessfully fetched {len(data)} series")
        print(f"Date range: {df.index.min()} to {df.index.max()}")

        return df

    def resample_to_monthly(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample all series to monthly frequency.

        Weekly data is averaged to monthly.
        Daily data is averaged to monthly.
        Monthly data is kept as-is.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with datetime index

        Returns:
        --------
        pd.DataFrame
            Monthly frequency DataFrame
        """
        print("\nResampling to monthly frequency...")

        # Resample to month-end frequency, taking the mean
        monthly_df = df.resample('ME').mean()

        # Forward fill any remaining gaps (up to 3 months)
        monthly_df = monthly_df.ffill(limit=3)

        print(f"Monthly observations: {len(monthly_df)}")

        return monthly_df

    def calculate_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate derived features from raw FRED data.

        Creates additional features that may be useful for forecasting:
        - Rate changes (month-over-month)
        - Year-over-year changes
        - Spread between rates

        Parameters:
        -----------
        df : pd.DataFrame
            Monthly FRED data

        Returns:
        --------
        pd.DataFrame
            DataFrame with additional derived features
        """
        print("\nCalculating derived features...")

        result = df.copy()

        # Mortgage rate changes
        if 'MORTGAGE30US' in df.columns:
            result['MORTGAGE_CHANGE_1M'] = df['MORTGAGE30US'].diff(1)
            result['MORTGAGE_CHANGE_12M'] = df['MORTGAGE30US'].diff(12)

        # Unemployment rate changes
        if 'CTURN' in df.columns:
            result['CT_UNEMP_CHANGE_12M'] = df['CTURN'].diff(12)
        elif 'UNRATE' in df.columns:
            result['UNEMP_CHANGE_12M'] = df['UNRATE'].diff(12)

        # Inflation (year-over-year CPI change)
        if 'CPIAUCSL' in df.columns:
            result['INFLATION_YOY'] = df['CPIAUCSL'].pct_change(12) * 100

        # Real mortgage rate (nominal rate - inflation)
        if 'MORTGAGE30US' in df.columns and 'INFLATION_YOY' in result.columns:
            result['REAL_MORTGAGE_RATE'] = df['MORTGAGE30US'] - result['INFLATION_YOY']

        # Fed funds spread (mortgage rate - fed funds)
        if 'MORTGAGE30US' in df.columns and 'FEDFUNDS' in df.columns:
            result['MORTGAGE_SPREAD'] = df['MORTGAGE30US'] - df['FEDFUNDS']

        # Consumer sentiment change
        if 'UMCSENT' in df.columns:
            result['SENTIMENT_CHANGE_12M'] = df['UMCSENT'].diff(12)

        # Housing price momentum (if Case-Shiller available)
        if 'CSUSHPINSA' in df.columns:
            result['HP_INDEX_YOY'] = df['CSUSHPINSA'].pct_change(12) * 100

        n_derived = len(result.columns) - len(df.columns)
        print(f"Created {n_derived} derived features")

        return result

    def save_data(self, df: pd.DataFrame, filename: str = 'fred_economic_data.csv') -> str:
        """
        Save FRED data to CSV file.

        Parameters:
        -----------
        df : pd.DataFrame
            Data to save
        filename : str
            Output filename

        Returns:
        --------
        str
            Path to saved file
        """
        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath)
        print(f"\nData saved to: {filepath}")
        return filepath

    def download_and_process(self, series_ids: Optional[List[str]] = None,
                             start_date: str = '2000-01-01',
                             end_date: Optional[str] = None,
                             include_derived: bool = True,
                             filename: str = 'fred_economic_data.csv') -> Tuple[pd.DataFrame, str]:
        """
        Complete pipeline: download, resample, calculate features, and save.

        Parameters:
        -----------
        series_ids : list of str, optional
            FRED series to download. If None, uses defaults.
        start_date : str
            Start date
        end_date : str, optional
            End date (defaults to present)
        include_derived : bool
            Whether to calculate derived features
        filename : str
            Output filename

        Returns:
        --------
        tuple
            (DataFrame, filepath)
        """
        # Fetch data
        df = self.fetch_multiple_series(series_ids, start_date, end_date)

        # Resample to monthly
        monthly_df = self.resample_to_monthly(df)

        # Calculate derived features
        if include_derived:
            monthly_df = self.calculate_derived_features(monthly_df)

        # Save to file
        filepath = self.save_data(monthly_df, filename)

        # Print summary
        self._print_summary(monthly_df)

        return monthly_df, filepath

    def _print_summary(self, df: pd.DataFrame):
        """Print summary statistics of the downloaded data."""
        print("\n" + "=" * 60)
        print("FRED DATA SUMMARY")
        print("=" * 60)
        print(f"\nShape: {df.shape}")
        print(f"Date range: {df.index.min().strftime('%Y-%m')} to {df.index.max().strftime('%Y-%m')}")
        print(f"\nColumns ({len(df.columns)}):")
        for col in df.columns:
            non_null = df[col].notna().sum()
            print(f"  - {col}: {non_null} observations")
        print("\nDescriptive Statistics:")
        print(df.describe().round(2).to_string())

    @classmethod
    def get_available_indicators(cls) -> pd.DataFrame:
        """
        Get a DataFrame of all available FRED indicators.

        Returns:
        --------
        pd.DataFrame
            DataFrame with indicator details
        """
        data = []
        for series_id, info in cls.HOUSING_INDICATORS.items():
            data.append({
                'series_id': series_id,
                'name': info['name'],
                'frequency': info['frequency'],
                'category': info['category'],
                'description': info['description']
            })
        return pd.DataFrame(data)


def create_fred_housing_dataset(api_key: str,
                                 start_date: str = '2000-01-01',
                                 output_dir: str = 'data/raw') -> pd.DataFrame:
    """
    Convenience function to create a complete FRED dataset for housing analysis.

    Parameters:
    -----------
    api_key : str
        FRED API key
    start_date : str
        Start date for data collection
    output_dir : str
        Directory to save output file

    Returns:
    --------
    pd.DataFrame
        Monthly economic indicators dataset
    """
    fetcher = FREDDataFetcher(api_key=api_key, data_dir=output_dir)

    # Use default indicators optimized for housing forecasting
    df, filepath = fetcher.download_and_process(
        start_date=start_date,
        include_derived=True,
        filename='fred_economic_data.csv'
    )

    return df


def main():
    """
    Main function demonstrating FRED data acquisition.
    """
    print("=" * 80)
    print("FRED Economic Data Acquisition for Housing Forecasting")
    print("=" * 80)

    # Check for API key
    api_key = os.environ.get('FRED_API_KEY')

    if not api_key:
        print("\nNo FRED API key found!")
        print("\nTo use this module, you need a free FRED API key:")
        print("1. Visit: https://fred.stlouisfed.org/docs/api/api_key.html")
        print("2. Create a free account and request an API key")
        print("3. Set it as an environment variable:")
        print("   export FRED_API_KEY='your-api-key-here'")
        print("\nAlternatively, pass it directly:")
        print("   fetcher = FREDDataFetcher(api_key='your-key')")

        print("\n\nAvailable FRED indicators for housing forecasting:")
        print("-" * 60)
        indicators = FREDDataFetcher.get_available_indicators()
        print(indicators.to_string(index=False))
        return

    # Download and process data
    fetcher = FREDDataFetcher(api_key=api_key)

    try:
        df, filepath = fetcher.download_and_process(
            start_date='2000-01-01',
            include_derived=True
        )
        print(f"\n{'=' * 80}")
        print(f"SUCCESS! FRED data saved to: {filepath}")
        print(f"{'=' * 80}")

    except Exception as e:
        print(f"\nError: {e}")
        print("\nPlease check your API key and internet connection.")


if __name__ == '__main__':
    main()

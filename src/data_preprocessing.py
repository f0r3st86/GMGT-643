"""
Data Preprocessing Module
Filters and processes Connecticut Real Estate Sales data
- Filters for residential properties
- Filters for arms-length transactions
- Creates monthly median price time series
- Merges with FRED economic indicators
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import Optional, List


class CTRealEstatePreprocessor:
    """
    Preprocesses Connecticut Real Estate Sales data for time series analysis.
    """

    def __init__(self, raw_data_path='data/raw/ct_real_estate_raw.csv',
                 output_dir='data/processed'):
        """
        Initialize the preprocessor.

        Parameters:
        -----------
        raw_data_path : str
            Path to raw data CSV file
        output_dir : str
            Directory to save processed data
        """
        self.raw_data_path = raw_data_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def load_raw_data(self):
        """
        Load raw data from CSV file.

        Returns:
        --------
        pd.DataFrame
            Raw data
        """
        print(f"Loading raw data from: {self.raw_data_path}")
        df = pd.read_csv(self.raw_data_path)
        print(f"Loaded {len(df):,} records")
        print(f"Columns: {list(df.columns)}")
        return df

    def clean_column_names(self, df):
        """
        Standardize column names.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe

        Returns:
        --------
        pd.DataFrame
            DataFrame with cleaned column names
        """
        # Convert to lowercase and replace spaces with underscores
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        return df

    def parse_dates(self, df, date_column='date_recorded'):
        """
        Parse date columns.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        date_column : str
            Name of the date column

        Returns:
        --------
        pd.DataFrame
            DataFrame with parsed dates
        """
        print(f"\nParsing dates from column: {date_column}")

        # Try common date column names if the specified one doesn't exist
        possible_date_columns = [date_column, 'date', 'sale_date', 'list_year']

        date_col = None
        for col in possible_date_columns:
            if col in df.columns:
                date_col = col
                break

        if date_col is None:
            print(f"Warning: Could not find date column. Available columns: {list(df.columns)}")
            return df

        print(f"Using date column: {date_col}")

        # Parse dates
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

        # Create year and month columns
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        df['year_month'] = df[date_col].dt.to_period('M')

        # Remove rows with invalid dates
        invalid_dates = df[date_col].isna().sum()
        if invalid_dates > 0:
            print(f"Removing {invalid_dates:,} rows with invalid dates")
            df = df[df[date_col].notna()].copy()

        return df

    def filter_residential(self, df, property_type_column='property_type'):
        """
        Filter for residential properties.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        property_type_column : str
            Name of property type column

        Returns:
        --------
        pd.DataFrame
            Filtered dataframe
        """
        print(f"\nFiltering for residential properties...")
        initial_count = len(df)

        # Try to find property type column
        possible_columns = [property_type_column, 'propertytype', 'property', 'residential_type']

        prop_col = None
        for col in possible_columns:
            if col in df.columns:
                prop_col = col
                break

        if prop_col is None:
            print(f"Warning: Could not find property type column")
            print(f"Available columns: {list(df.columns)}")
            print("Attempting to filter based on common residential keywords...")

            # If no property type column, try to infer from other columns
            # Look for residential indicators in any text column
            residential_keywords = ['single', 'family', 'residential', 'condo', 'two', 'three', 'four']

            # Create a mask for likely residential properties
            mask = pd.Series([False] * len(df), index=df.index)

            for col in df.select_dtypes(include=['object']).columns:
                for keyword in residential_keywords:
                    mask |= df[col].astype(str).str.lower().str.contains(keyword, na=False)

            df = df[mask].copy()

        else:
            # Filter based on property type column
            print(f"Property types in data: {df[prop_col].value_counts().head(10)}")

            # Define residential property types (adjust based on actual data)
            residential_types = [
                'Residential',
                'Single Family',
                'Condo',
                'Two Family',
                'Three Family',
                'Four Family',
                'Condominium'
            ]

            # Case-insensitive matching
            mask = df[prop_col].astype(str).str.lower().str.contains(
                '|'.join([t.lower() for t in residential_types]),
                na=False
            )
            df = df[mask].copy()

        final_count = len(df)
        print(f"Kept {final_count:,} residential properties ({final_count/initial_count*100:.1f}%)")

        return df

    def filter_arms_length(self, df, sale_amount_column='sale_amount'):
        """
        Filter for arms-length transactions.

        Arms-length transactions are genuine market transactions between unrelated parties.
        We filter out:
        - Very low prices (< $10,000) - likely non-market transactions
        - Very high prices (> 99th percentile) - outliers
        - Non-usable sales codes (gifts, foreclosures, etc.)

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        sale_amount_column : str
            Name of sale amount column

        Returns:
        --------
        pd.DataFrame
            Filtered dataframe
        """
        print(f"\nFiltering for arms-length transactions...")
        initial_count = len(df)

        # Find sale amount column
        possible_columns = [sale_amount_column, 'assessed_value', 'price', 'saleprice']

        amount_col = None
        for col in possible_columns:
            if col in df.columns:
                amount_col = col
                break

        if amount_col is None:
            print(f"Warning: Could not find sale amount column")
            print(f"Available columns: {list(df.columns)}")
            return df

        print(f"Using sale amount column: {amount_col}")

        # Convert to numeric
        df[amount_col] = pd.to_numeric(df[amount_col], errors='coerce')

        # Remove null prices
        df = df[df[amount_col].notna()].copy()

        # Filter for reasonable prices
        min_price = 10000  # $10,000 minimum
        percentile_99 = df[amount_col].quantile(0.99)
        max_price = percentile_99 * 1.5  # 1.5x the 99th percentile

        print(f"Price range: ${min_price:,.0f} to ${max_price:,.0f}")
        print(f"Median price before filter: ${df[amount_col].median():,.0f}")

        df = df[(df[amount_col] >= min_price) & (df[amount_col] <= max_price)].copy()

        # Filter based on sales code if available
        if 'sale_code' in df.columns or 'non_use_code' in df.columns:
            code_col = 'sale_code' if 'sale_code' in df.columns else 'non_use_code'
            print(f"\nFiltering based on {code_col}")

            # Keep only valid sales codes
            # Typically empty/null codes indicate normal market sales
            # Or codes like '0', '1', etc. indicate valid sales
            valid_codes = ['', ' ', '0', '1', 'A']

            df = df[df[code_col].fillna('').astype(str).str.strip().isin(valid_codes)].copy()

        final_count = len(df)
        print(f"Kept {final_count:,} arms-length transactions ({final_count/initial_count*100:.1f}%)")
        print(f"Median price after filter: ${df[amount_col].median():,.0f}")

        return df

    def create_monthly_series(self, df, date_column='date_recorded',
                              price_column='sale_amount',
                              start_date='2001-01-01',
                              end_date='2023-12-31'):
        """
        Create monthly median price time series.

        Parameters:
        -----------
        df : pd.DataFrame
            Filtered dataframe
        date_column : str
            Name of date column
        price_column : str
            Name of price column
        start_date : str
            Start date for time series
        end_date : str
            End date for time series

        Returns:
        --------
        pd.DataFrame
            Monthly time series with columns: date, median_price, count
        """
        print(f"\nCreating monthly median price time series...")

        # Find the actual date and price columns
        date_col = None
        for col in [date_column, 'date', 'sale_date', 'list_year']:
            if col in df.columns:
                date_col = col
                break

        price_col = None
        for col in [price_column, 'assessed_value', 'price', 'saleprice']:
            if col in df.columns:
                price_col = col
                break

        if date_col is None or price_col is None:
            raise ValueError(f"Could not find date or price columns")

        # Ensure year_month column exists
        if 'year_month' not in df.columns:
            df['year_month'] = pd.to_datetime(df[date_col]).dt.to_period('M')

        # Filter date range
        df = df[
            (df[date_col] >= pd.to_datetime(start_date)) &
            (df[date_col] <= pd.to_datetime(end_date))
        ].copy()

        print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")

        # Calculate monthly statistics
        monthly = df.groupby('year_month').agg({
            price_col: ['median', 'mean', 'count', 'std']
        }).reset_index()

        # Flatten column names
        monthly.columns = ['year_month', 'median_price', 'mean_price', 'count', 'std_price']

        # Convert period to timestamp
        monthly['date'] = monthly['year_month'].dt.to_timestamp()

        # Create complete date range (fill missing months)
        date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
        complete_df = pd.DataFrame({'date': date_range})

        # Merge with monthly data
        monthly = complete_df.merge(monthly[['date', 'median_price', 'mean_price', 'count', 'std_price']],
                                    on='date', how='left')

        # Interpolate missing values (if any)
        missing_count = monthly['median_price'].isna().sum()
        if missing_count > 0:
            print(f"Warning: {missing_count} months with missing data. Interpolating...")
            monthly['median_price'] = monthly['median_price'].interpolate(method='linear')
            monthly['mean_price'] = monthly['mean_price'].interpolate(method='linear')
            monthly['count'] = monthly['count'].fillna(0).astype(int)

        print(f"\nTime series created:")
        print(f"  Observations: {len(monthly)}")
        print(f"  Date range: {monthly['date'].min()} to {monthly['date'].max()}")
        print(f"  Median price range: ${monthly['median_price'].min():,.0f} to ${monthly['median_price'].max():,.0f}")
        print(f"  Average transactions per month: {monthly['count'].mean():.0f}")

        return monthly

    def save_processed_data(self, df, filename='ct_housing_monthly.csv'):
        """
        Save processed data to CSV.

        Parameters:
        -----------
        df : pd.DataFrame
            Processed data
        filename : str
            Output filename
        """
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"\nProcessed data saved to: {filepath}")
        return filepath

    def preprocess(self):
        """
        Run full preprocessing pipeline.

        Returns:
        --------
        pd.DataFrame
            Processed monthly time series
        """
        print("=" * 80)
        print("Data Preprocessing Pipeline")
        print("=" * 80)

        # Load raw data
        df = self.load_raw_data()

        # Clean column names
        df = self.clean_column_names(df)

        # Parse dates
        df = self.parse_dates(df)

        # Filter for residential properties
        df = self.filter_residential(df)

        # Filter for arms-length transactions
        df = self.filter_arms_length(df)

        # Create monthly time series
        monthly = self.create_monthly_series(df)

        # Save processed data
        self.save_processed_data(monthly)

        print("\n" + "=" * 80)
        print("Preprocessing Complete!")
        print("=" * 80)

        return monthly

    def merge_with_fred_data(self, housing_df: pd.DataFrame,
                              fred_path: str = 'data/raw/fred_economic_data.csv',
                              lag_periods: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Merge housing data with FRED economic indicators.

        Parameters:
        -----------
        housing_df : pd.DataFrame
            Monthly housing price data with 'date' column
        fred_path : str
            Path to FRED data CSV file
        lag_periods : list of int, optional
            List of lag periods to create for economic variables.
            E.g., [1, 3, 6, 12] creates 1-month, 3-month, 6-month, 12-month lags.
            If None, no lagged features are created.

        Returns:
        --------
        pd.DataFrame
            Merged dataset with housing and economic indicators
        """
        print("\nMerging housing data with FRED economic indicators...")

        if not os.path.exists(fred_path):
            print(f"Warning: FRED data file not found at {fred_path}")
            print("Please run fred_data_acquisition.py first to download FRED data")
            return housing_df

        # Load FRED data
        fred_df = pd.read_csv(fred_path, parse_dates=['date'], index_col='date')
        print(f"Loaded FRED data: {fred_df.shape[0]} observations, {fred_df.shape[1]} variables")

        # Ensure housing data has datetime index for merging
        housing_df = housing_df.copy()
        housing_df['date'] = pd.to_datetime(housing_df['date'])

        # Convert FRED to month-end to match housing data
        fred_df.index = fred_df.index + pd.offsets.MonthEnd(0) - pd.offsets.MonthEnd(1) + pd.offsets.MonthBegin(0)

        # Merge on date
        merged = housing_df.merge(
            fred_df.reset_index(),
            on='date',
            how='left'
        )

        # Create lagged features if requested
        if lag_periods:
            merged = self._create_lagged_features(merged, fred_df.columns.tolist(), lag_periods)

        # Report merge results
        n_matched = merged.dropna(subset=fred_df.columns[:1]).shape[0]
        print(f"Merged dataset: {merged.shape[0]} observations, {merged.shape[1]} columns")
        print(f"Observations with FRED data: {n_matched}")

        return merged

    def _create_lagged_features(self, df: pd.DataFrame,
                                 feature_cols: List[str],
                                 lag_periods: List[int]) -> pd.DataFrame:
        """
        Create lagged versions of economic indicator features.

        Lagged features are useful because economic conditions often affect
        housing prices with a delay.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with features to lag
        feature_cols : list of str
            Column names to create lags for
        lag_periods : list of int
            Number of periods to lag (e.g., [1, 3, 6, 12])

        Returns:
        --------
        pd.DataFrame
            DataFrame with additional lagged features
        """
        print(f"Creating lagged features for periods: {lag_periods}")

        df = df.copy()

        # Key variables to lag (not all variables need lags)
        key_vars = [
            'MORTGAGE30US', 'CTURN', 'UNRATE', 'FEDFUNDS',
            'INFLATION_YOY', 'UMCSENT', 'HP_INDEX_YOY'
        ]

        for col in feature_cols:
            if col in df.columns and col in key_vars:
                for lag in lag_periods:
                    df[f'{col}_LAG{lag}'] = df[col].shift(lag)

        n_new = len(df.columns) - len(feature_cols) - 4  # Subtract original housing cols
        print(f"Created {n_new} lagged features")

        return df


def merge_housing_and_fred_data(housing_path: str = 'data/processed/ct_housing_monthly.csv',
                                 fred_path: str = 'data/raw/fred_economic_data.csv',
                                 output_path: str = 'data/processed/ct_housing_with_fred.csv',
                                 lag_periods: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Standalone function to merge housing and FRED data files.

    Parameters:
    -----------
    housing_path : str
        Path to processed housing data CSV
    fred_path : str
        Path to FRED economic data CSV
    output_path : str
        Path to save merged dataset
    lag_periods : list of int, optional
        Lag periods for economic indicators (e.g., [1, 3, 6, 12])

    Returns:
    --------
    pd.DataFrame
        Merged dataset
    """
    print("=" * 80)
    print("Merging Housing and FRED Economic Data")
    print("=" * 80)

    # Load housing data
    print(f"\nLoading housing data from: {housing_path}")
    housing_df = pd.read_csv(housing_path, parse_dates=['date'])
    print(f"Housing data: {housing_df.shape[0]} observations")

    # Load FRED data
    print(f"\nLoading FRED data from: {fred_path}")
    fred_df = pd.read_csv(fred_path, parse_dates=['date'], index_col='date')
    print(f"FRED data: {fred_df.shape[0]} observations, {fred_df.shape[1]} variables")
    print(f"FRED variables: {list(fred_df.columns)}")

    # Align dates - ensure both are at month start
    housing_df['date'] = pd.to_datetime(housing_df['date']).dt.to_period('M').dt.to_timestamp()
    fred_df.index = pd.to_datetime(fred_df.index).to_period('M').to_timestamp()

    # Merge
    merged = housing_df.merge(
        fred_df.reset_index().rename(columns={'index': 'date'}),
        on='date',
        how='left'
    )

    # Create lagged features if requested
    if lag_periods:
        print(f"\nCreating lagged features for periods: {lag_periods}")
        key_vars = [
            'MORTGAGE30US', 'CTURN', 'UNRATE', 'FEDFUNDS',
            'INFLATION_YOY', 'UMCSENT', 'HP_INDEX_YOY', 'REAL_MORTGAGE_RATE'
        ]

        for col in fred_df.columns:
            if col in key_vars:
                for lag in lag_periods:
                    merged[f'{col}_LAG{lag}'] = merged[col].shift(lag)

    # Save merged data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"\nMerged data saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("MERGED DATA SUMMARY")
    print("=" * 80)
    print(f"Shape: {merged.shape}")
    print(f"Date range: {merged['date'].min()} to {merged['date'].max()}")
    print(f"\nColumns ({len(merged.columns)}):")
    for col in merged.columns:
        non_null = merged[col].notna().sum()
        null_pct = (1 - non_null / len(merged)) * 100
        print(f"  {col}: {non_null} obs ({null_pct:.1f}% missing)")

    return merged


def main():
    """
    Main function to preprocess CT Real Estate data.
    """
    preprocessor = CTRealEstatePreprocessor()
    monthly_data = preprocessor.preprocess()

    # Display summary statistics
    print("\nSummary Statistics:")
    print(monthly_data[['median_price', 'mean_price', 'count']].describe())

    # Show first and last few months
    print("\nFirst 12 months:")
    print(monthly_data.head(12))

    print("\nLast 12 months:")
    print(monthly_data.tail(12))


if __name__ == '__main__':
    main()

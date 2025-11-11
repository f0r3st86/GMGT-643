"""
Connecticut Real Estate Data - Manual Download Guide

Unfortunately, Connecticut's Open Data Portal (data.ct.gov) has restricted
programmatic API access (returns 403 Forbidden for all API endpoints).

This script provides detailed instructions for manual download and includes
utilities to validate and process the downloaded data.
"""

import pandas as pd
import os
from datetime import datetime


def print_download_instructions():
    """
    Print detailed instructions for manually downloading CT Real Estate data.
    """
    print("="*80)
    print("CONNECTICUT REAL ESTATE DATA - MANUAL DOWNLOAD INSTRUCTIONS")
    print("="*80)
    print()
    print("Connecticut has restricted API access to their Open Data Portal.")
    print("You must download the data manually through their website.")
    print()
    print("STEP-BY-STEP INSTRUCTIONS:")
    print("-" * 80)
    print()
    print("1. Open your web browser and go to:")
    print("   https://data.ct.gov/")
    print()
    print("2. In the search box, type:")
    print("   'Real Estate Sales'")
    print()
    print("3. Look for one of these datasets:")
    print("   - 'Real Estate Sales 2001-2020 GL'")
    print("   - 'Real Estate Sales 2001-2021 GL'")
    print("   - 'Real Estate Sales 2001-2022 GL' (or latest year)")
    print()
    print("4. Click on the dataset to open it")
    print()
    print("5. Click the 'Export' button (usually in the top right)")
    print()
    print("6. Select 'CSV' format")
    print()
    print("7. Save the file to:")
    print(f"   {os.path.abspath('data/raw/ct_real_estate_raw.csv')}")
    print()
    print("   (Create the 'data/raw' directory if it doesn't exist)")
    print()
    print("="*80)
    print()
    print("ALTERNATIVE: If you have the data from another source")
    print("-" * 80)
    print()
    print("If you already have Connecticut real estate sales data from:")
    print("  - A research database")
    print("  - A previous download")
    print("  - An instructor/colleague")
    print()
    print("Place the CSV file at:")
    print(f"  {os.path.abspath('data/raw/ct_real_estate_raw.csv')}")
    print()
    print("Required columns (approximate names - we'll auto-detect):")
    print("  - Date/Sale Date/List Year (transaction date)")
    print("  - Sale Amount/Price/Assessed Value (price)")
    print("  - Property Type/Residential Type (property classification)")
    print("  - Town (location)")
    print()
    print("="*80)
    print()


def validate_downloaded_file(filepath='data/raw/ct_real_estate_raw.csv'):
    """
    Validate that the downloaded file exists and has the right structure.

    Parameters:
    -----------
    filepath : str
        Path to the downloaded CSV file

    Returns:
    --------
    bool
        True if file is valid, False otherwise
    """
    print("Validating downloaded file...")
    print()

    # Check if file exists
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        print()
        print("Please download the file and place it at the location above.")
        return False

    print(f"✓ File found: {filepath}")

    # Check file size
    file_size = os.path.getsize(filepath)
    file_size_mb = file_size / (1024 * 1024)
    print(f"✓ File size: {file_size_mb:.2f} MB")

    if file_size < 1000:
        print("❌ File is too small - may be empty or corrupted")
        return False

    # Try to load and inspect the file
    try:
        print()
        print("Loading file...")
        df = pd.read_csv(filepath, nrows=1000)  # Load first 1000 rows for validation

        print(f"✓ Successfully loaded CSV")
        print(f"✓ Columns found: {len(df.columns)}")
        print()
        print("Column names:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i:2d}. {col}")

        print()
        print(f"Sample record count: {len(df)}")

        # Check for expected columns (flexible matching)
        expected_fields = {
            'date': ['date', 'sale_date', 'list_year', 'date_recorded', 'sale date'],
            'price': ['sale_amount', 'price', 'assessed_value', 'sale amount', 'assessed value'],
            'property_type': ['property_type', 'residential_type', 'property type', 'type'],
        }

        found_fields = {}
        for field_type, possible_names in expected_fields.items():
            for col in df.columns:
                if any(name.lower() in col.lower() for name in possible_names):
                    found_fields[field_type] = col
                    break

        print()
        print("Field mapping (auto-detected):")
        for field_type, col_name in found_fields.items():
            print(f"  ✓ {field_type}: '{col_name}'")

        missing_fields = set(expected_fields.keys()) - set(found_fields.keys())
        if missing_fields:
            print()
            print(f"⚠ Missing fields: {', '.join(missing_fields)}")
            print("  (The preprocessing script will attempt to work around this)")

        print()
        print("="*80)
        print("VALIDATION SUMMARY")
        print("="*80)
        print("✓ File is valid and can be processed!")
        print()
        print("Next step: Run the preprocessing script:")
        print("  python src/data_preprocessing.py")
        print()

        return True

    except Exception as e:
        print()
        print(f"❌ Error reading file: {e}")
        print()
        print("The file may be corrupted or in the wrong format.")
        print("Please re-download and try again.")
        return False


def check_for_alternative_data_sources():
    """
    Suggest alternative data sources for Connecticut real estate data.
    """
    print("="*80)
    print("ALTERNATIVE DATA SOURCES")
    print("="*80)
    print()
    print("If you cannot access data.ct.gov, consider these alternatives:")
    print()
    print("1. Zillow Research Data")
    print("   URL: https://www.zillow.com/research/data/")
    print("   - Free historical home value data")
    print("   - State and metro-level aggregates")
    print("   - Monthly median prices available")
    print()
    print("2. FRED (Federal Reserve Economic Data)")
    print("   URL: https://fred.stlouisfed.org/")
    print("   - Search: 'Connecticut home prices'")
    print("   - Free economic data")
    print("   - API available")
    print()
    print("3. Realtor.com Economic Research")
    print("   URL: https://www.realtor.com/research/data/")
    print("   - Housing market data")
    print("   - County and metro level")
    print()
    print("4. Academic Data Resources")
    print("   - Check with your university library")
    print("   - Many have access to real estate databases")
    print("   - WRDS, Bloomberg, etc.")
    print()
    print("5. Use Synthetic Data (Already Included)")
    print("   - Run: python src/generate_synthetic_data.py")
    print("   - Creates realistic CT housing data")
    print("   - Based on actual market trends")
    print("   - Good for methodology demonstration")
    print()
    print("="*80)
    print()


def main():
    """
    Main function - guide user through data acquisition.
    """
    print()
    print_download_instructions()

    # Check if file already exists
    filepath = 'data/raw/ct_real_estate_raw.csv'

    if os.path.exists(filepath):
        print(f"Found existing file: {filepath}")
        print()
        response = input("Validate this file? (y/n): ").strip().lower()
        if response == 'y':
            validate_downloaded_file(filepath)
    else:
        print(f"File not found: {filepath}")
        print()
        print("Options:")
        print("  1. Download manually (recommended)")
        print("  2. See alternative data sources")
        print("  3. Use synthetic data")
        print()
        choice = input("Enter choice (1-3): ").strip()

        if choice == '2':
            print()
            check_for_alternative_data_sources()
        elif choice == '3':
            print()
            print("Generating synthetic data...")
            import subprocess
            subprocess.run(['python', 'src/generate_synthetic_data.py'])
        else:
            print()
            print("After downloading, run:")
            print(f"  python {__file__}")
            print()
            print("to validate your download.")


if __name__ == '__main__':
    main()

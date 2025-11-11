"""
Data Acquisition Module
Downloads Connecticut Real Estate Sales data from CT Open Data Portal
Dataset: Real Estate Sales 2001-2023
"""

import pandas as pd
import requests
import os
from sodapy import Socrata
from datetime import datetime


class CTRealEstateDataFetcher:
    """
    Fetches Connecticut Real Estate Sales data from the state's Open Data portal.

    The dataset contains property transaction records from 2001-2023.
    Dataset ID: 5mzw-sjtu (Connecticut Real Estate Sales 2001-2020 GL)
    Note: May need to check for updated dataset IDs on data.ct.gov
    """

    def __init__(self, data_dir='data/raw'):
        """
        Initialize the data fetcher.

        Parameters:
        -----------
        data_dir : str
            Directory to save raw data files
        """
        self.data_dir = data_dir
        self.domain = 'data.ct.gov'
        # Primary dataset ID - this may need updating
        self.dataset_id = '5mzw-sjtu'

        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)

    def fetch_data_via_api(self, limit=None):
        """
        Fetch data using Socrata API.

        Parameters:
        -----------
        limit : int, optional
            Maximum number of records to fetch. If None, fetches all records.

        Returns:
        --------
        pd.DataFrame
            Raw real estate sales data
        """
        print(f"Connecting to CT Open Data Portal ({self.domain})...")

        try:
            # Initialize Socrata client (unauthenticated - has rate limits)
            client = Socrata(self.domain, None)

            # Fetch data
            print(f"Fetching data from dataset {self.dataset_id}...")
            if limit:
                results = client.get(self.dataset_id, limit=limit)
            else:
                # Fetch all records (this might take a while)
                results = client.get(self.dataset_id, limit=2000000)

            # Convert to DataFrame
            df = pd.DataFrame.from_records(results)
            print(f"Successfully fetched {len(df):,} records")

            client.close()
            return df

        except Exception as e:
            print(f"Error fetching data via API: {e}")
            print("\nAlternative: Download manually from:")
            print(f"https://{self.domain}/resource/{self.dataset_id}")
            raise

    def fetch_data_via_url(self, url=None):
        """
        Fetch data via direct CSV download URL.

        Parameters:
        -----------
        url : str, optional
            Direct download URL. If None, constructs from dataset_id.

        Returns:
        --------
        pd.DataFrame
            Raw real estate sales data
        """
        if url is None:
            url = f"https://{self.domain}/resource/{self.dataset_id}.csv?$limit=9999999"

        print(f"Downloading data from URL...")
        print(f"URL: {url}")

        try:
            df = pd.read_csv(url)
            print(f"Successfully downloaded {len(df):,} records")
            return df

        except Exception as e:
            print(f"Error downloading data: {e}")
            raise

    def save_raw_data(self, df, filename=None):
        """
        Save raw data to CSV file.

        Parameters:
        -----------
        df : pd.DataFrame
            Data to save
        filename : str, optional
            Output filename. If None, uses timestamp.
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'ct_real_estate_raw_{timestamp}.csv'

        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Data saved to: {filepath}")
        return filepath

    def download_and_save(self, method='api', limit=None, filename='ct_real_estate_raw.csv'):
        """
        Download data and save to file.

        Parameters:
        -----------
        method : str
            Download method: 'api' or 'url'
        limit : int, optional
            Record limit for API method
        filename : str
            Output filename

        Returns:
        --------
        str
            Path to saved file
        """
        if method == 'api':
            df = self.fetch_data_via_api(limit=limit)
        elif method == 'url':
            df = self.fetch_data_via_url()
        else:
            raise ValueError(f"Unknown method: {method}")

        filepath = self.save_raw_data(df, filename=filename)

        # Print basic info
        print("\nDataset Info:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nFirst few rows:")
        print(df.head())

        return filepath


def main():
    """
    Main function to download CT Real Estate data.
    """
    print("=" * 80)
    print("Connecticut Real Estate Sales Data Acquisition")
    print("=" * 80)

    fetcher = CTRealEstateDataFetcher()

    # Try to download data
    # Start with API method (you may need to adjust based on dataset availability)
    try:
        filepath = fetcher.download_and_save(method='url', filename='ct_real_estate_raw.csv')
        print(f"\n{'=' * 80}")
        print(f"SUCCESS! Data downloaded to: {filepath}")
        print(f"{'=' * 80}")

    except Exception as e:
        print(f"\nDownload failed: {e}")
        print("\nManual Download Instructions:")
        print("1. Visit: https://data.ct.gov/")
        print("2. Search for 'Real Estate Sales 2001-2023' or similar")
        print("3. Download the CSV file")
        print("4. Save it to: data/raw/ct_real_estate_raw.csv")


if __name__ == '__main__':
    main()

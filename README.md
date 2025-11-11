# Connecticut Housing Price Forecasting: Time Series Analysis

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/f0r3st86/GMGT-643/blob/main/notebooks/CT_Housing_Price_Forecasting_Colab.ipynb)

## Project Overview

This research project examines whether classical time series methods can accurately forecast median residential sale prices in Connecticut. The study tests whether Holt-Winters' exponential smoothing provides significant improvements over simple models like naïve and seasonal naïve forecasts, applying forecasting techniques to real housing market data with particular attention to crisis periods like the 2008 Financial Crisis and the COVID-19 pandemic.

## Research Questions

1. Can classical time series methods accurately forecast Connecticut housing prices?
2. Does Holt-Winters exponential smoothing outperform simpler forecasting methods?
3. How do forecasting errors vary during economic crisis periods?
4. What seasonal patterns exist in Connecticut housing prices?

## Dataset

**Source**: Connecticut Open Data Portal - Real Estate Sales 2001-2023

**Coverage**:
- Time period: January 2001 - December 2023 (276 monthly observations)
- Geographic scope: State of Connecticut
- Property types: Residential arms-length sales

**Filtering Criteria**:
- Residential properties only (single-family, condos, multi-family)
- Arms-length transactions (genuine market transactions between unrelated parties)
- Price range: $10,000 minimum, outliers removed

## Methodology

### Training and Testing Split

- **Training Period**: 2001-2018 (18 years)
- **Testing Period**: 2019-2023 (5 years)
- This split simulates real-world forecasting conditions

### Forecasting Models

Five forecasting approaches are compared:

1. **Naïve Model**: Uses the most recent observed price as the forecast
2. **Seasonal Naïve Model**: Uses the price from the same month one year ago
3. **Moving Average Model**: Averages recent months to smooth short-term fluctuations
4. **Holt-Winters Exponential Smoothing**: Captures level, trend, and seasonal patterns with adaptive weighting
5. **OLS Regression**: Uses time trend and monthly dummy variables

### Evaluation Metrics

Models are evaluated using three standard error metrics:

- **RMSE (Root Mean Squared Error)**: Penalizes large forecast errors
- **MAE (Mean Absolute Error)**: Shows average error in dollar terms
- **MAPE (Mean Absolute Percentage Error)**: Shows average error as a percentage of price

## Project Structure

```
GMGT-643/
├── data/
│   ├── raw/                    # Raw data files (not tracked in git)
│   └── processed/              # Processed time series data
├── src/
│   ├── data_acquisition.py     # Downloads CT Real Estate data
│   ├── data_preprocessing.py   # Filters and creates time series
│   ├── models.py               # Five forecasting models
│   ├── evaluation.py           # Performance metrics (RMSE, MAE, MAPE)
│   ├── visualization.py        # Plotting functions
│   └── main_analysis.py        # Main analysis pipeline
├── notebooks/
│   └── CT_Housing_Price_Forecasting_Colab.ipynb  # Google Colab notebook (complete analysis)
├── results/
│   ├── figures/                # Visualization outputs
│   └── tables/                 # Results tables
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation & Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd GMGT-643
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### ⚡ Quickest Start: Google Colab (Recommended)

The easiest way to run this analysis is using Google Colab - no setup required!

**Option 1: Direct Link**
1. Open the notebook in Colab: [CT_Housing_Price_Forecasting_Colab.ipynb](notebooks/CT_Housing_Price_Forecasting_Colab.ipynb)
2. Click "Open in Colab" or upload to your Google Drive
3. Run all cells sequentially (Runtime > Run all)
4. Results and visualizations will appear inline

**Option 2: Upload to Colab**
1. Go to https://colab.research.google.com/
2. Upload `notebooks/CT_Housing_Price_Forecasting_Colab.ipynb`
3. Run all cells

The Colab notebook includes:
- ✓ Automatic package installation
- ✓ Data acquisition and preprocessing
- ✓ All 5 forecasting models
- ✓ Comprehensive visualizations
- ✓ Model evaluation and comparison
- ✓ Crisis period analysis
- ✓ Results export

---

### Quick Start: Run Complete Analysis Locally

```bash
# Option 1: Run all steps sequentially
python src/data_acquisition.py
python src/data_preprocessing.py
python src/main_analysis.py

# Option 2: Run main analysis (if data already processed)
python src/main_analysis.py
```

### Step-by-Step Workflow

#### Step 1: Data Acquisition

Download Connecticut Real Estate Sales data:

```bash
python src/data_acquisition.py
```

This will download the raw data to `data/raw/ct_real_estate_raw.csv`.

**Manual Download Alternative**: If the automated download fails:
1. Visit https://data.ct.gov/
2. Search for "Real Estate Sales 2001-2023"
3. Download the CSV file
4. Save it as `data/raw/ct_real_estate_raw.csv`

#### Step 2: Data Preprocessing

Filter data and create monthly time series:

```bash
python src/data_preprocessing.py
```

This script:
- Filters for residential properties
- Filters for arms-length transactions
- Creates monthly median price series
- Saves processed data to `data/processed/ct_housing_monthly.csv`

#### Step 3: Run Analysis

Execute the complete forecasting analysis:

```bash
python src/main_analysis.py
```

This script:
- Loads processed data
- Splits into training and testing periods
- Fits all 5 forecasting models
- Generates forecasts for 2019-2023
- Evaluates model performance
- Creates visualizations
- Analyzes crisis period performance
- Saves all results

### Interactive Analysis (Local Jupyter)

If you prefer to run the notebook locally:

```bash
jupyter notebook notebooks/CT_Housing_Price_Forecasting_Colab.ipynb
```

**Note**: The notebook is optimized for Google Colab but works in local Jupyter with minor modifications (remove Colab-specific code like `files.download()`).

## Expected Outcomes

Based on the research design, the expected outcomes are:

1. **Forecast Accuracy**: MAPE of 4-8% for the best models
2. **Model Performance**: Holt-Winters outperforms simple models by 20-40%
3. **Crisis Impact**: Larger forecasting errors during crisis periods (2020 COVID-19)
4. **Seasonality**: Clear seasonal patterns with higher prices in spring and summer

## Results

After running the analysis, results will be saved to:

- **Visualizations**: `results/figures/`
  - `01_time_series.png` - Complete time series plot
  - `02_train_test_split.png` - Training and testing periods
  - `03_forecast_comparison.png` - All model forecasts vs. actual
  - `04_model_comparison.png` - Performance metrics comparison
  - `05_forecast_errors.png` - Error patterns over time
  - `06_seasonal_pattern.png` - Monthly seasonal patterns
  - `07_crisis_analysis.png` - Crisis period analysis

- **Tables**: `results/tables/`
  - `model_comparison.csv` - Model performance metrics
  - `crisis_analysis.csv` - Crisis period performance

## Key Contributions

This project makes two key contributions:

1. **Practical Application**: Demonstrates the application of classical time series methods to a real-world dataset directly relevant to economic decision-making

2. **Quantified Value**: Shows stakeholders whether sophisticated models are worth the complexity over simple rules of thumb by quantifying forecasting errors in real dollar terms

## Technologies Used

- **Python 3.8+**
- **Data Analysis**: pandas, numpy
- **Time Series**: statsmodels
- **Visualization**: matplotlib, seaborn
- **Statistical Analysis**: scipy

## Future Extensions

Potential extensions for this research:

1. Geographic analysis by county or city
2. Property type stratification (single-family vs. condos)
3. Advanced models (ARIMA, SARIMA, Prophet, LSTM)
4. External variables (unemployment, interest rates, inventory)
5. Forecast combination methods
6. Real-time updating dashboard

## References

- Connecticut Open Data Portal: https://data.ct.gov/
- Hyndman, R.J., & Athanasopoulos, G. (2021). *Forecasting: principles and practice* (3rd ed.)
- Box, G.E.P., Jenkins, G.M., Reinsel, G.C., & Ljung, G.M. (2015). *Time Series Analysis: Forecasting and Control* (5th ed.)

## Author

GMGT-643 Research Project

## License

This project is for educational purposes.

---

**Last Updated**: November 2024
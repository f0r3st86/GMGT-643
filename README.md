# Connecticut Housing Price Forecasting: Time Series Analysis

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

Six forecasting approaches are compared:

1. **Naïve Model**: Uses the most recent observed price as the forecast
2. **Seasonal Naïve Model**: Uses the price from the same month one year ago
3. **Moving Average Model**: Averages recent months to smooth short-term fluctuations
4. **Holt-Winters Exponential Smoothing**: Captures level, trend, and seasonal patterns with adaptive weighting
5. **OLS Regression**: Uses time trend and monthly dummy variables
6. **OLS with Exogenous Variables**: Incorporates FRED economic indicators as additional predictors

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
│   ├── fred_data_acquisition.py # Downloads FRED economic indicators
│   ├── data_preprocessing.py   # Filters and creates time series
│   ├── models.py               # Six forecasting models (incl. exogenous)
│   ├── evaluation.py           # Performance metrics (RMSE, MAE, MAPE)
│   ├── visualization.py        # Plotting functions
│   └── main_analysis.py        # Main analysis pipeline
├── notebooks/
│   └── analysis.ipynb          # Jupyter notebook for interactive analysis
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

### Quick Start: Run Complete Analysis

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

### Interactive Analysis

Use the Jupyter notebook for interactive exploration:

```bash
jupyter notebook notebooks/analysis.ipynb
```

## Using FRED Economic Data (Enhanced Analysis)

The project now supports incorporating FRED (Federal Reserve Economic Data) indicators to improve forecasting accuracy.

### Available FRED Indicators

| Series ID | Description | Category |
|-----------|-------------|----------|
| `MORTGAGE30US` | 30-Year Fixed Mortgage Rate | Financing |
| `CTURN` | Connecticut Unemployment Rate | Employment |
| `UNRATE` | National Unemployment Rate | Employment |
| `FEDFUNDS` | Federal Funds Rate | Monetary Policy |
| `CPIAUCSL` | Consumer Price Index | Inflation |
| `CSUSHPINSA` | Case-Shiller Home Price Index | Housing Market |
| `UMCSENT` | Consumer Sentiment Index | Sentiment |

### Step 1: Get a FRED API Key (Free)

1. Visit: https://fred.stlouisfed.org/docs/api/api_key.html
2. Create a free account
3. Request an API key
4. Set it as an environment variable:

```bash
export FRED_API_KEY='your-api-key-here'
```

### Step 2: Download FRED Data

```bash
python src/fred_data_acquisition.py
```

This will download economic indicators to `data/raw/fred_economic_data.csv`.

### Step 3: Merge Housing and FRED Data

```python
from src.data_preprocessing import merge_housing_and_fred_data

# Merge datasets with optional lagged features
merged_data = merge_housing_and_fred_data(
    housing_path='data/processed/ct_housing_monthly.csv',
    fred_path='data/raw/fred_economic_data.csv',
    output_path='data/processed/ct_housing_with_fred.csv',
    lag_periods=[1, 3, 6, 12]  # Create lagged versions of indicators
)
```

### Step 4: Run Analysis with Exogenous Variables

```python
from src.models import fit_and_forecast_with_exogenous, compare_exogenous_impact

# Compare models with and without FRED data
results, model = compare_exogenous_impact(
    train_data=train_prices,
    exog_train=train_fred_data,
    test_data=test_prices,
    exog_test=test_fred_data
)
```

### Derived Features

The FRED module automatically creates derived features:

- **INFLATION_YOY**: Year-over-year CPI change
- **REAL_MORTGAGE_RATE**: Nominal rate minus inflation
- **MORTGAGE_SPREAD**: Mortgage rate minus Fed Funds rate
- **MORTGAGE_CHANGE_12M**: 12-month change in mortgage rates
- **HP_INDEX_YOY**: Year-over-year Case-Shiller change

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
- **External Data**: fredapi (Federal Reserve Economic Data)

## Future Extensions

Potential extensions for this research:

1. Geographic analysis by county or city
2. Property type stratification (single-family vs. condos)
3. Advanced models (ARIMA, SARIMA, Prophet, LSTM)
4. Additional FRED indicators (housing inventory, building permits)
5. Forecast combination methods
6. Real-time updating dashboard
7. Machine learning models with economic features

## References

- Connecticut Open Data Portal: https://data.ct.gov/
- Hyndman, R.J., & Athanasopoulos, G. (2021). *Forecasting: principles and practice* (3rd ed.)
- Box, G.E.P., Jenkins, G.M., Reinsel, G.C., & Ljung, G.M. (2015). *Time Series Analysis: Forecasting and Control* (5th ed.)

## Author

GMGT-643 Research Project

## License

This project is for educational purposes.

---

**Last Updated**: December 2024
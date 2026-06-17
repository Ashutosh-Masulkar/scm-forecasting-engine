# SCM Demand Forecasting Engine

**Live app:** [Launch on Streamlit Cloud](https://scm-forecasting-engine.streamlit.app) <!-- update link after deploying -->

**One-line business summary:** A supply-chain planning app that forecasts weekly demand, selects the best model by MAPE, and converts forecast output into inventory policy decisions.

## Business Problem

Retail supply chains need accurate demand forecasts to balance product availability, working capital, and inventory risk. Poor forecasts can create excess inventory, higher carrying costs, missed sales, and reactive replenishment decisions.

This project simulates a demand planning workflow where a planner can:

- Review historical sales patterns across a retail store network
- Compare multiple forecasting models
- Select the most accurate model using validation MAPE
- Translate forecasted demand into safety stock, reorder point, and carrying cost estimates
- Explore service-level tradeoffs between customer availability and inventory cost

## Dataset Description

The project uses a public Walmart sales dataset containing weekly sales history for 45 stores. The dataset includes:

- `Store`: Walmart store identifier
- `Date`: Weekly sales date
- `Weekly_Sales`: Weekly sales dollars
- `Holiday_Flag`: Indicator for holiday weeks
- `Temperature`: Local temperature
- `Fuel_Price`: Fuel price
- `CPI`: Consumer Price Index
- `Unemployment`: Unemployment rate

**Demand proxy note:** The dataset provides `Weekly_Sales` in dollars, not item-level unit demand. For this portfolio project, `Weekly_Sales` is used as a demand proxy. In a production supply-chain system, this would typically be replaced with SKU/store-level unit demand or converted using average selling price.

## Key Features

- Interactive Streamlit planning workbench
- Demand review for individual stores or the full store network
- All Stores aggregate analysis plus store-level analysis
- Weekly trend and 4-week rolling trend views
- Monthly seasonality analysis
- Forecast horizon selection for 4, 8, or 12 weeks
- Three-model forecasting engine: ARIMA, Prophet, and XGBoost
- Validation-based MAPE comparison
- Automatic winner selection using lowest MAPE
- Future forecast visualization with confidence bounds
- Actual vs predicted validation chart
- Inventory policy calculator for safety stock, reorder point, inventory value, and annual holding cost
- What-if analysis for service-level and inventory-cost tradeoffs
- Business impact calculator logic for estimating value from forecast accuracy improvements

## App Pages

### 1. Demand Review

Explores historical demand before forecasting. Users can select either **All Stores** or an individual store, review weekly sales trends, switch to a 4-week rolling view, inspect seasonality, and view source rows.

### 2. Forecast Engine

Runs ARIMA, Prophet, and XGBoost for the selected scope and forecast horizon. The app compares models using MAPE and automatically selects the lowest-MAPE model as the planning forecast.

### 3. Inventory Policy

Converts the winning forecast into actionable inventory metrics. Users can adjust unit cost, holding cost percentage, lead time, and service level to see how policy choices affect safety stock, reorder point, inventory value, and annual carrying cost.

## Forecasting Models

### ARIMA

A traditional time-series forecasting model that uses historical sales patterns to project future demand.

### Prophet

A seasonal forecasting model designed to capture trend and recurring seasonal patterns in weekly retail sales.

### XGBoost

A machine-learning forecasting model that uses historical demand lags, rolling statistics, holiday flags, weather, fuel price, CPI, unemployment, week, month, and year features.

## MAPE-Based Winner Selection

The app validates each model against the latest historical period and calculates Mean Absolute Percentage Error:

```text
MAPE = mean(abs(actual - forecast) / actual) * 100
```

The model with the lowest MAPE is selected as the winning model. This creates a simple model competition framework that mirrors real demand planning workflows where forecast accuracy drives planning confidence.

## All Stores vs Individual Store Analysis

The app supports two planning views:

- **All Stores:** Aggregates weekly sales across the full network to support network-level demand planning.
- **Individual Store:** Filters demand history to a single store to analyze localized demand patterns, volatility, and replenishment needs.

This distinction is important because network-level demand may be smoother, while store-level demand may show more local volatility.

## Inventory Formulas

### Safety Stock

```text
Safety Stock = Z x demand standard deviation x sqrt(lead time)
```

Where:

- `Z` is the z-score for the selected service level
- `demand standard deviation` represents historical demand variability
- `lead time` is measured in weeks

### Reorder Point

```text
Reorder Point = average weekly demand x lead time + safety stock
```

The reorder point estimates the inventory level at which replenishment should be triggered.

## Business Impact Calculator

The forecasting engine includes logic to estimate the financial impact of forecast accuracy improvement. It uses current MAPE, improved MAPE, average weekly demand, unit cost, and holding cost percentage to estimate:

- Current forecast error exposure
- Improved forecast error exposure
- Reduced error units
- Inventory value reduction
- Annual holding cost savings

This connects forecast accuracy to business value, which is critical for supply-chain analytics storytelling.

## What-If and Service-Level Tradeoff

The Inventory Policy page includes a service-level tradeoff view. Users can change service level assumptions and see how higher availability targets increase safety stock and carrying cost.

This demonstrates a core supply-chain tradeoff:

```text
Higher service level -> lower stockout risk -> higher inventory cost
Lower service level -> lower inventory cost -> higher stockout risk
```

## Tech Stack

- **Python** for data processing and forecasting logic
- **Streamlit** for the interactive web app
- **Pandas** and **NumPy** for data preparation
- **Statsmodels** for ARIMA forecasting
- **Prophet** for seasonal time-series forecasting
- **XGBoost** for machine-learning forecasting
- **Scikit-learn** for model evaluation metrics
- **Plotly** for interactive visualizations

## How to Run Locally

1. Clone the repository.

2. Create and activate a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies.

```bash
pip install -r requirements.txt
```

4. Run the Streamlit app.

```bash
streamlit run app.py
```

5. Open the local Streamlit URL shown in the terminal.


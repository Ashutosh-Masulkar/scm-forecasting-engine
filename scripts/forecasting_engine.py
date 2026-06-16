# scripts/forecasting_engine.py

import warnings
from statistics import NormalDist

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet


warnings.filterwarnings("ignore")


# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------

def load_walmart_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    required = ["Store", "Date", "Weekly_Sales", "Holiday_Flag", "Temperature", "Fuel_Price", "CPI", "Unemployment"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date", "Weekly_Sales"])
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)
    return df


def get_store_data(df: pd.DataFrame, store_number: int) -> pd.DataFrame:
    store_df = df[df["Store"] == store_number].copy().sort_values("Date").reset_index(drop=True)
    if store_df.empty:
        raise ValueError(f"No data found for Store {store_number}")
    return store_df


# ---------------------------------------------------------
# ACCURACY METRICS
# ---------------------------------------------------------

def calculate_mape(actual, predicted) -> float:
    return mean_absolute_percentage_error(np.array(actual), np.array(predicted)) * 100


def calculate_rmse(actual, predicted) -> float:
    return float(np.sqrt(mean_squared_error(np.array(actual), np.array(predicted))))


def calculate_mae(actual, predicted) -> float:
    return float(mean_absolute_error(np.array(actual), np.array(predicted)))


# ---------------------------------------------------------
# ARIMA
# ---------------------------------------------------------

def run_arima_forecast(store_df: pd.DataFrame, horizon: int) -> dict:
    sales = store_df["Weekly_Sales"].astype(float)
    train, test = sales.iloc[:-horizon], sales.iloc[-horizon:]

    fitted = SARIMAX(train, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0),
                     enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    val_forecast = fitted.forecast(steps=horizon)

    final_fitted = SARIMAX(sales, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0),
                           enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    fc_result = final_fitted.get_forecast(steps=horizon)
    future = fc_result.predicted_mean
    ci = fc_result.conf_int()

    future_dates = pd.date_range(
        start=store_df["Date"].max() + pd.Timedelta(weeks=1), periods=horizon, freq="W-FRI"
    )

    return {
        "model_name": "ARIMA",
        "mape": calculate_mape(test, val_forecast),
        "rmse": calculate_rmse(test, val_forecast),
        "mae": calculate_mae(test, val_forecast),
        "forecast_df": pd.DataFrame({
            "Date": future_dates, "Model": "ARIMA",
            "Forecast": future.values,
            "Lower_Bound": ci.iloc[:, 0].values,
            "Upper_Bound": ci.iloc[:, 1].values,
        }),
        "validation_df": pd.DataFrame({
            "Date": store_df["Date"].iloc[-horizon:].values,
            "Actual": test.values,
            "Predicted": val_forecast.values,
            "Model": "ARIMA",
        }),
    }


# ---------------------------------------------------------
# PROPHET
# ---------------------------------------------------------

def run_prophet_forecast(store_df: pd.DataFrame, horizon: int) -> dict:
    pdf = store_df[["Date", "Weekly_Sales"]].rename(columns={"Date": "ds", "Weekly_Sales": "y"})
    train, test = pdf.iloc[:-horizon], pdf.iloc[-horizon:]

    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False, interval_width=0.95)
    model.fit(train)
    val_pred = model.predict(model.make_future_dataframe(periods=horizon, freq="W-FRI"))
    val_forecast = val_pred.tail(horizon)["yhat"].values

    final = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False, interval_width=0.95)
    final.fit(pdf)
    fut = final.predict(final.make_future_dataframe(periods=horizon, freq="W-FRI")).tail(horizon)

    return {
        "model_name": "Prophet",
        "mape": calculate_mape(test["y"], val_forecast),
        "rmse": calculate_rmse(test["y"], val_forecast),
        "mae": calculate_mae(test["y"], val_forecast),
        "forecast_df": pd.DataFrame({
            "Date": fut["ds"].values, "Model": "Prophet",
            "Forecast": fut["yhat"].values,
            "Lower_Bound": fut["yhat_lower"].values,
            "Upper_Bound": fut["yhat_upper"].values,
        }),
        "validation_df": pd.DataFrame({
            "Date": store_df["Date"].iloc[-horizon:].values,
            "Actual": test["y"].values,
            "Predicted": val_forecast,
            "Model": "Prophet",
        }),
    }


# ---------------------------------------------------------
# XGBOOST
# ---------------------------------------------------------

def _xgb_feature_columns() -> list:
    return ["Holiday_Flag", "Temperature", "Fuel_Price", "CPI", "Unemployment",
            "Week", "Month", "Year", "Lag_1", "Lag_2", "Lag_4", "Lag_8",
            "Rolling_Mean_4", "Rolling_Std_4"]


def _build_xgb_features(store_df: pd.DataFrame) -> pd.DataFrame:
    df = store_df.copy()
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Month"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year
    df["Lag_1"] = df["Weekly_Sales"].shift(1)
    df["Lag_2"] = df["Weekly_Sales"].shift(2)
    df["Lag_4"] = df["Weekly_Sales"].shift(4)
    df["Lag_8"] = df["Weekly_Sales"].shift(8)
    df["Rolling_Mean_4"] = df["Weekly_Sales"].shift(1).rolling(4).mean()
    df["Rolling_Std_4"] = df["Weekly_Sales"].shift(1).rolling(4).std()
    return df.dropna().reset_index(drop=True)


def run_xgboost_forecast(store_df: pd.DataFrame, horizon: int) -> dict:
    feature_df = _build_xgb_features(store_df)
    cols = _xgb_feature_columns()

    train_df, test_df = feature_df.iloc[:-horizon], feature_df.iloc[-horizon:]
    X_train, y_train = train_df[cols], train_df["Weekly_Sales"]
    X_test, y_test = test_df[cols], test_df["Weekly_Sales"]

    params = dict(n_estimators=300, learning_rate=0.05, max_depth=3,
                  subsample=0.9, colsample_bytree=0.9, random_state=42,
                  objective="reg:squarederror")

    val_model = XGBRegressor(**params)
    val_model.fit(X_train, y_train)
    val_forecast = val_model.predict(X_test)
    residual_std = float(np.std(y_test.values - val_forecast))

    final_model = XGBRegressor(**params)
    final_model.fit(feature_df[cols], feature_df["Weekly_Sales"])
    feature_importance = dict(zip(cols, final_model.feature_importances_.tolist()))

    history = list(store_df["Weekly_Sales"].astype(float).values)
    last_row = store_df.iloc[-1]
    future_dates = pd.date_range(
        start=store_df["Date"].max() + pd.Timedelta(weeks=1), periods=horizon, freq="W-FRI"
    )

    future_values = []
    for fd in future_dates:
        row = pd.DataFrame({
            "Holiday_Flag": [0],
            "Temperature": [last_row["Temperature"]],
            "Fuel_Price": [last_row["Fuel_Price"]],
            "CPI": [last_row["CPI"]],
            "Unemployment": [last_row["Unemployment"]],
            "Week": [fd.isocalendar().week],
            "Month": [fd.month],
            "Year": [fd.year],
            "Lag_1": [history[-1]],
            "Lag_2": [history[-2]],
            "Lag_4": [history[-4]],
            "Lag_8": [history[-8]],
            "Rolling_Mean_4": [np.mean(history[-4:])],
            "Rolling_Std_4": [np.std(history[-4:])],
        })
        pred = float(final_model.predict(row[cols])[0])
        future_values.append(pred)
        history.append(pred)

    future_arr = np.array(future_values)

    return {
        "model_name": "XGBoost",
        "mape": calculate_mape(y_test, val_forecast),
        "rmse": calculate_rmse(y_test, val_forecast),
        "mae": calculate_mae(y_test, val_forecast),
        "feature_importance": feature_importance,
        "forecast_df": pd.DataFrame({
            "Date": future_dates, "Model": "XGBoost",
            "Forecast": future_values,
            "Lower_Bound": future_arr - 1.96 * residual_std,
            "Upper_Bound": future_arr + 1.96 * residual_std,
        }),
        "validation_df": pd.DataFrame({
            "Date": test_df["Date"].values,
            "Actual": y_test.values,
            "Predicted": val_forecast,
            "Model": "XGBoost",
        }),
    }


# ---------------------------------------------------------
# AGGREGATION
# ---------------------------------------------------------

def build_forecast_results(results: list) -> dict:
    metrics_df = pd.DataFrame({
        "Model": [r["model_name"] for r in results],
        "MAPE (%)": [round(r["mape"], 2) for r in results],
        "RMSE ($)": [round(r["rmse"], 0) for r in results],
        "MAE ($)": [round(r["mae"], 0) for r in results],
    }).sort_values("MAPE (%)").reset_index(drop=True)

    winning_name = metrics_df.iloc[0]["Model"]
    winning_result = next(r for r in results if r["model_name"] == winning_name)

    # Always surface XGBoost feature importance if available, regardless of which model won
    xgb_result = next((r for r in results if r["model_name"] == "XGBoost"), None)

    return {
        "metrics_df": metrics_df,
        "winning_model": winning_name,
        "winning_mape": float(metrics_df.iloc[0]["MAPE (%)"]),
        "winning_forecast_df": winning_result["forecast_df"],
        "all_forecasts_df": pd.concat([r["forecast_df"] for r in results], ignore_index=True),
        "all_validation_df": pd.concat([r["validation_df"] for r in results], ignore_index=True),
        "feature_importance": xgb_result["feature_importance"] if xgb_result else None,
    }


def run_all_forecasts(store_df: pd.DataFrame, horizon: int) -> dict:
    results = []
    for fn in [run_arima_forecast, run_prophet_forecast, run_xgboost_forecast]:
        try:
            results.append(fn(store_df, horizon))
        except Exception:
            pass
    if not results:
        raise RuntimeError("All forecasting models failed.")
    return build_forecast_results(results)


# ---------------------------------------------------------
# INVENTORY OPTIMIZATION
# ---------------------------------------------------------

def calculate_safety_stock(demand_std: float, lead_time_weeks: float, service_level: float) -> float:
    z = NormalDist().inv_cdf(service_level / 100)
    return max(z * demand_std * np.sqrt(lead_time_weeks), 0)


def calculate_reorder_point(average_weekly_demand: float, lead_time_weeks: float, safety_stock: float) -> float:
    return average_weekly_demand * lead_time_weeks + safety_stock


def calculate_inventory_metrics(
    forecast_df: pd.DataFrame,
    historical_sales: pd.Series,
    unit_cost: float,
    holding_cost_percentage: float,
    lead_time_weeks: float,
    service_level: float,
) -> dict:
    avg_demand = forecast_df["Forecast"].mean()
    demand_std = historical_sales.std()
    safety_stock = calculate_safety_stock(demand_std, lead_time_weeks, service_level)
    reorder_point = calculate_reorder_point(avg_demand, lead_time_weeks, safety_stock)
    inventory_value = safety_stock * unit_cost
    return {
        "average_forecast_demand": avg_demand,
        "demand_std": demand_std,
        "safety_stock": safety_stock,
        "reorder_point": reorder_point,
        "inventory_value": inventory_value,
        "annual_holding_cost": inventory_value * (holding_cost_percentage / 100),
    }


# ---------------------------------------------------------
# BUSINESS IMPACT
# ---------------------------------------------------------

def calculate_forecast_accuracy_impact(
    current_mape: float,
    average_weekly_demand: float,
    unit_cost: float,
    holding_cost_percentage: float,
    improvement_percentage: float = 10,
) -> dict:
    improved_mape = current_mape * (1 - improvement_percentage / 100)
    current_error = average_weekly_demand * (current_mape / 100)
    improved_error = average_weekly_demand * (improved_mape / 100)
    reduced_error = current_error - improved_error
    inv_reduction = reduced_error * unit_cost
    return {
        "current_mape": current_mape,
        "improved_mape": improved_mape,
        "current_error_units": current_error,
        "improved_error_units": improved_error,
        "reduced_error_units": reduced_error,
        "inventory_value_reduction": inv_reduction,
        "annual_holding_savings": inv_reduction * (holding_cost_percentage / 100),
    }

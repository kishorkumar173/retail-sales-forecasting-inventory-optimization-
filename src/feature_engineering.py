"""
Module: feature_engineering.py
Description: Generates multi-horizon time-series features (lags, rolling statistics,
             calendar seasonality, cyclical encodings, and promotional features)
             for retail demand forecasting.
"""

import pandas as pd
import numpy as np
import os

def create_features(
    input_path: str = "data/processed/retail_sales_cleaned.csv",
    output_path: str = "data/processed/retail_features.csv"
) -> pd.DataFrame:
    """
    Constructs high-impact tabular features from historical retail time-series data.
    
    Parameters:
    -----------
    input_path : str, path to cleaned CSV
    output_path : str, path to output feature-engineered CSV
    
    Returns:
    --------
    pd.DataFrame: Feature matrix ready for Gradient Boosting / ML models
    """
    df = pd.read_csv(input_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["store_id", "sku_id", "date"]).reset_index(drop=True)
    
    # Target column to forecast
    target_col = "target_demand_clipped" if "target_demand_clipped" in df.columns else "units_sold"
    
    print("[INFO] Generating Calendar & Temporal Features...")
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    
    # Cyclical representations of periodic calendar components
    df["sin_day_of_week"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["cos_day_of_week"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
    
    print("[INFO] Generating Historical Lag Features (1, 2, 7, 14, 21, 28 days)...")
    for lag in [1, 2, 7, 14, 21, 28]:
        df[f"lag_{lag}"] = df.groupby(["store_id", "sku_id"])[target_col].shift(lag)
        
    print("[INFO] Generating Rolling Window Aggregations (7, 14, 28 days)...")
    # Shift by 1 first so current day's demand is not leaked into the rolling window
    shifted_target = df.groupby(["store_id", "sku_id"])[target_col].shift(1)
    
    # 7-day rolling window
    df["rolling_mean_7"] = df.groupby(["store_id", "sku_id"])[target_col].transform(
        lambda s: s.shift(1).rolling(7, min_periods=3).mean()
    )
    df["rolling_std_7"] = df.groupby(["store_id", "sku_id"])[target_col].transform(
        lambda s: s.shift(1).rolling(7, min_periods=3).std()
    )
    df["rolling_max_7"] = df.groupby(["store_id", "sku_id"])[target_col].transform(
        lambda s: s.shift(1).rolling(7, min_periods=3).max()
    )
    df["rolling_min_7"] = df.groupby(["store_id", "sku_id"])[target_col].transform(
        lambda s: s.shift(1).rolling(7, min_periods=3).min()
    )
    
    # 14-day & 28-day rolling window
    df["rolling_mean_14"] = df.groupby(["store_id", "sku_id"])[target_col].transform(
        lambda s: s.shift(1).rolling(14, min_periods=7).mean()
    )
    df["rolling_std_14"] = df.groupby(["store_id", "sku_id"])[target_col].transform(
        lambda s: s.shift(1).rolling(14, min_periods=7).std()
    )
    df["rolling_mean_28"] = df.groupby(["store_id", "sku_id"])[target_col].transform(
        lambda s: s.shift(1).rolling(28, min_periods=14).mean()
    )
    df["rolling_std_28"] = df.groupby(["store_id", "sku_id"])[target_col].transform(
        lambda s: s.shift(1).rolling(28, min_periods=14).std()
    )
    
    # 7-day Exponential Moving Average (EMA)
    df["ema_7"] = df.groupby(["store_id", "sku_id"])[target_col].transform(
        lambda s: s.shift(1).ewm(span=7, adjust=False).mean()
    )
    
    # Momentum & Velocity Features
    df["ratio_lag1_to_roll7"] = df["lag_1"] / (df["rolling_mean_7"] + 1e-5)
    df["ratio_roll7_to_roll28"] = df["rolling_mean_7"] / (df["rolling_mean_28"] + 1e-5)
    
    # Price & Promotion Elasticity Features
    df["price_discount_ratio"] = df["selling_price"] / (df["base_unit_price"] + 1e-5)
    df["is_promo_weekend"] = df["on_promotion"] * df["is_weekend"]
    
    # Drop warm-up rows (first 28 days have NaN for lag_28)
    initial_len = len(df)
    df = df.dropna(subset=["lag_28", "rolling_mean_28"]).reset_index(drop=True)
    print(f"[INFO] Dropped {initial_len - len(df)} warm-up rows containing NaNs. Remaining rows: {len(df)}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Feature engineering finished. Features saved to {output_path}")
    return df

if __name__ == "__main__":
    create_features()

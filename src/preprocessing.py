"""
Module: preprocessing.py
Description: Cleans, audits, unbiases censored demand (stockout correction),
             handles missing timestamps, and formats types for modeling.
"""

import pandas as pd
import numpy as np
import os

def clean_retail_data(
    input_path: str = "data/raw/retail_sales_raw.csv",
    output_path: str = "data/processed/retail_sales_cleaned.csv"
) -> pd.DataFrame:
    """
    Cleans raw retail transaction data and applies retail demand unbiasing.
    
    Parameters:
    -----------
    input_path : str, path to raw CSV
    output_path : str, path to cleaned output CSV
    
    Returns:
    --------
    pd.DataFrame: Cleaned and structured DataFrame
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found at {input_path}. Run data_generator.py first.")
        
    df = pd.read_csv(input_path)
    
    # 1. Parse Dates and sort
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["store_id", "sku_id", "date"]).reset_index(drop=True)
    
    # 2. Check and handle missing values
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print(f"[INFO] Found null values:\n{null_counts[null_counts > 0]}")
        # Numerical columns filled with median, categoricals with mode
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        cat_cols = df.select_dtypes(include=["object"]).columns
        for col in cat_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
            
    # 3. Demand Unbiasing / Censored Demand Correction (Critical Retail Concept)
    # When stockout_flag == 1, units_sold is truncated by inventory exhaustion.
    # To train an accurate forecasting model on true customer demand, we unbias
    # censored days using unconstrained demand or rolling median when unconstrained demand is unavailable.
    if "unconstrained_demand" in df.columns:
        df["target_demand"] = df["unconstrained_demand"]
    else:
        # Fallback if raw demand wasn't logged: estimate uncensored demand via rolling 14-day median
        df["rolling_normal_demand"] = df.groupby(["store_id", "sku_id"])["units_sold"].transform(
            lambda s: s.rolling(14, min_periods=3).median()
        )
        df["target_demand"] = np.where(
            df["stockout_flag"] == 1,
            np.maximum(df["units_sold"], df["rolling_normal_demand"].fillna(df["units_sold"])),
            df["units_sold"]
        )
        df.drop(columns=["rolling_normal_demand"], inplace=True)
        
    # 4. Outlier Capping (IQR Method per SKU)
    def cap_outliers(group):
        q25 = group["target_demand"].quantile(0.25)
        q75 = group["target_demand"].quantile(0.75)
        iqr = q75 - q25
        upper_limit = q75 + 3.0 * iqr # Using 3.0 for extreme outliers to preserve holiday surges
        group["target_demand_clipped"] = group["target_demand"].clip(lower=0, upper=upper_limit)
        return group
        
    df = df.groupby(["store_id", "sku_id"], group_keys=False).apply(cap_outliers)
    
    # 5. Type Casting for Performance
    df["store_id"] = df["store_id"].astype("category")
    df["sku_id"] = df["sku_id"].astype("category")
    df["category"] = df["category"].astype("category")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Preprocessing completed. Cleaned dataset saved to {output_path} ({df.shape[0]} rows)")
    return df

if __name__ == "__main__":
    clean_retail_data()

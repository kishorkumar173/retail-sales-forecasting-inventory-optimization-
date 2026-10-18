"""
Module: forecaster.py
Description: Trains an industry-grade LightGBM / Gradient Boosting multi-SKU retail
             demand forecasting model using temporal validation splits, computes
             business evaluation metrics (WAPE, RMSE, MAE), and extracts feature importances.
"""

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Try importing LightGBM; fallback to HistGradientBoostingRegressor if LightGBM not yet installed
try:
    import lightgbm as lgb
    USE_LIGHTGBM = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingRegressor
    USE_LIGHTGBM = False

def calculate_wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Weighted Absolute Percentage Error (WAPE) - Retail industry gold standard.
    WAPE = sum(|Actual - Predicted|) / sum(Actual) * 100
    Avoids MAPE's division-by-zero flaw on intermittent zero-demand days.
    """
    actual_sum = np.sum(actual)
    if actual_sum == 0:
        return 0.0
    return float(np.sum(np.abs(actual - predicted)) / actual_sum * 100)

def train_and_evaluate(
    input_path: str = "data/processed/retail_features.csv",
    test_days: int = 60,
    model_output_path: str = "models/retail_forecaster.joblib",
    predictions_output_path: str = "data/processed/forecast_predictions.csv",
    plots_dir: str = "outputs/figures"
):
    """
    Trains the forecasting model using temporal train-test split.
    """
    df = pd.read_csv(input_path)
    df["date"] = pd.to_datetime(df["date"])
    
    target_col = "target_demand_clipped" if "target_demand_clipped" in df.columns else "units_sold"
    
    # 1. Temporal Cutoff Split (No random shuffling in time-series!)
    max_date = df["date"].max()
    split_date = max_date - pd.Timedelta(days=test_days)
    
    print(f"[INFO] Time-based split: Training up to {split_date.strftime('%Y-%m-%d')}, Testing from {(split_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
    
    # Feature columns
    feature_cols = [
        "day_of_week", "day_of_month", "month", "quarter", "week_of_year",
        "is_weekend", "is_month_start", "is_month_end",
        "sin_day_of_week", "cos_day_of_week", "sin_month", "cos_month",
        "lag_1", "lag_2", "lag_7", "lag_14", "lag_21", "lag_28",
        "rolling_mean_7", "rolling_std_7", "rolling_max_7", "rolling_min_7",
        "rolling_mean_14", "rolling_std_14", "rolling_mean_28", "rolling_std_28",
        "ema_7", "ratio_lag1_to_roll7", "ratio_roll7_to_roll28",
        "price_discount_ratio", "is_promo_weekend", "on_promotion", "discount_pct",
        "lead_time_days"
    ]
    
    # Encode categorical features for LightGBM
    cat_cols = ["store_id", "sku_id", "category"]
    for col in cat_cols:
        df[col] = df[col].astype("category")
        feature_cols.append(col)
        
    train_mask = df["date"] <= split_date
    test_mask = df["date"] > split_date
    
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    print(f"[INFO] Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    
    # 2. Model Training
    if USE_LIGHTGBM:
        print("[INFO] Training LightGBM Regressor (Industry Standard)...")
        model = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=6,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
        )
    else:
        print("[INFO] LightGBM not available, training HistGradientBoostingRegressor...")
        model = HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=42
        )
        # Convert categories to integers for HistGradientBoosting
        for col in cat_cols:
            X_train[col] = X_train[col].cat.codes
            X_test[col] = X_test[col].cat.codes
        model.fit(X_train, y_train)
        
    # 3. Model Predictions & Post-Processing (Demand cannot be negative)
    test_preds = np.maximum(0, model.predict(X_test))
    test_df["predicted_demand"] = test_preds
    
    # 4. Benchmark against Naive Baseline (Lag 7: Same day last week)
    naive_preds = test_df["lag_7"].values
    
    # 5. Metrics Computation
    rmse_model = np.sqrt(mean_squared_error(y_test, test_preds))
    mae_model = mean_absolute_error(y_test, test_preds)
    wape_model = calculate_wape(y_test.values, test_preds)
    r2_model = r2_score(y_test, test_preds)
    
    rmse_naive = np.sqrt(mean_squared_error(y_test, naive_preds))
    wape_naive = calculate_wape(y_test.values, naive_preds)
    
    print("=" * 65)
    print("FORECASTING PERFORMANCE EVALUATION")
    print("=" * 65)
    print(f"ML Model (LightGBM)   -> RMSE: {rmse_model:.2f} | MAE: {mae_model:.2f} | WAPE: {wape_model:.2f}% | R2: {r2_model:.4f}")
    print(f"Naive Baseline (Lag-7)-> RMSE: {rmse_naive:.2f} | WAPE: {wape_naive:.2f}%")
    print(f"Accuracy Improvement  -> {((wape_naive - wape_model) / wape_naive * 100):.2f}% error reduction over baseline")
    print("=" * 65)
    
    # 6. Save Model and Predictions
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model, model_output_path)
    print(f"[SUCCESS] Model artifact saved -> {model_output_path}")
    
    os.makedirs(os.path.dirname(predictions_output_path), exist_ok=True)
    test_df.to_csv(predictions_output_path, index=False)
    print(f"[SUCCESS] Predictions table exported -> {predictions_output_path}")
    
    # 7. Generate Feature Importance Plot
    os.makedirs(plots_dir, exist_ok=True)
    if USE_LIGHTGBM and hasattr(model, "feature_importances_"):
        plt.figure(figsize=(10, 8))
        feat_imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(20)
        sns.barplot(x=feat_imp.values, y=feat_imp.index, palette="viridis")
        plt.title("Top 20 Feature Importances - Retail Demand Forecaster")
        plt.xlabel("Importance Score")
        plt.tight_layout()
        feat_imp_path = os.path.join(plots_dir, "feature_importance.png")
        plt.savefig(feat_imp_path, dpi=300)
        plt.close()
        print(f"[SUCCESS] Feature importance chart saved -> {feat_imp_path}")
        
    return model, test_df

if __name__ == "__main__":
    train_and_evaluate()

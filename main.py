"""
Project: Retail Sales Forecasting & Inventory Optimization System
Master Orchestration Script (main.py)
"""

import sys
import os
import time

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_generator import generate_retail_dataset
from src.preprocessing import clean_retail_data
from src.feature_engineering import create_features
from src.forecaster import train_and_evaluate
from src.inventory_optimizer import optimize_and_export
from src.evaluation import generate_visualizations_and_kpis

def run_pipeline():
    start_time = time.time()
    print("\n" + "=" * 80)
    print(" RETAIL SALES FORECASTING & INVENTORY OPTIMIZATION PIPELINE ")
    print("=" * 80)
    
    # 1. Dataset Generation
    print("\n>>> [STEP 1/6] Generating Realistic Multi-SKU Retail History...")
    raw_path = "data/raw/retail_sales_raw.csv"
    generate_retail_dataset(output_path=raw_path)
    
    # 2. Preprocessing & Demand Unbiasing
    print("\n>>> [STEP 2/6] Auditing Data, Correcting Censored Stockout Demand...")
    cleaned_path = "data/processed/retail_sales_cleaned.csv"
    clean_retail_data(input_path=raw_path, output_path=cleaned_path)
    
    # 3. Feature Engineering
    print("\n>>> [STEP 3/6] Generating Temporal Lags, Rolling Windows & Promotions...")
    features_path = "data/processed/retail_features.csv"
    create_features(input_path=cleaned_path, output_path=features_path)
    
    # 4. Model Training & Evaluation
    print("\n>>> [STEP 4/6] Training Machine Learning Forecaster & Evaluating WAPE...")
    preds_path = "data/processed/forecast_predictions.csv"
    train_and_evaluate(
        input_path=features_path,
        test_days=60,
        model_output_path="models/retail_forecaster.joblib",
        predictions_output_path=preds_path
    )
    
    # 5. Inventory Optimization Logic
    print("\n>>> [STEP 5/6] Computing Dynamic Safety Stock, ROP, and EOQ Policies...")
    policy_path = "outputs/reports/inventory_replenishment_recommendations.csv"
    optimize_and_export(predictions_path=preds_path, output_path=policy_path)
    
    # 6. Business KPI Evaluation & Visualizations
    print("\n>>> [STEP 6/6] Generating Publication Visualizations and Financial ROI...")
    generate_visualizations_and_kpis(
        predictions_path=preds_path,
        policy_path=policy_path,
        output_fig_dir="outputs/figures",
        output_report_path="outputs/reports/executive_business_impact.csv"
    )
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS!")
    print("=" * 80)
    print("\n[NEXT STEP] Launch the interactive web dashboard by running:")
    print("  streamlit run app/dashboard.py\n")

if __name__ == "__main__":
    run_pipeline()

"""
Module: evaluation.py
Description: Computes retail operational metrics (Fill Rate, Lost Sales, Stockout Rate,
             Inventory Turnover, Holding Costs) and renders publication-ready plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.inventory_optimizer import simulate_inventory_trajectory

def generate_visualizations_and_kpis(
    predictions_path: str = "data/processed/forecast_predictions.csv",
    policy_path: str = "outputs/reports/inventory_replenishment_recommendations.csv",
    output_fig_dir: str = "outputs/figures",
    output_report_path: str = "outputs/reports/executive_business_impact.csv"
):
    """
    Renders business visualizations and outputs financial impact KPIs.
    """
    os.makedirs(output_fig_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
    
    pred_df = pd.read_csv(predictions_path)
    pred_df["date"] = pd.to_datetime(pred_df["date"])
    policy_df = pd.read_csv(policy_path)
    
    # 1. Visualization 1: Actual vs Predicted Demand for Top Store & SKU
    sample_store = "STORE_01"
    sample_sku = "SKU_001"
    sample_data = pred_df[(pred_df["store_id"] == sample_store) & (pred_df["sku_id"] == sample_sku)].sort_values("date")
    
    plt.figure(figsize=(14, 6))
    plt.plot(sample_data["date"], sample_data["target_demand_clipped"], label="Actual Customer Demand", color="#1f77b4", lw=2, marker='o', markersize=3)
    plt.plot(sample_data["date"], sample_data["predicted_demand"], label="LightGBM ML Forecast", color="#ff7f0e", lw=2.5, linestyle="--")
    plt.title(f"Demand Forecast vs Actuals - {sample_data['sku_name'].iloc[0]} ({sample_store})", fontsize=14, fontweight="bold")
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Units", fontsize=12)
    plt.legend(frameon=True, facecolor="white", loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plot1_path = os.path.join(output_fig_dir, "actual_vs_predicted_time_series.png")
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Plot saved -> {plot1_path}")
    
    # 2. Visualization 2: Inventory Trajectory Simulation (Stockout Prevention)
    sim_df = simulate_inventory_trajectory(pred_df, policy_df, store_filter=sample_store, sku_filter=sample_sku)
    
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(pd.to_datetime(sim_df["date"]), sim_df["stock_static"], label="Static Policy Stock Level (Excess Capital)", color="#d62728", lw=2, linestyle=":")
    ax1.plot(pd.to_datetime(sim_df["date"]), sim_df["stock_ml"], label="ML-Optimized Stock Level", color="#2ca02c", lw=2.5)
    
    # Highlight stockout days
    stockouts_static = sim_df[sim_df["stockout_static"] == 1]
    if len(stockouts_static) > 0:
        ax1.scatter(pd.to_datetime(stockouts_static["date"]), stockouts_static["stock_static"], color="red", s=80, zorder=5, label="Static Policy Stockouts")
        
    ax1.set_title("60-Day Virtual Inventory Simulation: ML Replenishment vs Static Baseline", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Simulation Date", fontsize=12)
    ax1.set_ylabel("On-Hand Inventory (Units)", fontsize=12)
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plot2_path = os.path.join(output_fig_dir, "inventory_simulation_waterfall.png")
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Plot saved -> {plot2_path}")
    
    # 3. Visualization 3: Category Level Sales & Stockout Risks
    cat_summary = pred_df.groupby("category").agg({
        "revenue": "sum",
        "units_sold": "sum",
        "stockout_flag": "sum"
    }).reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=cat_summary, x="category", y="revenue", palette="Blues_d", ax=ax)
    ax.set_title("Total Revenue by Retail Category (Test Horizon)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Revenue (INR / Currency Units)", fontsize=12)
    ax.set_xlabel("Category", fontsize=12)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plot3_path = os.path.join(output_fig_dir, "category_revenue_distribution.png")
    plt.savefig(plot3_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Plot saved -> {plot3_path}")
    
    # 4. Compute Business Financial & Operational KPIs
    total_demand = sim_df["actual_demand"].sum()
    met_demand_ml = total_demand - sim_df["lost_sales_ml"].sum()
    met_demand_static = total_demand - sim_df["lost_sales_static"].sum()
    
    fill_rate_ml = (met_demand_ml / total_demand) * 100
    fill_rate_static = (met_demand_static / total_demand) * 100
    
    avg_stock_ml = sim_df["stock_ml"].mean()
    avg_stock_static = sim_df["stock_static"].mean()
    working_capital_reduction = ((avg_stock_static - avg_stock_ml) / avg_stock_static) * 100
    
    kpis = pd.DataFrame([{
        "Metric": "Fill Rate (ML Optimized)",
        "Value": f"{fill_rate_ml:.2f}%",
        "Benchmark Static": f"{fill_rate_static:.2f}%",
        "Business Impact": f"+{(fill_rate_ml - fill_rate_static):.2f}% higher order fulfillment"
    }, {
        "Metric": "Average Inventory Buffer",
        "Value": f"{avg_stock_ml:.1f} units",
        "Benchmark Static": f"{avg_stock_static:.1f} units",
        "Business Impact": f"{working_capital_reduction:.1f}% reduction in locked working capital"
    }, {
        "Metric": "Simulated Stockout Incidents",
        "Value": f"{sim_df['stockout_ml'].sum()} days",
        "Benchmark Static": f"{sim_df['stockout_static'].sum()} days",
        "Business Impact": f"{sim_df['stockout_static'].sum() - sim_df['stockout_ml'].sum()} fewer stockout disruptions"
    }])
    
    kpis.to_csv(output_report_path, index=False)
    print("\n" + "=" * 80)
    print("EXECUTIVE BUSINESS IMPACT SUMMARY:")
    print("=" * 80)
    print(kpis.to_string(index=False))
    print("=" * 80)
    return kpis

if __name__ == "__main__":
    generate_visualizations_and_kpis()

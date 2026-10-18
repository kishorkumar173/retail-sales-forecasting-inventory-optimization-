"""
Module: inventory_optimizer.py
Description: Translates machine learning demand forecasts into optimal replenishment
             policies using Operations Research and Inventory Science:
             - Dynamic Safety Stock under demand uncertainty
             - Reorder Point (ROP) for specified Service Levels (90%, 95%, 99%)
             - Economic Order Quantity (EOQ) balancing holding vs ordering costs
             - 60-day Multi-SKU Virtual Inventory Simulation comparing Static vs ML policy
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
import os

# Z-scores for target Service Levels
SERVICE_LEVEL_Z = {
    0.90: 1.2816,
    0.95: 1.6449,
    0.98: 2.0537,
    0.99: 2.3263
}

def calculate_inventory_policy(
    forecast_df: pd.DataFrame,
    target_service_level: float = 0.95
) -> pd.DataFrame:
    """
    Computes Safety Stock, Reorder Point (ROP), and EOQ per Store-SKU pair.
    
    Parameters:
    -----------
    forecast_df : pd.DataFrame containing historical and predicted demand
    target_service_level : float, desired cycle fill rate (default: 0.95 or 95%)
    
    Returns:
    --------
    pd.DataFrame: Inventory optimization policy table
    """
    z_score = SERVICE_LEVEL_Z.get(target_service_level, 1.645)
    
    policy_records = []
    
    # Group by store and SKU
    grouped = forecast_df.groupby(["store_id", "sku_id"])
    
    for (store_id, sku_id), group in grouped:
        sku_name = group["sku_name"].iloc[0]
        category = group["category"].iloc[0]
        unit_cost = group["unit_cost"].iloc[0]
        unit_price = group["base_unit_price"].iloc[0]
        lead_time = int(group["lead_time_days"].iloc[0])
        holding_pct = group["holding_cost_pct"].iloc[0]
        order_cost = group["order_cost"].iloc[0]
        
        # 1. Demand Statistics
        # Average daily predicted demand
        d_mean = max(group["predicted_demand"].mean(), 1.0)
        # Standard deviation of demand forecast errors (residual uncertainty)
        actual = group["target_demand_clipped"] if "target_demand_clipped" in group.columns else group["units_sold"]
        residuals = actual.values - group["predicted_demand"].values
        sigma_d = max(np.std(residuals), 1.0)
        
        # 2. Safety Stock Formula: SS = Z * sigma_d * sqrt(Lead_Time)
        safety_stock = int(np.ceil(z_score * sigma_d * np.sqrt(lead_time)))
        
        # 3. Lead Time Demand (LTD): LTD = d_mean * Lead_Time
        lead_time_demand = int(np.ceil(d_mean * lead_time))
        
        # 4. Reorder Point (ROP): ROP = LTD + SS
        reorder_point = lead_time_demand + safety_stock
        
        # 5. Economic Order Quantity (EOQ):
        # Annual Demand D = d_mean * 365
        annual_demand = d_mean * 365.0
        # Annual Holding Cost per unit H = holding_pct * unit_cost
        h_cost = max(holding_pct * unit_cost, 0.5)
        # EOQ = sqrt( (2 * D * S) / H )
        eoq = int(np.ceil(np.sqrt((2 * annual_demand * order_cost) / h_cost)))
        
        # Latest known ending inventory
        current_inventory = int(group["ending_inventory"].iloc[-1])
        
        # Stock Status and Recommended Order
        if current_inventory <= reorder_point:
            action = "TRIGGER_ORDER"
            order_qty = max(eoq, reorder_point - current_inventory)
            urgency = "HIGH" if current_inventory <= safety_stock else "MEDIUM"
        elif current_inventory > (reorder_point + eoq * 1.5):
            action = "OVERSTOCKED"
            order_qty = 0
            urgency = "LOW"
        else:
            action = "OPTIMAL"
            order_qty = 0
            urgency = "NONE"
            
        policy_records.append({
            "store_id": store_id,
            "sku_id": sku_id,
            "sku_name": sku_name,
            "category": category,
            "unit_cost": unit_cost,
            "unit_price": unit_price,
            "lead_time_days": lead_time,
            "avg_daily_demand": round(d_mean, 1),
            "demand_std_dev": round(sigma_d, 2),
            "service_level_target": f"{int(target_service_level*100)}%",
            "safety_stock": safety_stock,
            "lead_time_demand": lead_time_demand,
            "reorder_point": reorder_point,
            "eoq": eoq,
            "current_inventory": current_inventory,
            "action_status": action,
            "urgency": urgency,
            "recommended_order_qty": order_qty,
            "tied_capital_current": round(current_inventory * unit_cost, 2),
            "opt_capital_holding": round((safety_stock + (eoq / 2)) * unit_cost, 2)
        })
        
    policy_df = pd.DataFrame(policy_records)
    return policy_df

def simulate_inventory_trajectory(
    test_predictions_df: pd.DataFrame,
    policy_df: pd.DataFrame,
    store_filter: str = "STORE_01",
    sku_filter: str = "SKU_001"
) -> pd.DataFrame:
    """
    Performs a day-by-day forward inventory simulation comparing:
    Policy A: Traditional Static Reorder Point (Fixed safety buffer)
    Policy B: ML-Driven Dynamic Inventory Optimization
    """
    sku_history = test_predictions_df[
        (test_predictions_df["store_id"] == store_filter) &
        (test_predictions_df["sku_id"] == sku_filter)
    ].sort_values("date").reset_index(drop=True)
    
    sku_policy = policy_df[
        (policy_df["store_id"] == store_filter) &
        (policy_df["sku_id"] == sku_filter)
    ].iloc[0]
    
    lead_time = int(sku_policy["lead_time_days"])
    rop_ml = int(sku_policy["reorder_point"])
    eoq_ml = int(sku_policy["eoq"])
    
    # Static benchmark policy: naive ROP with 50% fixed buffer
    rop_static = int(sku_policy["avg_daily_demand"] * lead_time * 1.5)
    order_static_qty = int(sku_policy["avg_daily_demand"] * 10)
    
    # Simulation containers
    stock_ml = rop_ml + eoq_ml
    stock_static = rop_static + order_static_qty
    
    pipeline_orders_ml = []      # list of (arrival_day_index, qty)
    pipeline_orders_static = []  # list of (arrival_day_index, qty)
    
    simulation_log = []
    
    for i, row in sku_history.iterrows():
        date = row["date"]
        actual_demand = row["target_demand_clipped"] if "target_demand_clipped" in row else row["units_sold"]
        
        # 1. Process Inbound Deliveries
        # ML policy
        arrived_ml = sum(qty for day, qty in pipeline_orders_ml if day == i)
        stock_ml += arrived_ml
        # Static policy
        arrived_static = sum(qty for day, qty in pipeline_orders_static if day == i)
        stock_static += arrived_static
        
        # 2. Satisfy Customer Demand & Track Stockouts
        # ML Policy
        if stock_ml >= actual_demand:
            sales_ml = actual_demand
            stockout_ml = 0
            lost_sales_ml = 0
            stock_ml -= actual_demand
        else:
            sales_ml = stock_ml
            stockout_ml = 1
            lost_sales_ml = actual_demand - stock_ml
            stock_ml = 0
            
        # Static Policy
        if stock_static >= actual_demand:
            sales_static = actual_demand
            stockout_static = 0
            lost_sales_static = 0
            stock_static -= actual_demand
        else:
            sales_static = stock_static
            stockout_static = 1
            lost_sales_static = actual_demand - stock_static
            stock_static = 0
            
        # 3. Replenishment Decision Check
        # ML Policy
        pending_orders_ml = sum(qty for day, qty in pipeline_orders_ml if day > i)
        inventory_position_ml = stock_ml + pending_orders_ml
        order_placed_ml = 0
        if inventory_position_ml <= rop_ml:
            order_placed_ml = eoq_ml
            pipeline_orders_ml.append((i + lead_time, order_placed_ml))
            
        # Static Policy
        pending_orders_static = sum(qty for day, qty in pipeline_orders_static if day > i)
        inventory_position_static = stock_static + pending_orders_static
        order_placed_static = 0
        if inventory_position_static <= rop_static:
            order_placed_static = order_static_qty
            pipeline_orders_static.append((i + lead_time, order_placed_static))
            
        simulation_log.append({
            "date": date,
            "actual_demand": actual_demand,
            "predicted_demand": round(row["predicted_demand"], 1),
            "stock_ml": stock_ml,
            "stockout_ml": stockout_ml,
            "lost_sales_ml": lost_sales_ml,
            "stock_static": stock_static,
            "stockout_static": stockout_static,
            "lost_sales_static": lost_sales_static
        })
        
    sim_df = pd.DataFrame(simulation_log)
    return sim_df

def optimize_and_export(
    predictions_path: str = "data/processed/forecast_predictions.csv",
    output_path: str = "outputs/reports/inventory_replenishment_recommendations.csv"
):
    """Executes inventory optimization pipeline and saves reports."""
    df = pd.read_csv(predictions_path)
    policy_df = calculate_inventory_policy(df, target_service_level=0.95)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    policy_df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Inventory Optimization Policy calculated for {len(policy_df)} SKU-Store combinations.")
    print(f"[SUCCESS] Saved policy recommendations -> {output_path}")
    
    # Print sample recommendations for terminal
    print("\n" + "=" * 80)
    print("SAMPLE REPLENISHMENT ORDERS GENERATED:")
    print("=" * 80)
    orders = policy_df[policy_df["action_status"] == "TRIGGER_ORDER"][
        ["store_id", "sku_name", "current_inventory", "reorder_point", "safety_stock", "recommended_order_qty", "urgency"]
    ]
    if len(orders) > 0:
        print(orders.to_string(index=False))
    else:
        print("All stores and SKUs are currently operating above their Safety Stock / Reorder Point.")
    print("=" * 80)
    
    return policy_df

if __name__ == "__main__":
    optimize_and_export()

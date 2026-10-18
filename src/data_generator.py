"""
Module: data_generator.py
Description: Generates high-fidelity synthetic multi-store, multi-SKU retail sales
             and inventory transaction datasets simulating real-world demand patterns,
             promotions, seasonality, lead times, and stockouts.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

def generate_retail_dataset(
    start_date: str = "2023-01-01",
    end_date: str = "2025-12-31",
    random_seed: int = 42,
    output_path: str = "data/raw/retail_sales_raw.csv"
) -> pd.DataFrame:
    """
    Generates a realistic multi-store, multi-SKU retail dataset.
    
    Parameters:
    -----------
    start_date : str, start date of sales history (YYYY-MM-DD)
    end_date : str, end date of sales history (YYYY-MM-DD)
    random_seed : int, reproducible seed
    output_path : str, path to save CSV
    
    Returns:
    --------
    pd.DataFrame: Simulated retail dataset
    """
    np.random.seed(random_seed)
    
    # 1. Define Product Catalog (SKU Level Master Data)
    products = [
        {
            "sku_id": "SKU_001",
            "sku_name": "Premium Basmati Rice 5kg",
            "category": "Groceries & Staples",
            "base_demand": 45,
            "unit_price": 499.0,
            "unit_cost": 360.0,
            "lead_time_days": 5,
            "holding_cost_pct": 0.18, # 18% annual holding cost
            "order_cost": 500.0       # Fixed cost per purchase order
        },
        {
            "sku_id": "SKU_002",
            "sku_name": "Organic Whole Milk 1L",
            "category": "Dairy & Chilled",
            "base_demand": 85,
            "unit_price": 72.0,
            "unit_cost": 52.0,
            "lead_time_days": 2,
            "holding_cost_pct": 0.25, # High holding/spoilage cost
            "order_cost": 150.0
        },
        {
            "sku_id": "SKU_003",
            "sku_name": "Instant Masala Noodles 12-Pack",
            "category": "Packaged Snacks",
            "base_demand": 60,
            "unit_price": 140.0,
            "unit_cost": 95.0,
            "lead_time_days": 4,
            "holding_cost_pct": 0.15,
            "order_cost": 300.0
        },
        {
            "sku_id": "SKU_004",
            "sku_name": "Sparkling Apple Juice 750ml",
            "category": "Beverages",
            "base_demand": 30,
            "unit_price": 165.0,
            "unit_cost": 110.0,
            "lead_time_days": 7,
            "holding_cost_pct": 0.16,
            "order_cost": 350.0
        },
        {
            "sku_id": "SKU_005",
            "sku_name": "Herbal Anti-Dandruff Shampoo 400ml",
            "category": "Personal Care",
            "base_demand": 25,
            "unit_price": 340.0,
            "unit_cost": 210.0,
            "lead_time_days": 10,
            "holding_cost_pct": 0.12,
            "order_cost": 400.0
        },
        {
            "sku_id": "SKU_006",
            "sku_name": "Whole Wheat Atta 10kg",
            "category": "Groceries & Staples",
            "base_demand": 55,
            "unit_price": 425.0,
            "unit_cost": 310.0,
            "lead_time_days": 6,
            "holding_cost_pct": 0.18,
            "order_cost": 450.0
        },
        {
            "sku_id": "SKU_007",
            "sku_name": "Cold-Pressed Sunflower Oil 1L",
            "category": "Groceries & Staples",
            "base_demand": 40,
            "unit_price": 190.0,
            "unit_cost": 135.0,
            "lead_time_days": 6,
            "holding_cost_pct": 0.17,
            "order_cost": 400.0
        },
        {
            "sku_id": "SKU_008",
            "sku_name": "Dark Roasted Coffee Beans 500g",
            "category": "Beverages",
            "base_demand": 20,
            "unit_price": 450.0,
            "unit_cost": 290.0,
            "lead_time_days": 8,
            "holding_cost_pct": 0.14,
            "order_cost": 350.0
        }
    ]
    
    # 2. Define Stores
    stores = [
        {"store_id": "STORE_01", "store_name": "Metro Mega Hypermarket", "traffic_factor": 1.25},
        {"store_id": "STORE_02", "store_name": "Suburban Express Market", "traffic_factor": 0.90},
        {"store_id": "STORE_03", "store_name": "Downtown City Center", "traffic_factor": 1.10}
    ]
    
    # 3. Generate Date Range
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    
    records = []
    
    for store in stores:
        for product in products:
            sku_id = product["sku_id"]
            base_demand = product["base_demand"] * store["traffic_factor"]
            
            # Simulated starting inventory level
            current_stock = int(base_demand * 12)
            
            for date in dates:
                # Calendar features
                day_of_week = date.dayofweek  # 0=Monday, 6=Sunday
                month = date.month
                day_of_year = date.dayofyear
                
                # Seasonality factors:
                # A) Weekend effect: Retail sales climb Friday-Sunday
                weekend_factor = 1.35 if day_of_week in [4, 5, 6] else 0.88
                
                # B) Annual seasonality: Festive peak in Oct-Nov (Diwali/New Year), Summer beverage peak
                if product["category"] == "Beverages" and month in [4, 5, 6]:
                    season_factor = 1.50
                elif month in [10, 11, 12]:
                    season_factor = 1.30 # Festive retail boost
                elif month in [7, 8]:
                    season_factor = 0.92 # Monsoon lull
                else:
                    season_factor = 1.00
                
                # C) Promotional campaign simulation (random ~15% chance of promo)
                on_promotion = 1 if np.random.rand() < 0.14 else 0
                promo_discount = np.random.choice([0.10, 0.15, 0.20, 0.25]) if on_promotion else 0.0
                promo_lift = (1.0 + (promo_discount * 2.8)) if on_promotion else 1.0
                
                # D) True underlying customer demand with Poisson/Normal stochastic variation
                expected_demand = base_demand * weekend_factor * season_factor * promo_lift
                raw_demand = np.random.poisson(lam=max(expected_demand, 1.0))
                
                # E) Inventory constrained sales simulation (Stockout logic):
                # If current_stock >= raw_demand, sales = raw_demand.
                # If current_stock < raw_demand, stockout occurs: sales = current_stock, lost_sales = raw_demand - current_stock
                if current_stock >= raw_demand:
                    actual_sales = raw_demand
                    lost_sales = 0
                    stockout_flag = 0
                    current_stock -= actual_sales
                else:
                    actual_sales = max(0, current_stock)
                    lost_sales = raw_demand - actual_sales
                    stockout_flag = 1
                    current_stock = 0
                
                # Simulated replenishment logic to prevent perpetual stockout
                # When stock drops below 4 days of base demand, an inbound shipment arrives after lead time
                if current_stock < (base_demand * 4) and np.random.rand() < 0.35:
                    replenishment_qty = int(base_demand * 10)
                    current_stock += replenishment_qty
                
                selling_price = round(product["unit_price"] * (1.0 - promo_discount), 2)
                revenue = round(actual_sales * selling_price, 2)
                cogs = round(actual_sales * product["unit_cost"], 2)
                gross_profit = round(revenue - cogs, 2)
                
                records.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "store_id": store["store_id"],
                    "store_name": store["store_name"],
                    "sku_id": product["sku_id"],
                    "sku_name": product["sku_name"],
                    "category": product["category"],
                    "unit_cost": product["unit_cost"],
                    "base_unit_price": product["unit_price"],
                    "selling_price": selling_price,
                    "on_promotion": on_promotion,
                    "discount_pct": promo_discount,
                    "lead_time_days": product["lead_time_days"],
                    "holding_cost_pct": product["holding_cost_pct"],
                    "order_cost": product["order_cost"],
                    "unconstrained_demand": raw_demand,
                    "units_sold": actual_sales,
                    "lost_sales": lost_sales,
                    "stockout_flag": stockout_flag,
                    "ending_inventory": current_stock,
                    "revenue": revenue,
                    "gross_profit": gross_profit
                })
                
    df = pd.DataFrame(records)
    
    # Ensure directory exists and export
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Synthetic retail dataset generated: {df.shape[0]} rows, {df.shape[1]} columns -> {output_path}")
    return df

if __name__ == "__main__":
    generate_retail_dataset()

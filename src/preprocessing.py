
import pandas as pd
import numpy as np

def generate_synthetic_data(n_products=5, n_days=730, seed=42):
    """
    Generate synthetic daily retail sales data.
    Simulates seasonality, trend, and noise per product.
    """
    np.random.seed(seed)
    dates = pd.date_range(start="2022-01-01", periods=n_days, freq="D")
    records = []

    for pid in range(1, n_products + 1):
        trend = np.linspace(100, 150, n_days)
        seasonality = 20 * np.sin(2 * np.pi * np.arange(n_days) / 365)
        weekly = 10 * np.sin(2 * np.pi * np.arange(n_days) / 7)
        noise = np.random.normal(0, 8, n_days)
        demand = trend + seasonality + weekly + noise + (pid * 10)
        demand = np.clip(demand, 0, None).round().astype(int)

        for i, d in enumerate(dates):
            records.append({
                "date": d,
                "product_id": f"P{pid:03d}",
                "demand": demand[i],
                "price": round(np.random.uniform(10, 100), 2),
                "promotion": int(np.random.rand() < 0.15)
            })

    df = pd.DataFrame(records)
    df.sort_values(["product_id", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def preprocess(df):
    """Clean and cast types."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["demand"] = df["demand"].clip(lower=0)
    df.drop_duplicates(subset=["date", "product_id"], inplace=True)
    df.sort_values(["product_id", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

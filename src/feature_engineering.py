
import pandas as pd
import numpy as np

def add_time_features(df):
    """Extract calendar features from date."""
    df = df.copy()
    df["day_of_week"]  = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"]        = df["date"].dt.month
    df["quarter"]      = df["date"].dt.quarter
    df["is_weekend"]   = (df["date"].dt.dayofweek >= 5).astype(int)
    df["year"]         = df["date"].dt.year
    return df


def add_lag_features(df, lags=[1, 7, 14, 21, 28]):
    """Create lag features per product."""
    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby("product_id")["demand"].shift(lag)
    return df


def add_rolling_features(df, windows=[7, 14, 28]):
    """Rolling mean and std per product."""
    df = df.copy()
    for w in windows:
        df[f"rolling_mean_{w}"] = (
            df.groupby("product_id")["demand"]
            .shift(1)
            .transform(lambda x: x.rolling(w, min_periods=1).mean())
        )
        df[f"rolling_std_{w}"] = (
            df.groupby("product_id")["demand"]
            .shift(1)
            .transform(lambda x: x.rolling(w, min_periods=1).std())
        )
    return df


def add_ewm_features(df, spans=[7, 14]):
    """Exponentially weighted mean per product."""
    df = df.copy()
    for s in spans:
        df[f"ewm_mean_{s}"] = (
            df.groupby("product_id")["demand"]
            .shift(1)
            .transform(lambda x: x.ewm(span=s, adjust=False).mean())
        )
    return df


def build_features(df):
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_ewm_features(df)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

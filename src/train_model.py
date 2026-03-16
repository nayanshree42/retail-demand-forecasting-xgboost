
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import pickle, os

FEATURE_COLS = [
    "day_of_week", "day_of_month", "week_of_year", "month", "quarter",
    "is_weekend", "year", "price", "promotion",
    "lag_1", "lag_7", "lag_14", "lag_21", "lag_28",
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_28",
    "rolling_std_7", "rolling_std_14", "rolling_std_28",
    "ewm_mean_7", "ewm_mean_14",
    "product_id_enc"
]
TARGET = "demand"


def encode_product(df):
    df = df.copy()
    le = LabelEncoder()
    df["product_id_enc"] = le.fit_transform(df["product_id"])
    return df, le


def time_based_split(df, test_days=90):
    cutoff = df["date"].max() - pd.Timedelta(days=test_days)
    train = df[df["date"] <= cutoff].copy()
    test  = df[df["date"] >  cutoff].copy()
    return train, test


def train(df, save_path="retail-demand-forecasting-xgboost/models/xgb_model.pkl"):
    df, le = encode_product(df)
    train_df, test_df = time_based_split(df)

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET]
    X_test  = test_df[FEATURE_COLS]
    y_test  = test_df[TARGET]

    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        early_stopping_rounds=30,
        eval_metric="rmse",
        verbosity=0
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump({"model": model, "label_encoder": le, "features": FEATURE_COLS}, f)

    print(f"✅ Model saved → {save_path}")
    return model, le, train_df, test_df

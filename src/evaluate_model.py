
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import xgboost as xgb


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))


def mae(y_true, y_pred):
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))


def evaluate(model, test_df, feature_cols, target="demand"):
    X_test = test_df[feature_cols]
    y_true = test_df[target].values
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0, None)

    metrics = {
        "MAPE (%)": round(mape(y_true, y_pred), 2),
        "RMSE":     round(rmse(y_true, y_pred), 2),
        "MAE":      round(mae(y_true, y_pred), 2),
    }
    print("\n📊 Evaluation Metrics")
    print("-" * 30)
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    return y_pred, metrics


def plot_forecast(test_df, y_pred, product_id="P001"):
    fig, ax = plt.subplots(figsize=(14, 5))
    subset = test_df[test_df["product_id"] == product_id].copy()
    subset["predicted"] = y_pred[test_df["product_id"] == product_id]

    ax.plot(subset["date"], subset["demand"],    label="Actual",    color="#2c7bb6", lw=2)
    ax.plot(subset["date"], subset["predicted"], label="Predicted", color="#d7191c", lw=2, linestyle="--")
    ax.fill_between(subset["date"], subset["demand"], subset["predicted"],
                    alpha=0.1, color="gray")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.xticks(rotation=45)
    ax.set_title(f"Demand Forecast vs Actual — {product_id}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("Units Sold")
    ax.legend(); plt.tight_layout(); plt.show()


def plot_feature_importance(model, feature_cols, top_n=15):
    importances = model.feature_importances_
    feat_df = pd.DataFrame({"feature": feature_cols, "importance": importances})
    feat_df = feat_df.sort_values("importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=feat_df, x="importance", y="feature", palette="Blues_r", ax=ax)
    ax.set_title("Top Feature Importances (XGBoost)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score"); ax.set_ylabel("")
    plt.tight_layout(); plt.show()


def plot_residuals(y_true, y_pred):
    residuals = np.array(y_true) - np.array(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].hist(residuals, bins=40, color="#4575b4", edgecolor="white")
    axes[0].set_title("Residual Distribution"); axes[0].set_xlabel("Residual")
    axes[1].scatter(y_pred, residuals, alpha=0.3, color="#4575b4", s=10)
    axes[1].axhline(0, color="red", lw=1.5, linestyle="--")
    axes[1].set_title("Residuals vs Predicted"); axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Residual")
    plt.tight_layout(); plt.show()

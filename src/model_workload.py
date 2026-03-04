# src/model_workload.py
"""
Train XGBoost to predict injury_in_next_7d from workload features.
"""

import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, classification_report, precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

def train_and_evaluate_workload_model(
    data_path: str = "data/processed/athlete_workload_features.csv",
    model_save_path: str = "models/workload_xgb.json",
    fig_dir: str = "figures"
):
    """
    Train and evaluate a time-series workload-based injury risk model
    using ACWR-style features and a 7-day-ahead injury label.
    """
    print("Loading processed data...")
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])

    # Sort BEFORE building X, y to keep chronological split aligned
    df = df.sort_values('date').reset_index(drop=True)

    # Features (no leakage)
    features = [
        'game_workload', 'acute_load_7d', 'chronic_load_28d', 'acwr',
        'acwr_spike_high', 'acwr_spike_very_high',
        'hip_mobility', 'hip_mobility_7d_mean', 'hip_mobility_28d_mean',
        'hip_mobility_pct_change'
    ]
    target = 'injury_in_next_7d'

    X = df[features].fillna(0)
    y = df[target]

    # Chronological split (80/20)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"Train: {X_train.shape}, {y_train.mean():.4f} positive")
    print(f"Test: {X_test.shape}, {y_test.mean():.4f} positive")

    # XGBoost with imbalance handling
    pos = y_train.sum()
    scale_pos_weight = (len(y_train) - pos) / pos if pos > 0 else 1

    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        max_depth=4,
        learning_rate=0.1,
        n_estimators=100
    )

    print("Training model...")
    model.fit(X_train, y_train)

    # Predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nTest ROC-AUC: {auc:.4f}")

    # Classification report at default threshold 0.5
    pred_binary = (y_pred_proba >= 0.5).astype(int)
    print("\nClassification Report (threshold 0.5):")
    print(classification_report(y_test, pred_binary))

    # Find best threshold for F1
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    print(f"\nBest threshold for max F1: {best_thresh:.3f}")
    print(f"F1 at best threshold: {f1_scores[best_idx]:.4f}")

    # Feature importance
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nFeature Importance:")
    print(importance)

    # Save native XGBoost model (avoids sklearn wrapper bug)
    Path(model_save_path).parent.mkdir(exist_ok=True)
    booster = model.get_booster()
    booster.save_model(model_save_path)
    print(f"Model saved (native): {model_save_path}")

    # Plot and save feature importance
    Path(fig_dir).mkdir(exist_ok=True)
    plt.figure(figsize=(10, 6))
    importance.head(10).plot(kind='barh', x='feature', y='importance', color='skyblue')
    plt.title('Top 10 Feature Importances (XGBoost)')
    plt.xlabel('Importance')
    plt.tight_layout()
    importance_plot = Path(fig_dir) / "feature_importance.png"
    plt.savefig(importance_plot, dpi=150)
    plt.close()
    print(f"Importance plot saved: {importance_plot}")

    # Save metrics to JSON
    metrics = {
        "roc_auc": float(auc),
        "best_threshold": float(best_thresh),
        "f1_at_best_threshold": float(f1_scores[best_idx]),
        "test_positive_rate": float(y_test.mean())
    }
    Path("results").mkdir(exist_ok=True)
    metrics_path = Path("results") / "workload_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved: {metrics_path}")

    return model, auc, importance

if __name__ == "__main__":
    train_and_evaluate_workload_model()

# src/model_multimodal.py
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, classification_report
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import json
from sklearn.model_selection import train_test_split

def train_multimodal_model(
    data_path: str = "data/processed/multimodal_processed.csv",
    fig_dir: str = "figures"
):
    """
    Train and interpret a session-level multimodal injury risk model
    using physiological, biomechanical, and contextual features.
    """
    df = pd.read_csv(data_path)
    target = 'injury_risk'
    features = [col for col in df.columns if col != target]

    X = df[features]
    y = df[target]

    # 80/20 split (stratified for imbalance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"Train: {X_train.shape}, positive rate {y_train.mean():.4f}")
    print(f"Test: {X_test.shape}, positive rate {y_test.mean():.4f}")

    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

    model = xgb.XGBClassifier(
        eval_metric='auc',
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        max_depth=5,
        learning_rate=0.1,
        n_estimators=150
    )

    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nTest ROC-AUC: {auc:.4f}")

    pred_binary = (y_pred_proba >= 0.5).astype(int)
    print("\nClassification Report @ 0.5:")
    print(classification_report(y_test, pred_binary))

    # SHAP for interpretability (sample for speed)
    sample_size = min(500, len(X_test))
    X_shap = X_test.sample(n=sample_size, random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    Path(fig_dir).mkdir(exist_ok=True)
    shap.summary_plot(shap_values, X_shap, show=False)
    plt.title("SHAP Feature Importance Summary")
    shap_plot = Path(fig_dir) / "shap_summary.png"
    plt.savefig(shap_plot, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"SHAP summary saved: {shap_plot}")

    shap_importance = pd.DataFrame({
        'feature': features,
        'shap_importance': abs(shap_values).mean(0)
    }).sort_values('shap_importance', ascending=False)
    print("\nTop SHAP Features:\n", shap_importance.head(10))

    # Save metrics
    metrics = {
        "roc_auc": float(auc),
        "test_positive_rate": float(y_test.mean())
    }
    Path("results").mkdir(exist_ok=True)
    metrics_path = Path("results") / "multimodal_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved: {metrics_path}")

    return model, auc, shap_importance

if __name__ == "__main__":
    train_multimodal_model()

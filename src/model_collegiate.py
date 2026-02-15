# src/model_collegiate.py
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from pathlib import Path

def train_collegiate_model(
    data_path: str = "data/processed/collegiate_processed.csv",
    fig_dir: str = "figures"
):
    df = pd.read_csv(data_path)
    target = 'Injury_Indicator'
    features = [col for col in df.columns if col != target]

    X = df[features]
    y = df[target]

    # Stratified split (70/30 for small data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    print(f"Train: {X_train.shape}, positive rate {y_train.mean():.4f}")
    print(f"Test: {X_test.shape}, positive rate {y_test.mean():.4f}")

    model = xgb.XGBClassifier(
        eval_metric='auc',
        random_state=42,
        scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum() if y_train.sum() > 0 else 1,
        max_depth=3,
        learning_rate=0.1,
        n_estimators=100
    )

    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nTest ROC-AUC: {auc:.4f}")

    pred_binary = (y_pred_proba >= 0.5).astype(int)
    print("\nClassification Report @ 0.5:")
    print(classification_report(y_test, pred_binary))

    # Feature importance
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nFeature Importance:\n", importance.head(10))

    # Save importance plot
    plt.figure(figsize=(10, 6))
    importance.head(10).plot(kind='barh', x='feature', y='importance', color='lightgreen')
    plt.title('Top Feature Importances (XGBoost)')
    plt.xlabel('Importance')
    plt.tight_layout()
    plot_path = Path(fig_dir) / "collegiate_feature_importance.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Importance plot saved: {plot_path}")

    return model, auc, importance


if __name__ == "__main__":
    train_collegiate_model()
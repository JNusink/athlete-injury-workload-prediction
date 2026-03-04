# src/visualize_multimodal.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def visualize_multimodal(df_path: str = "data/processed/multimodal_processed.csv", fig_dir: str = "figures"):
    Path(fig_dir).mkdir(exist_ok=True)
    df = pd.read_csv(df_path)

    target = 'injury_risk'

    # 1. Risk distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x=target, data=df)
    plt.title('Injury Risk Balance')
    plt.xlabel('Injury Risk (0 = No, 1 = Yes)')
    plt.ylabel('Count')
    plt.savefig(Path(fig_dir) / "injury_risk_balance.png", dpi=150)
    plt.close()

    # 2. Boxplots for top suspected features vs risk
    key_features = ['impact_force', 'fatigue_index', 'workload_intensity', 'previous_injury_history']
    for feat in key_features:
        if feat in df.columns:
            plt.figure(figsize=(8, 5))
            sns.boxplot(x=target, y=feat, data=df)
            plt.title(f'{feat.capitalize()} by Injury Risk')
            plt.savefig(Path(fig_dir) / f"{feat}_by_risk.png", dpi=150)
            plt.close()

    # 3. Correlation heatmap (top 10 numeric features + target)
    numeric_cols = df.select_dtypes(include='number').columns[:15]  # limit for clarity
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap')
    plt.savefig(Path(fig_dir) / "correlation_heatmap.png", dpi=150)
    plt.close()

    print("Visualizations saved in figures/")


if __name__ == "__main__":
    visualize_multimodal()
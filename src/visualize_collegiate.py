# src/visualize_collegiate.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def visualize_collegiate(df_path: str = "data/raw/collegiate_athlete_injury_dataset.csv", fig_dir: str = "figures"):
    Path(fig_dir).mkdir(exist_ok=True)
    df = pd.read_csv(df_path)

    target = 'Injury_Indicator'

    # 1. Injury rate by Gender
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Gender', y=target, data=df, estimator='mean')
    plt.title('Injury Rate by Gender')
    plt.ylabel('Proportion Injured')
    plt.savefig(Path(fig_dir) / "injury_by_gender.png", dpi=150)
    plt.close()

    # 2. Injury rate by Position
    plt.figure(figsize=(10, 5))
    sns.barplot(x='Position', y=target, data=df, estimator='mean')
    plt.title('Injury Rate by Position')
    plt.ylabel('Proportion Injured')
    plt.savefig(Path(fig_dir) / "injury_by_position.png", dpi=150)
    plt.close()

    # 3. Boxplots for key risk scores
    key_scores = ['ACL_Risk_Score', 'Load_Balance_Score', 'Recovery_Days_Per_Week', 'Training_Intensity']
    for score in key_scores:
        if score in df.columns:
            plt.figure(figsize=(8, 5))
            sns.boxplot(x=target, y=score, data=df)
            plt.title(f'{score} by Injury Status')
            plt.savefig(Path(fig_dir) / f"{score.lower()}_by_injury.png", dpi=150)
            plt.close()

    print("Visualizations saved in figures/")


if __name__ == "__main__":
    visualize_collegiate()
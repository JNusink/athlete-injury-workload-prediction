# src/explore_multimodal.py
import pandas as pd

def explore_multimodal_data(path: str = "data/raw/sports_multimodal_data.csv"):
    print(f"Loading: {path}")
    df = pd.read_csv(path)

    print(f"\nShape: {df.shape}")
    print(f"Columns ({len(df.columns)} total):\n{', '.join(df.columns)}")
    print(f"\nMissing values per column:\n{df.isna().sum()[df.isna().sum() > 0]}")

    if 'injury_risk' in df.columns:
        print(f"\nInjury risk balance:\n{df['injury_risk'].value_counts(normalize=True)}")
        print(f"Positive rate: {df['injury_risk'].mean():.4f}")

    print("\nSample head:\n", df.head(5).to_string())

    # Basic stats on key numeric columns
    key_cols = ['heart_rate', 'impact_force', 'fatigue_index', 'workload_intensity', 'previous_injury_history']
    print("\nKey numeric stats:\n", df[key_cols].describe())

    return df


if __name__ == "__main__":
    explore_multimodal_data()
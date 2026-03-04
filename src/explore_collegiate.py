# src/explore_collegiate.py
import pandas as pd

def explore_collegiate_data(path: str = "data/raw/collegiate_athlete_injury_dataset.csv"):
    print(f"Loading: {path}")
    df = pd.read_csv(path)

    print(f"\nShape: {df.shape}")
    print(f"Columns:\n{', '.join(df.columns)}")
    print(f"\nMissing values:\n{df.isna().sum()[df.isna().sum() > 0]}")

    if 'Injury_Indicator' in df.columns:
        print(f"\nInjury Indicator balance:\n{df['Injury_Indicator'].value_counts(normalize=True)}")
        print(f"Positive rate: {df['Injury_Indicator'].mean():.4f}")

    print("\nCategorical summaries:")
    for col in ['Gender', 'Position']:
        if col in df.columns:
            print(f"\n{col}:\n{df[col].value_counts()}")

    print("\nNumeric stats (selected):\n")
    numeric_cols = ['Age', 'Height_cm', 'Weight_kg', 'Training_Intensity', 'Training_Hours_Per_Week',
                    'Recovery_Days_Per_Week', 'Match_Count_Per_Week', 'Rest_Between_Events_Days',
                    'Fatigue_Score', 'Performance_Score', 'Team_Contribution_Score', 'Load_Balance_Score',
                    'ACL_Risk_Score']
    print(df[numeric_cols].describe())

    print("\nSample head (first 5 rows):\n", df.head().to_string())

    return df


if __name__ == "__main__":
    explore_collegiate_data()
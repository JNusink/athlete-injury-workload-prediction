# src/process_multimodal.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path

def process_multimodal_data(
    raw_path: str = "data/raw/sports_multimodal_data.csv",
    output_path: str = "data/processed/multimodal_processed.csv"
):
    """
    Clean, scale, and engineer features for session-level multimodal
    injury risk modeling from sports_multimodal_data.csv.
    """
    print(f"Loading: {raw_path}")
    df = pd.read_csv(raw_path)

    # Drop any unnamed/index columns if present
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Basic cleaning
    # Clip extreme negative impact_force (if physical sense)
    if 'impact_force' in df.columns:
        df['impact_force'] = df['impact_force'].clip(lower=0)

    # Target
    target = 'injury_risk'
    if target not in df.columns:
        raise ValueError("Target 'injury_risk' not found!")

    # Features (exclude target)
    features = [col for col in df.columns if col != target]

    # Standardize numeric features
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    # Simple engineered features
    if 'impact_force' in df.columns and 'speed' in df.columns:
        df['impact_per_speed'] = df['impact_force'] / (df['speed'] + 1e-6)  # avoid div by zero
        features.append('impact_per_speed')

    if 'fatigue_index' in df.columns and 'workload_intensity' in df.columns:
        df['fatigue_load_interaction'] = df['fatigue_index'] * df['workload_intensity']
        features.append('fatigue_load_interaction')

    # Save
    Path(output_path).parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved processed data: {output_path} (shape: {df.shape})")
    print(f"Features used: {features}")

    return df, features, target

if __name__ == "__main__":
    process_multimodal_data()

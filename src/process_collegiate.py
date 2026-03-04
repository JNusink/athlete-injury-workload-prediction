# src/process_collegiate.py
# src/process_collegiate.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pathlib import Path

def process_collegiate_data(
    raw_path: str = "data/raw/collegiate_athlete_injury_dataset.csv",
    output_path: str = "data/processed/collegiate_processed.csv"
):
    """
    Encode and scale collegiate athlete profiles for baseline
    injury risk modeling (demographics + training + scores).
    """
    print(f"Loading: {raw_path}")
    df = pd.read_csv(raw_path)

    # Target
    target = 'Injury_Indicator'
    if target not in df.columns:
        raise ValueError("Target 'Injury_Indicator' not found!")

    # Drop ID column (not useful for modeling)
    if 'Athlete_ID' in df.columns:
        df = df.drop('Athlete_ID', axis=1)

    # Define feature types
    categorical_cols = ['Gender', 'Position']
    numeric_cols = [col for col in df.columns if col not in categorical_cols + [target]]

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
        ]
    )

    # Apply
    X_processed = preprocessor.fit_transform(df.drop(target, axis=1))
    y = df[target]

    # Get feature names after encoding
    cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
    feature_names = numeric_cols + list(cat_names)

    # Rebuild DataFrame
    df_processed = pd.DataFrame(X_processed, columns=feature_names)
    df_processed[target] = y.values

    # Save
    Path(output_path).parent.mkdir(exist_ok=True)
    df_processed.to_csv(output_path, index=False)
    print(f"Saved processed data: {output_path} (shape: {df_processed.shape})")
    print(f"Features after encoding: {df_processed.columns.tolist()}")

    return df_processed, feature_names, target

if __name__ == "__main__":
    process_collegiate_data()

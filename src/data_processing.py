# src/data_processing.py
import pandas as pd
import numpy as np
from pathlib import Path

def load_and_process_workload_data(
    raw_path: str = "data/raw/mergedData.csv",
    output_path: str = "data/processed/athlete_workload_features.csv"
) -> pd.DataFrame:
    print(f"Loading file: {Path(raw_path).absolute()}")
    
    if not Path(raw_path).exists():
        raise FileNotFoundError(f"Cannot find {raw_path}. Please check path and file name.")
    
    df = pd.read_csv(raw_path)
    print(f"Raw shape: {df.shape}")
    print("Columns:", df.columns.tolist())
    print("Sample head:\n", df.head(5).to_string())
    print("Unique athlete_ids:", sorted(df['athlete_id'].unique()))
    print("Unique metrics:", df['metric'].unique())
    print("Injury value counts:\n", df['injury'].value_counts(dropna=False))
    
    # Convert date and sort
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['athlete_id', 'date']).reset_index(drop=True)
    
    # Standardize injury
    df['injury'] = df['injury'].replace({'Yes': 1, 'yes': 1, 'YES': 1, 'No': 0, 'no': 0, 'NO': 0}).fillna(0).astype(int)
    
    # Pivot to wide (one row per athlete-date)
    df_wide = df.pivot_table(
        index=['athlete_id', 'date'],
        columns='metric',
        values='value',
        aggfunc='first'
    ).reset_index()
    
    # Bring back game_workload and injury
    wi = df[['athlete_id', 'date', 'game_workload', 'injury']].drop_duplicates()
    df_wide = df_wide.merge(wi, on=['athlete_id', 'date'], how='left')
    
    # Forward-fill hip_mobility per athlete, then fill remaining with 0
    if 'hip_mobility' in df_wide.columns:
        df_wide['hip_mobility'] = df_wide.groupby('athlete_id')['hip_mobility'].ffill().fillna(0)
    
    # Rolling features
    df_wide['acute_load_7d'] = df_wide.groupby('athlete_id')['game_workload'].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )
    df_wide['chronic_load_28d'] = df_wide.groupby('athlete_id')['game_workload'].transform(
        lambda x: x.rolling(28, min_periods=1).mean()
    )
    
    epsilon = 1e-6
    df_wide['acwr'] = df_wide['acute_load_7d'] / (df_wide['chronic_load_28d'] + epsilon)
    
    df_wide['acwr_spike_high'] = (df_wide['acwr'] > 1.5).astype(int)
    df_wide['acwr_spike_very_high'] = (df_wide['acwr'] > 2.0).astype(int)
    
    if 'hip_mobility' in df_wide.columns:
        df_wide['hip_mobility_7d_mean'] = df_wide.groupby('athlete_id')['hip_mobility'].transform(
            lambda x: x.rolling(7, min_periods=1).mean()
        )
        df_wide['hip_mobility_28d_mean'] = df_wide.groupby('athlete_id')['hip_mobility'].transform(
            lambda x: x.rolling(28, min_periods=1).mean()
        )
        df_wide['hip_mobility_pct_change'] = (
            (df_wide['hip_mobility'] - df_wide['hip_mobility_28d_mean']) 
            / (df_wide['hip_mobility_28d_mean'] + epsilon) * 100
        )
    
    # Target: injury anywhere in the next 7 days
    df_wide['injury_in_next_7d'] = df_wide.groupby('athlete_id')['injury'].transform(
        lambda x: x.shift(-1).rolling(7, min_periods=1).max()
    ).fillna(0).astype(int)
    
    # Drop rows with missing load features
    df_wide = df_wide.dropna(subset=['acute_load_7d', 'chronic_load_28d', 'acwr'])
    
    print("\nProcessed sample:")
    print(df_wide[['athlete_id', 'date', 'game_workload', 'hip_mobility', 'acwr', 'injury_in_next_7d']].head(12))
    
    print(f"\nFinal shape: {df_wide.shape}")
    print(f"Positive injury_in_next_7d cases: {df_wide['injury_in_next_7d'].sum()} ({df_wide['injury_in_next_7d'].mean():.4f})")
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_wide.to_csv(output_path, index=False)
    print(f"Saved: {Path(output_path).absolute()}")
    
    return df_wide


if __name__ == "__main__":
    load_and_process_workload_data()
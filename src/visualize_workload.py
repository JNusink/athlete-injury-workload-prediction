# src/visualize_workload.py
"""
Visualizations for workload and ACWR features.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_athlete_timeline(df: pd.DataFrame, athlete_ids: list = [1, 2, 3], save_dir: str = "figures"):
    """
    Plot workload, ACWR, and injury markers for selected athletes.
    """
    Path(save_dir).mkdir(exist_ok=True)
    
    for athlete in athlete_ids:
        df_ath = df[df['athlete_id'] == athlete].copy()
        df_ath = df_ath.set_index('date')
        
        fig, ax1 = plt.subplots(figsize=(14, 6))
        
        # Workload on left axis (bar or line)
        ax1.bar(df_ath.index, df_ath['game_workload'], color='lightblue', alpha=0.6, label='Daily Workload')
        ax1.set_ylabel('Game Workload', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # ACWR on right axis
        ax2 = ax1.twinx()
        ax2.plot(df_ath.index, df_ath['acwr'], color='darkorange', linewidth=2, label='ACWR')
        ax2.axhline(1.5, color='red', linestyle='--', alpha=0.7, label='High Spike Threshold')
        ax2.axhline(2.0, color='darkred', linestyle='--', alpha=0.7, label='Very High Spike')
        ax2.set_ylabel('ACWR', color='darkorange')
        ax2.tick_params(axis='y', labelcolor='darkorange')
        
        # Injury markers
        injuries = df_ath[df_ath['injury'] == 1]
        if not injuries.empty:
            ax1.scatter(injuries.index, injuries['game_workload'], color='red', s=100, 
                        marker='X', label='Injury Day', zorder=10)
        
        # Title and legend
        plt.title(f"Athlete {athlete}: Workload & ACWR Timeline")
        fig.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95))
        plt.tight_layout()
        
        save_path = Path(save_dir) / f"athlete_{athlete}_timeline.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved: {save_path}")

def plot_acwr_boxplot(df: pd.DataFrame, save_dir: str = "figures"):
    """
    Boxplot: ACWR on days with injury in next 7d vs not.
    """
    Path(save_dir).mkdir(exist_ok=True)
    
    df_plot = df[['acwr', 'injury_in_next_7d']].copy()
    df_plot['Risk Period'] = df_plot['injury_in_next_7d'].map({1: 'Injury in Next 7 Days', 0: 'Safe'})
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Risk Period', y='acwr', data=df_plot, palette='Set2')
    plt.title('ACWR Distribution: Risk vs Safe Periods')
    plt.ylabel('Acute:Chronic Workload Ratio (ACWR)')
    plt.xlabel('')
    plt.ylim(0, df_plot['acwr'].quantile(0.99))  # zoom in, ignore extreme outliers
    
    save_path = Path(save_dir) / "acwr_risk_boxplot.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")
    
    # Quick stats
    print("\nACWR Stats:")
    print(df_plot.groupby('Risk Period')['acwr'].describe())


if __name__ == "__main__":
    processed_path = "data/processed/athlete_workload_features.csv"
    df = pd.read_csv(processed_path)
    df['date'] = pd.to_datetime(df['date'])  # ensure date is datetime
    
    print("Generating visualizations...")
    plot_athlete_timeline(df, athlete_ids=[1, 2, 3])  # change IDs if needed
    plot_acwr_boxplot(df)
    print("Done!")
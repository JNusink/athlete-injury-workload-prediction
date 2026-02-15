## Workload Monitoring & Injury Risk Prediction

**Goal**: Use daily game workload and hip mobility trends to compute Acute:Chronic Workload Ratio (ACWR) and predict whether an injury occurs in the next 7 days.

### Data Processing & Features
- **Source**: `mergedData.csv` (~43,800 rows, ~30 athletes, 2016–2018)
- **Key features engineered**:
  - Acute load (7-day rolling mean)
  - Chronic load (28-day rolling mean)
  - ACWR = acute / chronic
  - ACWR spikes (>1.5 and >2.0)
  - Hip mobility trends (7d/28d mean, % change)
  - Target: `injury_in_next_7d` (1 if any injury in following 7 days)

Processed dataset saved: `data/processed/athlete_workload_features.csv` (21,900 rows).

**Insight**: ACWR is significantly higher before injury periods  
- Mean ACWR (risk periods): **1.74**  
- Mean ACWR (safe periods): **0.91**  

### Visualizations

#### Athlete Timelines (examples)
Daily workload (blue bars), ACWR (orange line), injury days (red X markers), high-risk thresholds (red dashed lines).

![Athlete 1 Timeline](figures/athlete_1_timeline.png)
![Athlete 2 Timeline](figures/athlete_2_timeline.png)

#### ACWR Distribution: Risk vs Safe Periods
Higher ACWR values clearly cluster before injury windows.

![ACWR Risk Boxplot](figures/acwr_risk_boxplot.png)

### Predictive Model (XGBoost)
**Model**: XGBoost classifier (chronological 80/20 train/test split, imbalance-weighted)

**Performance** (test set):
- ROC-AUC: **0.7917** (strong discrimination)
- At threshold 0.5:
  - Recall (positives): **0.65** → detects 65% of high-risk periods
  - Precision (positives): 0.13 → many false alarms (acceptable for prevention)
- Optimal threshold for max F1: **0.772** (F1 = 0.2828)

**Top predictors** (feature importance):
1. Acute load (7-day) → 0.506 (recent workload dominates risk)
2. Chronic load (28-day) → 0.115
3. ACWR → 0.089
4–7. Hip mobility trends (means & % change) provide supporting signal

![Feature Importance](figures/feature_importance.png)

**Interpretation**: Acute load spikes are the strongest signal of impending injury — consistent with sports science literature on workload-injury relationships. The model could alert coaches to reduce training when ACWR > 1.5–1.7.

**Limitations**: Small number of athletes (~30), imbalanced classes, no external validation set. Future work: larger cohorts, cross-validation, hyperparameter tuning.
AI Injury Risk Co-Pilot
=======================

Enter daily training -> Get "train or rest" advice instantly.

Predicts injury risk in next 7 days using workload + hip mobility.

What coaches see:
Hip: 13% (CRASH) | Workload: Normal  
-> HIGH RISK - "Light mobility only today"

Workload Guide (Session-RPE - Used by NFL/NBA/Soccer)
-----------------------------------------------------
Workload = Minutes x RPE (1-10) (Global sports science standard)

RPE Scale - Ask athlete post-session: "How hard was that? (1-10)"
RPE  Feels Like        Example
1-2  Rest             Warm-up, walking
3-4  Easy             Light jog, skills practice
5-6  Moderate         Normal practice pace
7    Hard             Game intensity
8-9  Very Hard        Sprints, heavy weights
10   Maximal          All-out race

Real Examples (Your Data)
Workload  Minutes  RPE  Session Type
200       60 min   3-4  Recovery run
400       80 min   5    Basketball practice
600       90 min   7    Game day
800+      100+ min 8+   Double session

Your athletes average: ~350/day (moderate team sport)

All Metrics Explained
---------------------
Input              | What measures     | Normal Range | Danger Zone
------------------------------------------------------------
Game Workload      | Today's stress    | 200-500      | 800+
Sleep Hours        | Recovery quality  | 7-9 hrs      | <6 hrs
Hip Mobility       | Hip flexibility   | 35-55%       | <25%
Acute Load (7d)    | Last week avg     | 150-300      | 500+
Chronic Load (28d) | Fitness base      | 140-250      | <100
ACWR               | Acute/Chronic     | ~1.0         | >1.5

Quick Start (5 minutes)
-----------------------
1. Install: pip install -r requirements.txt
2. Train: python src/model_workload.py
3. Launch: streamlit run app.py

Live: http://localhost:8501

How It Works (3 Steps)
----------------------
Daily logs -> [data_processing.py] -> 10 risk features (ACWR, hip % change)
             -> [model_workload.py] -> XGBoost model (AUC 0.83) 
             -> [app.py] -> "15% HIGH RISK - Active recovery"

Your Innovation: Hip Safety Net
Model misses hip crashes -> YOUR RULE forces HIGH RISK
Hip <25% OR drops >60% -> Clinical override

Performance
-----------
AUC: 0.83 (83% injury detection)
ACWR spikes: 55% feature importance
Hip crashes: Safety net catches model blind spots
Live predictions: <1 second

File Structure
--------------
├── app.py                 (Live predictor - Streamlit)
├── src/
│   ├── data_processing.py (Raw -> ML features)
│   └── model_workload.py  (Train XGBoost)
├── data/raw/mergedData.csv (Athlete logs)
├── models/workload_xgb.json (Trained model)
├── requirements.txt       (pip install -r)
└── figures/              (Model plots)

Technical Details
-----------------
Model: XGBoost binary classifier
Target: injury_in_next_7d (0/1)
Features: 10 workload + hip metrics
Validation: Chronological 80/20 split
Method: Session-RPE (global standard)

Credits
-------
Jared Nusink - Built for athlete safety

Star this repo! https://github.com/JNusink/athlete-injury-workload-prediction
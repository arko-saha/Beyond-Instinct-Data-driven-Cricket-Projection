import nbformat
import sys

def inject_cells():
    notebook_path = "Data_preprocessing.ipynb"
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # The Markdown Cell
    md_content = """# Custom Feature Engineering 
(Added via automated scripting)
Calculating context-aware match features like Current Run Rate (CRR), Required Run Rate (RRR), Wickets in Hand, and 10-match Rolling Statistics for batters and bowlers."""

    md_cell = nbformat.v4.new_markdown_cell(md_content)

    # The Code Cell
    code_content = """import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("Starting Feature Engineering...")

# Ensure 'runs_off_bat' and 'extras' are numeric
df['runs_off_bat'] = pd.to_numeric(df['runs_off_bat'], errors='coerce').fillna(0)
df['extras'] = pd.to_numeric(df['extras'], errors='coerce').fillna(0)
df['total_runs_this_ball'] = df['runs_off_bat'] + df['extras']

# Sort dataframe chronologically
df = df.sort_values(by=['start_date', 'match_id', 'innings', 'completed_over', 'ball_no']).reset_index(drop=True)

# 1. Cumulative Runs and Wickets in Innings
df['cumulative_runs'] = df.groupby(['match_id', 'innings'])['total_runs_this_ball'].cumsum()

# Handle dismissals
df['is_wicket'] = df['player_dismissed'].notnull().astype(int)
df['cumulative_wickets'] = df.groupby(['match_id', 'innings'])['is_wicket'].cumsum()
df['wickets_in_hand'] = 10 - df['cumulative_wickets']

# 2. Balls Remaining and CRR
df['total_balls_bowled'] = df.groupby(['match_id', 'innings']).cumcount() + 1
df['balls_remaining'] = 120 - df['total_balls_bowled'] 
df['balls_remaining'] = df['balls_remaining'].apply(lambda x: max(x, 0))

df['CRR'] = (df['cumulative_runs'] / df['total_balls_bowled']) * 6

# 3. Required Run Rate (RRR)
first_innings_total = df[df['innings'] == '1'].groupby('match_id')['total_runs_this_ball'].sum().reset_index()
if first_innings_total.empty:
    first_innings_total = df[df['innings'] == 1].groupby('match_id')['total_runs_this_ball'].sum().reset_index()
first_innings_total.rename(columns={'total_runs_this_ball': 'target_runs'}, inplace=True)
first_innings_total['target_runs'] += 1

df = df.merge(first_innings_total, on='match_id', how='left')
df['target_runs'] = df['target_runs'].fillna(-1) 

# Calculate RRR
innings_col = df['innings'].astype(str)
df['RRR'] = np.where(innings_col == '2', 
                     ((df['target_runs'] - df['cumulative_runs']) / df['balls_remaining'].replace(0, 1)) * 6,
                     0)
df.loc[df['RRR'] > 36, 'RRR'] = 36  # Capping RRR

# 4. Rolling stats
print("Calculating Rolling Features...")
batter_match = df.groupby(['striker', 'match_id', 'start_date']).agg(
    runs_scored=('runs_off_bat', 'sum'),
    balls_faced=('ball_no', 'count'),  
    dismissed=('is_wicket', 'sum') 
).reset_index().sort_values(by=['striker', 'start_date'])

bowler_match = df.groupby(['bowler', 'match_id', 'start_date']).agg(
    runs_conceded=('total_runs_this_ball', 'sum'),
    balls_bowled=('ball_no', 'count'),
    wickets_taken=('is_wicket', 'sum')
).reset_index().sort_values(by=['bowler', 'start_date'])

# Batter Rolling
batter_match['runs_last_10'] = batter_match.groupby('striker')['runs_scored'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).sum())
batter_match['balls_last_10'] = batter_match.groupby('striker')['balls_faced'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).sum())
batter_match['dismissed_last_10'] = batter_match.groupby('striker')['dismissed'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).sum())

batter_match['batter_sr_l10'] = (batter_match['runs_last_10'] / batter_match['balls_last_10']) * 100
batter_match['batter_avg_l10'] = batter_match['runs_last_10'] / batter_match['dismissed_last_10'].replace(0, 1)

# Bowler Rolling
bowler_match['runs_last_10'] = bowler_match.groupby('bowler')['runs_conceded'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).sum())
bowler_match['balls_last_10'] = bowler_match.groupby('bowler')['balls_bowled'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).sum())
bowler_match['wickets_last_10'] = bowler_match.groupby('bowler')['wickets_taken'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).sum())

bowler_match['bowler_eco_l10'] = (bowler_match['runs_last_10'] / (bowler_match['balls_last_10'] / 6))
bowler_match['bowler_sr_l10'] = bowler_match['balls_last_10'] / bowler_match['wickets_last_10'].replace(0, 1)

# Merge back 
df = df.merge(batter_match[['match_id', 'striker', 'batter_sr_l10', 'batter_avg_l10']], on=['match_id', 'striker'], how='left')
df = df.merge(bowler_match[['match_id', 'bowler', 'bowler_eco_l10', 'bowler_sr_l10']], on=['match_id', 'bowler'], how='left')

df['batter_sr_l10'] = df['batter_sr_l10'].fillna(120)
df['batter_avg_l10'] = df['batter_avg_l10'].fillna(25)
df['bowler_eco_l10'] = df['bowler_eco_l10'].fillna(8.0)
df['bowler_sr_l10'] = df['bowler_sr_l10'].fillna(20)

print("Feature Engineering Complete.")
display(df[['striker', 'bowler', 'cumulative_runs', 'wickets_in_hand', 'CRR', 'RRR', 'batter_sr_l10', 'bowler_eco_l10']].head())"""

    code_cell = nbformat.v4.new_code_cell(code_content)

    # XGBoost Modeling Cell
    md2_content = """# Model Architecture & Custom Variance Penalty Training
Using XGBoost with a Custom Loss Function.
Instead of standard MSE, we penalize the model heavily for large variance (e.g. predicting 1 when the batter hits a 6) but forgive minor deviations."""
    md2_cell = nbformat.v4.new_markdown_cell(md2_content)

    xgboost_cell = """import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Drop non-numeric columns and nan
features = ['completed_over', 'ball_no', 'wickets_in_hand', 'balls_remaining', 'cumulative_runs', 
            'CRR', 'RRR', 'batter_sr_l10', 'batter_avg_l10', 'bowler_eco_l10', 'bowler_sr_l10']

df_model = df[features + ['runs_off_bat']].dropna()
X = df_model[features]
y = df_model['runs_off_bat']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False) # Sequential split is better for cricket

# Custom Variance-based Loss Formulation
# We want to penalize heavily when the prediction is completely wrong.
# Let's use a Pseudo-Huber loss or Custom exponential penalty.
# XGBoost requires gradient and hessian.
def custom_variance_loss(preds, dtrain):
    labels = dtrain.get_label()
    diff = preds - labels
    # We apply a cubic penalty mathematically:
    # Loss = |diff|^3
    # Gradient (1st deriv) = 3 * diff * |diff|
    # Hessian (2nd deriv) = 6 * |diff|
    
    grad = 3.0 * diff * np.abs(diff)
    hess = 6.0 * np.abs(diff)
    
    # Adding an epsilon to hessian to avoid 0
    hess = np.maximum(hess, 1e-6)
    return grad, hess

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Model parameters
params = {
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}

print("Training XGBoost with Custom Variance Penalty...")
# We use standard objective as baseline, and then train custom:
model_standard = xgb.train({'objective': 'reg:squarederror', **params}, dtrain, num_boost_round=150)
model_custom = xgb.train(params, dtrain, num_boost_round=150, obj=custom_variance_loss)

# Evaluation
preds_std = model_standard.predict(dtest)
preds_custom = model_custom.predict(dtest)

std_rmse = np.sqrt(mean_squared_error(y_test, preds_std))
cust_rmse = np.sqrt(mean_squared_error(y_test, preds_custom))
print(f"Standard Model RMSE: {std_rmse:.4f}")
print(f"Custom Penalty Model RMSE: {cust_rmse:.4f}")

# The goal of the custom penalty was to reduce completely WRONG predictions (Error > 3 runs)
errors_std = np.abs(preds_std - y_test)
errors_cust = np.abs(preds_custom - y_test)

bad_misses_std = np.sum(errors_std > 3)
bad_misses_cust = np.sum(errors_cust > 3)

print(f"Standard Model - Extreme Errors (>3 runs difference): {bad_misses_std}")
print(f"Custom Model - Extreme Errors (>3 runs difference): {bad_misses_cust}")
"""
    xgb_code_cell = nbformat.v4.new_code_cell(xgboost_cell)

    nb.cells.extend([md_cell, code_cell, md2_cell, xgb_code_cell])

    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
        
    print("Successfully injected cells.")

if __name__ == "__main__":
    inject_cells()

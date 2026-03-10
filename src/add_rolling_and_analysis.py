"""
Add rolling batter/bowler form features and prediction distribution analysis.
1. Add 'bowler', 'non_striker', 'player_dismissed' to new_vars
2. Insert rolling feature cell after the existing feature engineering cell
3. Update XGBoost cell to use rolling features
4. Add prediction distribution visualization cell
"""
import nbformat

notebook_path = 'Data_preprocessing.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# ============================================================
# Step 1: Add missing columns to new_vars
# ============================================================
for cell in nb.cells:
    if cell.cell_type == 'code' and 'df1 = df[new_vars]' in cell.source:
        if "'bowler'" not in cell.source:
            cell.source = cell.source.replace(
                "'striker', 'completed_over'",
                "'striker', 'bowler', 'non_striker', 'player_dismissed', 'completed_over'"
            )
            print("Step 1: Added bowler, non_striker, player_dismissed to new_vars")
        else:
            print("Step 1: Columns already present")
        break

# ============================================================
# Step 2: Find the feature engineering cell and insert rolling features after it
# ============================================================
ROLLING_MD = '''# Rolling Player Form Features
Computing **Batter Strike Rate** and **Bowler Economy Rate** over each player's last 10 match appearances using only *past* data (shifted to avoid leakage).'''

ROLLING_CELL = '''print("Computing rolling player form features...")

# ── Batter match-level aggregation ──
batter_match = (
    df1.groupby(['striker', 'match_id', 'start_date'])
    .agg(
        runs_scored=('runs_off_bat', 'sum'),
        balls_faced=('ball_no', 'count'),
        dismissed=('is_wicket', 'sum')
    )
    .reset_index()
    .sort_values(by=['striker', 'start_date'])
)

# Rolling over last 10 matches (shifted by 1 to use only PAST data)
batter_match['runs_L10'] = batter_match.groupby('striker')['runs_scored'].transform(
    lambda x: x.shift(1).rolling(10, min_periods=1).sum()
)
batter_match['balls_L10'] = batter_match.groupby('striker')['balls_faced'].transform(
    lambda x: x.shift(1).rolling(10, min_periods=1).sum()
)
batter_match['dismissed_L10'] = batter_match.groupby('striker')['dismissed'].transform(
    lambda x: x.shift(1).rolling(10, min_periods=1).sum()
)

batter_match = batter_match.assign(
    batter_sr_l10=(batter_match['runs_L10'] / batter_match['balls_L10'].replace(0, 1)) * 100,
    batter_avg_l10=batter_match['runs_L10'] / batter_match['dismissed_L10'].replace(0, 1)
)

# ── Bowler match-level aggregation ──
bowler_match = (
    df1.groupby(['bowler', 'match_id', 'start_date'])
    .agg(
        runs_conceded=('total_runs', 'sum'),
        balls_bowled=('ball_no', 'count'),
        wickets_taken=('is_wicket', 'sum')
    )
    .reset_index()
    .sort_values(by=['bowler', 'start_date'])
)

bowler_match['runs_L10'] = bowler_match.groupby('bowler')['runs_conceded'].transform(
    lambda x: x.shift(1).rolling(10, min_periods=1).sum()
)
bowler_match['balls_L10'] = bowler_match.groupby('bowler')['balls_bowled'].transform(
    lambda x: x.shift(1).rolling(10, min_periods=1).sum()
)
bowler_match['wkts_L10'] = bowler_match.groupby('bowler')['wickets_taken'].transform(
    lambda x: x.shift(1).rolling(10, min_periods=1).sum()
)

bowler_match = bowler_match.assign(
    bowler_eco_l10=(bowler_match['runs_L10'] / (bowler_match['balls_L10'] / 6).replace(0, 1)),
    bowler_sr_l10=bowler_match['balls_L10'] / bowler_match['wkts_L10'].replace(0, 1)
)

# ── Merge back to df1 ──
# Drop any existing rolling columns first (safe re-run)
for col in ['batter_sr_l10', 'batter_avg_l10', 'bowler_eco_l10', 'bowler_sr_l10']:
    if col in df1.columns:
        df1 = df1.drop(columns=[col])

batter_merge = batter_match[['match_id', 'striker', 'batter_sr_l10', 'batter_avg_l10']].copy()
bowler_merge = bowler_match[['match_id', 'bowler', 'bowler_eco_l10', 'bowler_sr_l10']].copy()

df1 = pd.merge(df1, batter_merge, on=['match_id', 'striker'], how='left')
df1 = pd.merge(df1, bowler_merge, on=['match_id', 'bowler'], how='left')

# Fill NaN (players with no history yet) with sensible defaults
df1['batter_sr_l10'] = df1['batter_sr_l10'].fillna(120.0)   # T20 average SR
df1['batter_avg_l10'] = df1['batter_avg_l10'].fillna(22.0)   # T20 average
df1['bowler_eco_l10'] = df1['bowler_eco_l10'].fillna(8.0)     # T20 average economy
df1['bowler_sr_l10'] = df1['bowler_sr_l10'].fillna(18.0)      # T20 average bowling SR

print("Rolling features added!")
print(f"df1 shape: {df1.shape}")
display(df1[['striker', 'bowler', 'batter_sr_l10', 'batter_avg_l10', 'bowler_eco_l10', 'bowler_sr_l10']].describe())
'''

# Find the feature engineering code cell and insert rolling cells after it
fe_idx = None
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code' and 'Starting Feature Engineering on df1' in cell.source:
        fe_idx = i
        break

if fe_idx is not None:
    insert_at = fe_idx + 1
    
    # Check if rolling cells already exist
    already_exists = False
    if insert_at < len(nb.cells) and 'Rolling Player Form' in nb.cells[insert_at].source:
        nb.cells[insert_at].source = ROLLING_MD
        nb.cells[insert_at + 1].source = ROLLING_CELL
        already_exists = True
        print("Step 2: Updated existing rolling cells")
    
    if not already_exists:
        nb.cells.insert(insert_at, nbformat.v4.new_markdown_cell(ROLLING_MD))
        nb.cells.insert(insert_at + 1, nbformat.v4.new_code_cell(ROLLING_CELL))
        print("Step 2: Inserted new rolling feature cells")
else:
    print("ERROR: Could not find feature engineering cell!")

# ============================================================
# Step 3: Update XGBoost cell to include rolling features
# ============================================================
NEW_XGB_CELL = '''import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Features including rolling player form
features = ['completed_over', 'ball_no', 'wickets_in_hand', 'balls_remaining',
            'cumulative_runs', 'CRR', 'RRR',
            'batter_sr_l10', 'batter_avg_l10', 'bowler_eco_l10', 'bowler_sr_l10']

df_model = df1[features + ['runs_off_bat']].dropna()
X = df_model[features]
y = df_model['runs_off_bat']

# Sequential split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# ── Custom Cubic Penalty Loss ──
def custom_variance_loss(preds, dtrain):
    labels = dtrain.get_label()
    diff = preds - labels
    grad = 3.0 * diff * np.abs(diff)
    hess = 6.0 * np.abs(diff)
    hess = np.maximum(hess, 1e-6)
    return grad, hess

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}

print("Training XGBoost with Custom Variance Penalty...")
model_standard = xgb.train({'objective': 'reg:squarederror', **params}, dtrain, num_boost_round=300)
model_custom = xgb.train(params, dtrain, num_boost_round=300, obj=custom_variance_loss)

preds_std = model_standard.predict(dtest)
preds_custom = model_custom.predict(dtest)

std_rmse = np.sqrt(mean_squared_error(y_test, preds_std))
cust_rmse = np.sqrt(mean_squared_error(y_test, preds_custom))
std_mae = mean_absolute_error(y_test, preds_std)
cust_mae = mean_absolute_error(y_test, preds_custom)
std_r2 = r2_score(y_test, preds_std)
cust_r2 = r2_score(y_test, preds_custom)

print(f"\\nStandard Model  RMSE: {std_rmse:.4f}  MAE: {std_mae:.4f}  R²: {std_r2:.4f}")
print(f"Custom Model    RMSE: {cust_rmse:.4f}  MAE: {cust_mae:.4f}  R²: {cust_r2:.4f}")

errors_std = np.abs(preds_std - y_test.values)
errors_cust = np.abs(preds_custom - y_test.values)
for t in [2, 3, 4, 6]:
    bad_std = int(np.sum(errors_std > t))
    bad_cust = int(np.sum(errors_cust > t))
    print(f"Errors > {t} runs: Standard={bad_std}, Custom={bad_cust}")
'''

for cell in nb.cells:
    if cell.cell_type == 'code' and 'custom_variance_loss' in cell.source:
        cell.source = NEW_XGB_CELL
        print("Step 3: Updated XGBoost cell with rolling features")
        break

# ============================================================
# Step 4: Add prediction distribution cell after XGBoost
# ============================================================
PRED_DIST_MD = '''# Prediction Distribution Analysis
Visualizing predicted vs actual value distributions to diagnose model behavior.'''

PRED_DIST_CELL = '''import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ── 1. Actual vs Predicted histogram ──
axes[0, 0].hist(y_test, bins=range(-1, 8), alpha=0.7, label='Actual', color='#2196F3', edgecolor='white')
axes[0, 0].hist(preds_std, bins=50, alpha=0.6, label='Predicted (Std)', color='#FF5722', edgecolor='white')
axes[0, 0].set_title('Standard Model: Actual vs Predicted', fontweight='bold')
axes[0, 0].set_xlabel('Runs off bat')
axes[0, 0].legend()

axes[0, 1].hist(y_test, bins=range(-1, 8), alpha=0.7, label='Actual', color='#2196F3', edgecolor='white')
axes[0, 1].hist(preds_custom, bins=50, alpha=0.6, label='Predicted (Custom)', color='#E91E63', edgecolor='white')
axes[0, 1].set_title('Custom Penalty: Actual vs Predicted', fontweight='bold')
axes[0, 1].set_xlabel('Runs off bat')
axes[0, 1].legend()

# ── 2. Prediction spread ──
axes[1, 0].boxplot([preds_std, preds_custom, y_test.values],
                    labels=['Std Pred', 'Custom Pred', 'Actual'],
                    patch_artist=True,
                    boxprops=dict(facecolor='#E3F2FD'))
axes[1, 0].set_title('Prediction Spread Comparison', fontweight='bold')
axes[1, 0].set_ylabel('Runs off bat')

# ── 3. Error distribution ──
axes[1, 1].hist(errors_std, bins=50, alpha=0.6, label='Std Errors', color='#FF5722')
axes[1, 1].hist(errors_cust, bins=50, alpha=0.6, label='Custom Errors', color='#E91E63')
axes[1, 1].set_title('Error Distribution', fontweight='bold')
axes[1, 1].set_xlabel('Absolute Error')
axes[1, 1].legend()

plt.tight_layout()
plt.show()

# ── Summary statistics ──
print("\\n=== Prediction Statistics ===")
print(f"{'':20} {'Actual':>10} {'Std Pred':>10} {'Custom Pred':>12}")
print("-" * 55)
for stat in ['mean', 'std', 'min', '25%', '50%', '75%', 'max']:
    actual = y_test.describe()[stat]
    std_val = pd.Series(preds_std).describe()[stat]
    cust_val = pd.Series(preds_custom).describe()[stat]
    print(f"{stat:20} {actual:>10.4f} {std_val:>10.4f} {cust_val:>12.4f}")

# ── Feature Importance ──
print("\\n=== Feature Importance (Standard Model) ===")
importance = model_standard.get_score(importance_type='gain')
sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
for feat, score in sorted_imp:
    print(f"  {feat:25} {score:>10.2f}")
'''

# Find the XGBoost cell and insert distribution analysis after it
xgb_idx = None
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code' and 'custom_variance_loss' in cell.source:
        xgb_idx = i
        break

if xgb_idx is not None:
    insert_at = xgb_idx + 1
    
    # Check if prediction distribution cells already exist
    already_exists = False
    if insert_at < len(nb.cells) and 'Prediction Distribution' in nb.cells[insert_at].source:
        nb.cells[insert_at].source = PRED_DIST_MD
        nb.cells[insert_at + 1].source = PRED_DIST_CELL
        already_exists = True
        print("Step 4: Updated existing prediction distribution cells")
    
    if not already_exists:
        nb.cells.insert(insert_at, nbformat.v4.new_markdown_cell(PRED_DIST_MD))
        nb.cells.insert(insert_at + 1, nbformat.v4.new_code_cell(PRED_DIST_CELL))
        print("Step 4: Inserted prediction distribution cells")
else:
    print("ERROR: Could not find XGBoost cell!")

# ============================================================
# Save
# ============================================================
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("\nAll done!")

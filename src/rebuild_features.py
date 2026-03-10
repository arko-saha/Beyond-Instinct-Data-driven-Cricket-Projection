"""
Fix the notebook so that:
1. Features (cumulative_runs, wickets_in_hand, CRR, RRR) are added to df1
2. RRR is NOT capped at 36
3. Feature cell is inserted right after df1 is created (Cell 21)
4. Remove or update the old feature cell that operated on df
"""
import nbformat

notebook_path = 'Data_preprocessing.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# ============================================================
# Step 1: Create the new feature engineering cell for df1
# ============================================================
NEW_FEATURE_MD = '''# Feature Engineering
Adding match-state context features to `df1`: **Cumulative Runs**, **Wickets in Hand**, **Current Run Rate (CRR)**, and **Required Run Rate (RRR)**.'''

NEW_FEATURE_CELL = '''import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("Starting Feature Engineering on df1...")

# Ensure numeric types
df1['runs_off_bat'] = pd.to_numeric(df1['runs_off_bat'], errors='coerce').fillna(0)
df1['extras'] = pd.to_numeric(df1['extras'], errors='coerce').fillna(0)
df1['total_runs_this_ball'] = df1['runs_off_bat'] + df1['extras']

# Sort chronologically within each match and innings
df1 = df1.sort_values(by=['start_date', 'match_id', 'innings', 'completed_over', 'ball_no']).reset_index(drop=True)

# ── 1. Cumulative Runs ──
df1['cumulative_runs'] = df1.groupby(['match_id', 'innings'])['total_runs_this_ball'].cumsum()

# ── 2. Wickets in Hand ──
df1['is_wicket'] = df1['player_dismissed'].notnull().astype(int)
df1['cumulative_wickets'] = df1.groupby(['match_id', 'innings'])['is_wicket'].cumsum()
df1['wickets_in_hand'] = 10 - df1['cumulative_wickets']

# ── 3. Balls Remaining & CRR ──
df1['total_balls_bowled'] = df1.groupby(['match_id', 'innings']).cumcount() + 1
df1['balls_remaining'] = (120 - df1['total_balls_bowled']).clip(lower=0)
df1['CRR'] = (df1['cumulative_runs'] / df1['total_balls_bowled']) * 6

# ── 4. Required Run Rate (RRR) ──
# Drop any leftover target_runs columns from previous runs
cols_to_drop = [c for c in df1.columns if 'target_runs' in c]
if cols_to_drop:
    df1 = df1.drop(columns=cols_to_drop)

# Calculate 1st innings total per match
innings_as_str = df1['innings'].astype(str)
first_innings_total = (
    df1[innings_as_str == '1']
    .groupby('match_id')['total_runs_this_ball']
    .sum()
    .reset_index()
    .rename(columns={'total_runs_this_ball': 'target_runs'})
)
first_innings_total['target_runs'] += 1  # Target = 1st innings total + 1

df1 = df1.merge(first_innings_total, on='match_id', how='left')
df1['target_runs'] = df1['target_runs'].fillna(-1)

# RRR only applies to 2nd innings (no capping)
df1['RRR'] = np.where(
    df1['innings'].astype(str) == '2',
    ((df1['target_runs'] - df1['cumulative_runs']) / df1['balls_remaining'].replace(0, 1)) * 6,
    0
)

print("Feature Engineering Complete!")
print(f"df1 shape: {df1.shape}")
display(df1[['match_id', 'innings', 'striker', 'bowler', 'runs_off_bat', 'cumulative_runs', 'wickets_in_hand', 'CRR', 'RRR']].head(10))
'''

# ============================================================
# Step 2: Find Cell 21 (df1 = df[new_vars]) and insert after it
# ============================================================
insert_idx = None
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code' and 'df1 = df[new_vars]' in cell.source:
        insert_idx = i + 1
        print(f"Found df1 creation at cell {i}, will insert features at cell {insert_idx}")
        break

if insert_idx is None:
    print("ERROR: Could not find df1 = df[new_vars] cell!")
    exit(1)

# Check if we already inserted a feature cell here before
already_exists = False
if insert_idx < len(nb.cells):
    next_cell = nb.cells[insert_idx]
    if next_cell.cell_type == 'markdown' and 'Feature Engineering' in next_cell.source:
        # Already inserted, replace it and the code cell after
        nb.cells[insert_idx].source = NEW_FEATURE_MD
        nb.cells[insert_idx + 1].source = NEW_FEATURE_CELL
        already_exists = True
        print("Replaced existing feature engineering cells")

if not already_exists:
    nb.cells.insert(insert_idx, nbformat.v4.new_markdown_cell(NEW_FEATURE_MD))
    nb.cells.insert(insert_idx + 1, nbformat.v4.new_code_cell(NEW_FEATURE_CELL))
    print("Inserted new feature engineering cells")

# ============================================================
# Step 3: Remove the OLD feature engineering cell that operated on df
# ============================================================
cells_to_remove = []
for i, cell in enumerate(nb.cells):
    # Skip the one we just inserted
    if i == insert_idx or i == insert_idx + 1:
        continue
    if cell.cell_type == 'markdown' and 'Custom Feature Engineering' in cell.source:
        cells_to_remove.append(i)
    if cell.cell_type == 'code' and 'cumulative_runs' in cell.source and "df['runs_off_bat']" in cell.source:
        cells_to_remove.append(i)

# Remove in reverse order to preserve indices
for idx in sorted(cells_to_remove, reverse=True):
    print(f"Removing old feature cell at index {idx}")
    del nb.cells[idx]

# ============================================================
# Step 4: Update the XGBoost cell to use df1
# ============================================================
NEW_XGB_CELL = '''import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Features for the model
features = ['completed_over', 'ball_no', 'wickets_in_hand', 'balls_remaining',
            'cumulative_runs', 'CRR', 'RRR']

df_model = df1[features + ['runs_off_bat']].dropna()
X = df_model[features]
y = df_model['runs_off_bat']

# Sequential split (no shuffling for time-series-like cricket data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# ── Custom Variance-based Loss (Cubic Penalty) ──
def custom_variance_loss(preds, dtrain):
    labels = dtrain.get_label()
    diff = preds - labels
    grad = 3.0 * diff * np.abs(diff)       # 1st derivative of |e|^3
    hess = 6.0 * np.abs(diff)              # 2nd derivative of |e|^3
    hess = np.maximum(hess, 1e-6)          # Avoid zero hessian
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

print("Training XGBoost models...")
model_standard = xgb.train({'objective': 'reg:squarederror', **params}, dtrain, num_boost_round=150)
model_custom = xgb.train(params, dtrain, num_boost_round=150, obj=custom_variance_loss)

# ── Evaluation ──
preds_std = model_standard.predict(dtest)
preds_custom = model_custom.predict(dtest)

print(f"\\n{'Metric':<30} {'Standard MSE':>15} {'Custom Penalty':>15}")
print("-" * 62)
for name, fn in [('RMSE', lambda y,p: np.sqrt(mean_squared_error(y,p))),
                  ('MAE', mean_absolute_error),
                  ('R-squared', r2_score)]:
    print(f"{name:<30} {fn(y_test, preds_std):>15.4f} {fn(y_test, preds_custom):>15.4f}")

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
        print("Updated XGBoost cell to use df1")
        break

# ============================================================
# Save
# ============================================================
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("\nDone! Features will be added to df1 right after it's created.")

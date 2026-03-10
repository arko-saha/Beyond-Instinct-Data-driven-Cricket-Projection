import nbformat
import sys

notebook_path = 'Data_preprocessing.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

code_cell = None
index = -1
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code' and 'first_innings_total = df' in cell.source:
        code_cell = cell
        index = i
        break

if code_cell:
    start_idx = code_cell.source.find('# 3. Required Run Rate (RRR)')
    end_idx = code_cell.source.find('# 4. Rolling stats')
    old_rrr_block = code_cell.source[start_idx:end_idx]

    new_rrr_block = """# 3. Required Run Rate (RRR)
# Safe cleanup of any previous 'target_runs' columns to prevent MergeError
cols_to_drop = [c for c in df.columns if 'target_runs' in c]
if cols_to_drop:
    df = df.drop(columns=cols_to_drop)

df['innings_str'] = df['innings'].astype(str)
first_innings_total = df[df['innings_str'] == '1'].groupby('match_id')['total_runs_this_ball'].sum().reset_index()

if not first_innings_total.empty:
    first_innings_total.rename(columns={'total_runs_this_ball': 'target_runs'}, inplace=True)
    first_innings_total['target_runs'] += 1
    df = df.merge(first_innings_total, on='match_id', how='left')
else:
    df['target_runs'] = -1

df['target_runs'] = df['target_runs'].fillna(-1)

# Calculate RRR
innings_col = df['innings'].astype(str)
df['RRR'] = np.where(innings_col == '2', 
                     ((df['target_runs'] - df['cumulative_runs']) / df['balls_remaining'].replace(0, 1)) * 6,
                     0)
df.loc[df['RRR'] > 36, 'RRR'] = 36  # Capping RRR

"""

    new_source = code_cell.source.replace(old_rrr_block, new_rrr_block)
    code_cell.source = new_source
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print("Fixed MergeError lock robustly!")
else:
    print("Cell not found!")

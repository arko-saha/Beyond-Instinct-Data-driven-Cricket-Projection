import nbformat
import sys

notebook_path = 'Data_preprocessing.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

code_cell = None
for cell in nb.cells:
    if cell.cell_type == 'code' and "df['innings_str'] = df['innings'].astype(str)" in cell.source:
        code_cell = cell
        break

if code_cell:
    new_source = code_cell.source.replace("""# 3. Required Run Rate (RRR)
df['innings_str'] = df['innings'].astype(str)
first_innings_total = df[df['innings_str'] == '1'].groupby('match_id')['total_runs_this_ball'].sum().reset_index()

if not first_innings_total.empty:
    first_innings_total.rename(columns={'total_runs_this_ball': 'target_runs'}, inplace=True)
    first_innings_total['target_runs'] += 1
    df = df.merge(first_innings_total, on='match_id', how='left')
else:
    df['target_runs'] = -1""", 
"""# 3. Required Run Rate (RRR)
df['innings_str'] = df['innings'].astype(str)
first_innings_total = df[df['innings_str'] == '1'].groupby('match_id')['total_runs_this_ball'].sum().reset_index()

if not first_innings_total.empty:
    first_innings_total.rename(columns={'total_runs_this_ball': 'target_runs'}, inplace=True)
    first_innings_total['target_runs'] += 1
    
    # Safely drop target_runs if it already exists from a previous cell execution
    if 'target_runs' in df.columns:
        df = df.drop(columns=['target_runs'])
        
    df = df.merge(first_innings_total, on='match_id', how='left')
else:
    if 'target_runs' not in df.columns:
        df['target_runs'] = -1""")

    code_cell.source = new_source
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print("Fixed MergeError lock!")
else:
    print("Cell not found!")

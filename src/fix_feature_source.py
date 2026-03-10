import nbformat
import json

notebook_path = 'Data_preprocessing.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

code_cell = None
for cell in nb.cells:
    if cell.cell_type == 'code' and 'total_runs_this_ball' in cell.source and 'cumulative_runs' in cell.source:
        code_cell = cell
        break

if code_cell:
    new_source = code_cell.source.replace("df['runs_off_bat'] = pd.to_numeric(df['runs_off_bat'], errors='coerce').fillna(0)",
    """print("Loading fresh data from merged.csv for accurate match tracking...")
df = pd.read_csv('data/merged.csv')

# Ensure columns map correctly for our feature engineering
if 'batter' in df.columns and 'striker' not in df.columns:
    df.rename(columns={'batter': 'striker'}, inplace=True)
if 'player_out' in df.columns and 'player_dismissed' not in df.columns:
    df.rename(columns={'player_out': 'player_dismissed'}, inplace=True)

df['runs_off_bat'] = pd.to_numeric(df['runs_off_bat'], errors='coerce').fillna(0)""")
    
    code_cell.source = new_source
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print('Fixed!')
else:
    print('Cell not found!')

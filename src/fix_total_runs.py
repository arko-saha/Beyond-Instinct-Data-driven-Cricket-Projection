"""
Fix: use existing columns in df1 instead of total_runs_this_ball
The merged.csv has 'total_runs' but new_vars might not include it.
So compute total_runs_this_ball from runs_off_bat + extras directly in the cell.
The real issue: new_vars needs 'extras' to exist in df1.
"""
import nbformat

notebook_path = 'Data_preprocessing.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# First, check what new_vars has and add 'extras' and 'total_runs' if missing
for cell in nb.cells:
    if cell.cell_type == 'code' and 'df1 = df[new_vars]' in cell.source:
        if "'total_runs'" not in cell.source:
            cell.source = cell.source.replace(
                "'runs_off_bat', 'extras'",
                "'runs_off_bat', 'extras', 'total_runs'"
            )
            print("Added 'total_runs' to new_vars")
        break

# Now update the feature engineering cell to use total_runs directly
for cell in nb.cells:
    if cell.cell_type == 'code' and 'Starting Feature Engineering on df1' in cell.source:
        cell.source = cell.source.replace(
            """# Ensure numeric types
df1['runs_off_bat'] = pd.to_numeric(df1['runs_off_bat'], errors='coerce').fillna(0)
df1['extras'] = pd.to_numeric(df1['extras'], errors='coerce').fillna(0)
df1['total_runs_this_ball'] = df1['runs_off_bat'] + df1['extras']""",
            """# Ensure numeric types
df1['runs_off_bat'] = pd.to_numeric(df1['runs_off_bat'], errors='coerce').fillna(0)
df1['total_runs'] = pd.to_numeric(df1['total_runs'], errors='coerce').fillna(0)"""
        )
        # Replace all references to total_runs_this_ball with total_runs
        cell.source = cell.source.replace('total_runs_this_ball', 'total_runs')
        print("Updated feature cell to use total_runs")
        break

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Done!")

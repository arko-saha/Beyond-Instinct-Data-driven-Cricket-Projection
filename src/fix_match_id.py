import nbformat

notebook_path = "Data_preprocessing.ipynb"
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

modified = False
for cell in nb.cells:
    if cell.cell_type == 'code':
        if "new_order = ['start_date', 'venue', 'innings'," in cell.source:
            cell.source = cell.source.replace("new_order = ['start_date', 'venue', 'innings',", "new_order = ['match_id', 'start_date', 'venue', 'innings',")
            modified = True
            break

if modified:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print('Successfully updated Data_preprocessing.ipynb to include match_id.')
else:
    print('Could not find the target code to replace.')

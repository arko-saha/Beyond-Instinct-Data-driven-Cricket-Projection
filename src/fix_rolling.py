import nbformat
import sys

notebook_path = 'Data_preprocessing.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

code_cell = None
index = -1
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code' and 'batter_sr_l10' in cell.source:
        code_cell = cell
        index = i
        break

if code_cell:
    new_source = code_cell.source.replace("""# Batter Rolling
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
df = df.merge(bowler_match[['match_id', 'bowler', 'bowler_eco_l10', 'bowler_sr_l10']], on=['match_id', 'bowler'], how='left')""",

"""# Batter Rolling
batter_match['runs_last_10'] = batter_match.groupby('striker')['runs_scored'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).sum())
batter_match['balls_last_10'] = batter_match.groupby('striker')['balls_faced'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).sum())
batter_match['dismissed_last_10'] = batter_match.groupby('striker')['dismissed'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).sum())

batter_sr_l10_calc = (batter_match['runs_last_10'] / batter_match['balls_last_10']) * 100
batter_match = batter_match.assign(batter_sr_l10=batter_sr_l10_calc)

batter_avg_l10_calc = batter_match['runs_last_10'] / batter_match['dismissed_last_10'].replace(0, 1)
batter_match = batter_match.assign(batter_avg_l10=batter_avg_l10_calc)

# Bowler Rolling
bowler_match['runs_last_10'] = bowler_match.groupby('bowler')['runs_conceded'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).sum())
bowler_match['balls_last_10'] = bowler_match.groupby('bowler')['balls_bowled'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).sum())
bowler_match['wickets_last_10'] = bowler_match.groupby('bowler')['wickets_taken'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).sum())

bowler_eco_l10_calc = (bowler_match['runs_last_10'] / (bowler_match['balls_last_10'] / 6))
bowler_match = bowler_match.assign(bowler_eco_l10=bowler_eco_l10_calc)

bowler_sr_l10_calc = bowler_match['balls_last_10'] / bowler_match['wickets_last_10'].replace(0, 1)
bowler_match = bowler_match.assign(bowler_sr_l10=bowler_sr_l10_calc)

# Merge back handling potential existing columns silently
for col in ['batter_sr_l10', 'batter_avg_l10']:
    if col in df.columns:
        df = df.drop(columns=[col])
for col in ['bowler_eco_l10', 'bowler_sr_l10']:
    if col in df.columns:
        df = df.drop(columns=[col])

df = pd.merge(df, batter_match[['match_id', 'striker', 'batter_sr_l10', 'batter_avg_l10']], on=['match_id', 'striker'], how='left')
df = pd.merge(df, bowler_match[['match_id', 'bowler', 'bowler_eco_l10', 'bowler_sr_l10']], on=['match_id', 'bowler'], how='left')""")
    
    code_cell.source = new_source
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print('Fixed Rolling Stats block!')
else:
    print('Cell not found!')

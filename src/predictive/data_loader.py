"""
Data loader for the predictive analysis module.
Standardizes loading of batting performance data.
"""

import pandas as pd
from pathlib import Path
import yaml

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_batters_data(file_path=None, config=None):
    if config is None:
        config = load_config()
    
    if file_path is None:
        file_path = config['paths']['raw_data']
    
    path = Path(file_path)
    if path.exists():
        df = pd.read_csv(path)
        if 'Player_Name' in df.columns and 'Player' not in df.columns:
            df.rename(columns={'Player_Name': 'Player'}, inplace=True)
        print(f"Loaded {len(df)} rows from {path}")
        return df
    else:
        raise FileNotFoundError(f"Data file not found at {path}")

def validate_schema(df, required_cols=None):
    if required_cols is None:
        required_cols = ['Player', 'Runs', 'BF', 'SR', 'Fours', 'Sixes', 'Opposition', 'Ground', 'Start_Date']
    
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    return True

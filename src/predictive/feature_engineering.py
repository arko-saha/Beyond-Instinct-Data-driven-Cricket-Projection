"""
Feature engineering for batting performance prediction.
"""

import pandas as pd
import numpy as np

def add_temporal_features(df):
    df = df.copy()
    if 'Start_Date' in df.columns:
        df['Year'] = df['Start_Date'].dt.year
        df['Month'] = df['Start_Date'].dt.month
        
        # Career age in days
        df = df.sort_values(['Player', 'Start_Date'])
        df['First_Match'] = df.groupby('Player')['Start_Date'].transform('min')
        df['Career_Age_Days'] = (df['Start_Date'] - df['First_Match']).dt.days
        df.drop(columns=['First_Match'], inplace=True)
        
    return df

def add_rolling_features(df, windows=[5, 10]):
    df = df.copy()
    df = df.sort_values(['Player', 'Start_Date'])
    
    for w in windows:
        # Rolling average of runs (excluding current match to avoid leakage)
        df[f'Runs_Avg_L{w}'] = df.groupby('Player')['Runs'].transform(
            lambda x: x.shift(1).rolling(window=w, min_periods=1).mean()
        ).fillna(0)
        
        df[f'SR_Avg_L{w}'] = df.groupby('Player')['SR'].transform(
            lambda x: x.shift(1).rolling(window=w, min_periods=1).mean()
        ).fillna(0)
        
    return df

def add_interaction_features(df):
    df = df.copy()
    # SR * BF is basically Runs, but can be useful if they are scaled differently
    # Boundary percentage
    safe_runs = df['Runs'].replace(0, 1)
    df['Boundary_Pct'] = ((df['Fours'] * 4) + (df['Sixes'] * 6)) / safe_runs
    
    return df

def engineer_all_features(df):
    df = add_temporal_features(df)
    df = add_rolling_features(df)
    df = add_interaction_features(df)
    return df

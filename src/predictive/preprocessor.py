"""
Preprocessing module for cricket batting prediction.
Handles cleaning, imputation, and encoding.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import yaml

def load_config(config_path="../config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class PredictivePreprocessor:
    def __init__(self, config=None):
        self.config = config if config else load_config()
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.categorical_features = ['Player', 'Opposition', 'Ground']
        self.numerical_features = ['Runs', 'BF', 'Fours', 'Sixes', 'SR']

    def clean_data(self, df):
        df = df.copy()
        
        # Replace DNB/TDNB with 0
        dnb_vals = self.config['data_cleaning']['dnb_values']
        for col in self.numerical_features:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: 0 if str(x).strip() in dnb_vals else x)
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
        # Filter invalid innings if any
        if 'Inns' in df.columns:
            df = df[df['Inns'] != '-']
            df['Inns'] = pd.to_numeric(df['Inns'], errors='coerce').fillna(1)
            
        # Convert Start_Date
        if 'Start_Date' in df.columns:
            df['Start_Date'] = pd.to_datetime(df['Start_Date'], errors='coerce')
        
        # Drop unwanted columns (excluding Start_Date as it's needed for sorting)
        drop_cols = self.config['data_cleaning'].get('drop_cols', [])
        cols_to_drop = [c for c in drop_cols if c in df.columns and c != 'Start_Date']
        if 'Unnamed: 0' in df.columns:
            cols_to_drop.append('Unnamed: 0')
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            
        return df

    def handle_outliers(self, df):
        # Implementation for winsorization if needed
        return df

    def fit_transform(self, df):
        df = self.clean_data(df)
        
        # Encoding
        for feat in self.categorical_features:
            if feat in df.columns:
                le = LabelEncoder()
                df[feat] = le.fit_transform(df[feat].astype(str))
                self.label_encoders[feat] = le
            
        # Scaling all numeric columns except target
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Runs' in numeric_cols:
            numeric_cols.remove('Runs') # Don't scale target
        
        if numeric_cols:
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
            self.numeric_cols_ = numeric_cols
        
        return df

    def transform(self, df):
        df = self.clean_data(df)
        
        for feat, le in self.label_encoders.items():
            if feat in df.columns:
                df[feat] = df[feat].astype(str).map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
            
        if hasattr(self, 'numeric_cols_'):
            df[self.numeric_cols_] = self.scaler.transform(df[self.numeric_cols_])
        return df

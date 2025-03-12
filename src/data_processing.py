import pandas as pd
import numpy as np
import yaml

def clean_data(raw_path: str) -> pd.DataFrame:
    """Clean raw X analytics data"""
    df = pd.read_csv(raw_path)
    
    # Handle missing values
    df['post_text'] = df['post_text'].fillna('')
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Remove URLs
    df['post_text'] = df['post_text'].str.replace(r'http\S+', '', regex=True)
    
    # Convert dates
    df['date'] = pd.to_datetime(df['date'])
    
    return df

if __name__ == "__main__":
    with open("../config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    df = clean_data(config['paths']['raw_data'])
    df.to_parquet(config['paths']['processed_data'])
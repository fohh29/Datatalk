#Module 1 — Data Preparation
import pandas as pd
import ast

def clean_data(df):
    # 1. Convert 'K' values (like 3.9K) to actual numbers (3900)
    def convert_k_to_num(value):
        if isinstance(value, str):
            if 'K' in value:
                return float(value.replace('K', '')) * 1000
        return value

    cols_to_fix = ['Times Listed', 'Number of Reviews', 'Plays', 'Playing', 'Backlogs', 'Wishlist']
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = df[col].apply(convert_k_to_num).astype(float)

    # 2. Fix the Genres column (Convert string "['RPG']" to a real list)
    if 'Genres' in df.columns:
        df['Genres'] = df['Genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # 3. Handle missing values
    df = df.fillna(0)
    
    return df

def get_outlier_report(df):
    # Simple outlier check for the Rating column
    if 'Rating' in df.columns:
        q1 = df['Rating'].quantile(0.25)
        q3 = df['Rating'].quantile(0.75)
        iqr = q3 - q1
        outliers = df[(df['Rating'] < (q1 - 1.5 * iqr)) | (df['Rating'] > (q3 + 1.5 * iqr))]
        return f"Found {len(outliers)} outliers in Rating."
    return "No Rating column found for outlier analysis."
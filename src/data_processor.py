#Module 1 — Data Preparation
import pandas as pd
import ast

def clean_data(df):
    # 1. Convert 'K' values (like 3.9K) to actual numbers (3900)
    def convert_k_to_num(value):
        if isinstance(value, str):
            value = value.replace('$', '').replace(',', '') # Clean extra symbols
            if 'K' in value:
                return float(value.replace('K', '')) * 1000
        try:
            return float(value)
        except:
            return value

    cols_to_fix = ['Times Listed', 'Number of Reviews', 'Plays', 'Playing', 'Backlogs', 'Wishlist', 'Rating']
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = df[col].apply(convert_k_to_num)

    # 2. Fix the Genres column (Convert list string to a plain comma-separated string)
    # This prevents the "unhashable type: list" error
    if 'Genres' in df.columns:
        def simplify_genres(x):
            try:
                if isinstance(x, str) and x.startswith('['):
                    return ", ".join(ast.literal_eval(x))
                return x
            except:
                return x
        df['Genres'] = df['Genres'].apply(simplify_genres)

    # 3. Handle missing values
    df = df.fillna(0)
    
    return df

def get_outlier_report(df):
    if 'Rating' in df.columns:
        df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
        q1 = df['Rating'].quantile(0.25)
        q3 = df['Rating'].quantile(0.75)
        iqr = q3 - q1
        outliers = df[(df['Rating'] < (q1 - 1.5 * iqr)) | (df['Rating'] > (q3 + 1.5 * iqr))]
        return f"Found {len(outliers)} outliers in Rating."
    return "No Rating column found."
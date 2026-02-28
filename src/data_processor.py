

import pandas as pd
import numpy as np
import ast
import re

def clean_data(df):
    """
    Generic data cleaning:
    - Handle missing values (fill numerics with median, categoricals with mode)
    - Convert individual values with 'K', 'M' suffixes to numbers, keeping non-convertible as is
    - Do not force columns to numeric if they contain non-numeric data
    - Attempt to parse list-like strings (e.g., genres) if detected
    - Convert potential date columns
    - Strip strings for consistency, but preserve case
    """
    
    # 1. Handle missing values
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        elif pd.api.types.is_object_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')

    # 2. Detect and convert values with suffixes like 'K', 'M', '$', etc., individually
    def convert_suffix_to_num(value):
        if pd.isna(value):
            return np.nan
        if not isinstance(value, str):
            return value  
        val_str = value.replace('$', '').replace(',', '').strip().upper()
        multipliers = {'K': 1e3, 'M': 1e6, 'B': 1e9}
        for suffix, mult in multipliers.items():
            if suffix in val_str:
                try:
                    num_part = re.sub(r'[^0-9.]', '', val_str)
                    return float(num_part) * mult
                except ValueError:
                    return value  
        try:
            return float(val_str)
        except ValueError:
            return value  

    for col in df.columns:
    
        non_na = df[col].dropna()
        if len(non_na) > 0:
            has_suffix = non_na.apply(lambda x: isinstance(x, str) and any(s in x.upper() for s in ['K', 'M', 'B', '$'])).mean() > 0.01
            if has_suffix:
                df[col] = df[col].apply(convert_suffix_to_num)
        
        
        if pd.api.types.is_object_dtype(df[col]):
            numeric_ratio = pd.to_numeric(df[col], errors='coerce').notna().mean()
            if numeric_ratio > 0.8:  
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
                df[col] = df[col].fillna(df[col].median())


    def parse_list_str(x):
        if isinstance(x, str) and x.startswith('[') and x.endswith(']'):
            try:
                items = ast.literal_eval(x)
                if isinstance(items, list):
                    return ", ".join(map(str, items))
            except:
                pass
        return x

    for col in df.select_dtypes(include='object').columns:
        if df[col].str.startswith('[').any() and df[col].str.endswith(']').any():
            df[col] = df[col].apply(parse_list_str)

    # 4. Detect and convert potential date columns
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            try:
                converted = pd.to_datetime(df[col], errors='coerce')
                if converted.notna().mean() > 0.5:  # If >50% valid dates
                    df[col] = converted
            except:
                pass

    
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip()

    return df

def get_outlier_report(df):
    """
    Generic outlier report for all numeric columns using IQR method.
    Returns a summary string.
    """
    outlier_summary = []
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) == 0:
        return "No numeric columns found for outlier detection."

    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue  # Skip if no variation
        outliers = df[(df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))]
        if not outliers.empty:
            outlier_summary.append(f"{col}: {len(outliers)} outliers detected.")

    if outlier_summary:
        return "\n".join(outlier_summary)
    else:
        return "No outliers detected in numeric columns."
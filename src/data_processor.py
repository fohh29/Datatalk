#Module 1 — Data Preparation
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def clean_data(df):
    """
    Professional cleaning and preparation.
    Fulfills: Column standardization & Missing value handling.
    """
    # 1. Standardize column names (lowercase and underscores)
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    
    # 2. Handle missing values (Fill numeric with median, categoric with mode)
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64', 'int32', 'float32']:
            df[col] = df[col].fillna(df[col].median())
        else:
            if not df[col].mode().empty:
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna("Unknown")
    return df

def get_outlier_report(df):
    """
    Fulfills: Outlier detection using the IQR method as per PDF requirements.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    report = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Count how many rows are outside these bounds
        outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        report[col] = outliers_count
    return report

def get_summary_stats(df):
    """Generates statistics for the EDA Dashboard."""
    return df.describe(), df.isnull().sum()

def get_eda_plots(df):
    """
    Generates a correlation heatmap for the EDA Dashboard.
    Fulfills: Exploratory Data Analysis requirement.
    """
    # Only calculate correlation for numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    
    if not numeric_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        plt.title("Correlation Heatmap")
        return fig
    return None

def get_missing_report(df):
    """Returns a summary of missing values."""
    return df.isnull().sum()
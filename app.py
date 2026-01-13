# The UI Shell
import streamlit as st
import pandas as pd
from src.data_processor import clean_data, get_outlier_report

st.set_page_config(page_title="DataTalk AI", layout="wide")

# Sidebar Navigation
st.sidebar.title("📁 Navigation")
menu = st.sidebar.radio("Go to:", ["Introduction", "Upload & Clean", "Chat with Data", "About Creator"])

if menu == "Introduction":
    st.title("🤖 DataTalk: Conversational Analytics")
    st.write("Welcome to DataTalk. Upload a dataset to begin your intelligent analysis.")
    st.markdown("""
    **Core Capabilities:**
    - **Automated Cleaning:** Fixes missing values and standardizes data.
    - **Smart Insights:** Ask questions in plain English.
    - **Outlier Detection:** Uses the IQR method to find data anomalies.
    """)

elif menu == "Upload & Clean":
    st.header("📤 Upload & Prepare Data")
    file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if file:
        df = pd.read_csv(file)
        st.write("### Raw Data Preview", df.head())
        
        if st.button("Run Professional Cleaning"):
            # This calls your logic from src/data_processor.py
            cleaned_df = clean_data(df)
            outliers = get_outlier_report(cleaned_df)
            st.session_state['data'] = cleaned_df
            st.success("Cleaning complete! Data is now standardized.")
            st.write("### Outlier Report (IQR Method):", outliers)

elif menu == "Chat with Data":
    st.header("💬 Chat with your Dataset")
    api_key = st.text_input("Enter OpenAI API Key", type="password")
    
    if 'data' in st.session_state:
        user_query = st.text_input("Ask a question (e.g., 'Which column has the highest values?')")
        if st.button("Ask AI"):
            from src.llm_engine import chat_with_data
            result = chat_with_data(st.session_state['data'], user_query, api_key)
            st.write("### Result:", result)
    else:
        st.warning("Please upload and clean data first!")

elif menu == "About Creator":
    st.header("👤 Project Developer")
    st.write("Built for the DataTalk Capstone Project.")
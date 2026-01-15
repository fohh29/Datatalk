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
            cleaned_df = clean_data(df)
            outliers = get_outlier_report(cleaned_df)
            st.session_state['data'] = cleaned_df  # We save it as 'data'
            st.success("Cleaning complete! Data is now standardized.")
            st.write("### Outlier Report (IQR Method):", outliers)

elif menu == "Chat with Data":
    st.header("💬 Chat with your Dataset")
    
    # We check if 'data' exists in session state (from the previous tab)
    if 'data' in st.session_state:
        user_query = st.text_input("Ask a question (e.g., 'Plot a bar chart of top 5 games by rating')")
        
        if st.button("Ask AI"):
            if user_query:
                from src.llm_engine import chat_with_data
                with st.spinner("AI is thinking..."):
                    # Call the function 
                    response = chat_with_data(st.session_state['data'], user_query)
                    
                    # THE VISUALIZATION :
                    if isinstance(response, str) and (".png" in response or ".jpg" in response):
                        st.image(response)
                        st.success("Chart generated successfully!")
                    else:
                        st.write("### Result:")
                        st.write(response)
    else:
        st.warning("Please upload and clean data first in the 'Upload & Clean' tab!")

elif menu == "About Creator":
    st.header("👤 Project Developer")
    st.write("Built for the DataTalk Capstone Project.")
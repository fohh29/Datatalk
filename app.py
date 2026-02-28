# The UI Shell
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from src.data_processor import clean_data, get_outlier_report
from src.llm_engine import chat_with_data

st.set_page_config(page_title="DataTalk AI", layout="wide")

# Sidebar Navigation
st.sidebar.title("📁 Navigation")
menu = st.sidebar.radio("Go to:", ["Introduction", "Upload & Clean", "EDA Dashboard", "Auto Visualizations", "Chat with Data", "About Creator"])

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
        
        # Display Raw Data initially
        st.write("### Raw Data Preview")
        st.dataframe(df.head())
        
        if st.button("Run Professional Cleaning"):
            with st.spinner("Standardizing data..."):
                # 1. Clean the data using your module logic
                cleaned_df = clean_data(df)
                outliers = get_outlier_report(cleaned_df)
                
                # 2. Store the CLEANED version in session state
                st.session_state['data'] = cleaned_df
                
                st.success("Cleaning complete! Data is now standardized.")
                
                # 3. DISPLAY THE CLEANED DATA (This removes the 'K' values from view)
                st.write("### Cleaned Data Preview")
                st.dataframe(st.session_state['data'].head())
                
                # 4. Show the Outlier Report
                st.info(f"📊 {outliers}")

elif menu == "EDA Dashboard":
    st.header("📊 EDA Dashboard")
    
    if 'data' in st.session_state:
        df = st.session_state['data']
        
        # Summary Statistics
        st.write("### Summary Statistics")
        st.dataframe(df.describe())
        
        # Numerical Analysis
        st.write("### Numerical Columns Analysis")
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        if not numeric_df.empty:
        # Built-in aggs
            built_in_stats = numeric_df.agg(['min', 'max', 'mean', 'std']).T

            # Custom stats (skew and kurtosis)
            custom_stats = numeric_df.apply(lambda x: pd.Series({'Skewness': skew(x, nan_policy='omit'), 'Kurtosis': kurtosis(x, nan_policy='omit')})).T

            # Combine them
            num_analysis = pd.concat([built_in_stats, custom_stats], axis=1)
            num_analysis.columns = ['Min', 'Max', 'Mean', 'Std Dev', 'Skewness', 'Kurtosis']
            st.dataframe(num_analysis)
        else:
            st.info("No numerical columns available.")
        
        # Categorical Analysis
        st.write("### Categorical Columns Analysis")
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            with st.expander(f"Analysis for {col}"):
                st.write("Unique Values:", df[col].nunique())
                st.write("Value Counts:")
                st.dataframe(df[col].value_counts().head(10))  # Top 10
        
        # Correlation Matrix and Heatmap
        st.write("### Correlation Matrix")
        if not numeric_df.empty:
            corr = numeric_df.corr()
            st.dataframe(corr)
            
            st.write("### Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        else:
            st.info("No numeric columns available for correlation.")
        
        # Missing Values Report
        st.write("### Missing Values Report")
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            st.write(missing)
        else:
            st.info("No missing values detected.")
        
        # Outlier Detection
        st.write("### Outlier Report")
        outliers = get_outlier_report(df)
        st.info(outliers)
        
        # Distribution Plots (Column-wise)
        st.write("### Column-wise Distributions")
        all_cols = df.columns.tolist()
        selected_col = st.selectbox("Select a column for distribution plot", all_cols)
        if selected_col:
            if pd.api.types.is_numeric_dtype(df[selected_col]):
                fig, ax = plt.subplots()
                sns.histplot(df[selected_col], kde=True, ax=ax)
                ax.set_title(f"Distribution of {selected_col}")
                st.pyplot(fig)
            else:
                fig, ax = plt.subplots()
                top_categories = df[selected_col].value_counts().nlargest(10)
                sns.barplot(x=top_categories.values, y=top_categories.index, ax=ax)
                ax.set_title(f"Top Categories in {selected_col}")
                st.pyplot(fig)
        
        # Interactive Chat Window in EDA
        st.write("### Ask Questions About Your Data")
        user_query = st.text_input("Type your question here (e.g., 'What are the key insights from this data?')", key="eda_query")
        
        if st.button("Ask AI", key="eda_ask"):
            if user_query:
                with st.spinner("AI is thinking..."):
                    response = chat_with_data(df, user_query)
                    st.markdown("### 🤖 AI Response:")
                    st.write(response)
    else:
        st.warning("Please upload and clean data first in the 'Upload & Clean' tab!")

elif menu == "Auto Visualizations":
    st.header("📈 Auto Visualizations")
    
    if 'data' in st.session_state:
        df = st.session_state['data']
        
        # Interactive Chart Selection
        st.write("### Create Custom Visualizations")
        chart_type = st.selectbox("Select Chart Type", ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram", "Boxplot", "Pie Chart"])
        
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        all_cols = df.columns.tolist()
        
        if chart_type in ["Bar Chart", "Line Chart", "Scatter Plot"]:
            x_col = st.selectbox("Select X-axis", all_cols)
            y_col = st.selectbox("Select Y-axis", numeric_cols)
            if st.button("Generate Chart"):
                fig, ax = plt.subplots(figsize=(10, 6))
                if chart_type == "Bar Chart":
                    sns.barplot(data=df, x=x_col, y=y_col, ax=ax)
                elif chart_type == "Line Chart":
                    sns.lineplot(data=df, x=x_col, y=y_col, ax=ax)
                elif chart_type == "Scatter Plot":
                    sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
                ax.set_title(f"{chart_type} of {y_col} vs {x_col}")
                st.pyplot(fig)
        
        elif chart_type == "Histogram":
            col = st.selectbox("Select Column", numeric_cols)
            if st.button("Generate Chart"):
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(df[col], kde=True, ax=ax)
                ax.set_title(f"Histogram of {col}")
                st.pyplot(fig)
        
        elif chart_type == "Boxplot":
            col = st.selectbox("Select Column", numeric_cols)
            if st.button("Generate Chart"):
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(x=df[col], ax=ax)
                ax.set_title(f"Boxplot of {col}")
                st.pyplot(fig)
        
        elif chart_type == "Pie Chart":
            col = st.selectbox("Select Categorical Column", df.select_dtypes(include=['object']).columns.tolist())
            if st.button("Generate Chart"):
                pie_data = df[col].value_counts()
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90)
                ax.set_title(f"Pie Chart of {col}")
                ax.axis('equal')
                st.pyplot(fig)
        
        # Auto-generated Visualizations (as before)
        st.write("### Auto-Generated Visualizations")
        
        # Auto histograms for numerics
        st.write("#### Histograms for Numeric Columns")
        for col in numeric_cols:
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax)
            ax.set_title(f"Histogram of {col}")
            st.pyplot(fig)
        
        # Auto bar charts for categoricals (top 10)
        st.write("#### Bar Charts for Categorical Columns")
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].nunique() <= 50:
                fig, ax = plt.subplots()
                top_categories = df[col].value_counts().nlargest(10)
                sns.barplot(x=top_categories.values, y=top_categories.index, ax=ax)
                ax.set_title(f"Top 10 Categories in {col}")
                st.pyplot(fig)
            else:
                st.info(f"Skipping {col} due to too many unique values.")
        
        # Auto boxplots for numerics
        st.write("#### Boxplots for Numeric Columns")
        for col in numeric_cols:
            fig, ax = plt.subplots()
            sns.boxplot(x=df[col], ax=ax)
            ax.set_title(f"Boxplot of {col}")
            st.pyplot(fig)
    else:
        st.warning("Please upload and clean data first in the 'Upload & Clean' tab!")

elif menu == "Chat with Data":
    st.header("💬 Chat with your Dataset")
    
    if 'data' in st.session_state:
        user_query = st.text_input("Ask a question (e.g., 'Plot a bar chart of top 5 games by rating')")
        
        if st.button("Ask AI"):
            if user_query:
                with st.spinner("AI is thinking..."):
                    response = chat_with_data(st.session_state['data'], user_query)
                    st.markdown("### 🤖 AI Response:")
                    st.write(response)
    else:
        st.warning("Please upload and clean data first in the 'Upload & Clean' tab!")
elif menu == "About Creator":
    st.header("👤 Project Developer")
    st.write("### Name: Fouziya")
    st.write("### Project: DataTalk AI")
    st.write("### Course: AI-ML Development Course")
    
    st.divider()
    
    st.markdown("""
    #### 🚀 About the Project
    **DataTalk** was developed to bridge the gap between complex raw data and actionable insights. 
    By combining **Streamlit** for the interface and **OpenAI's LLM** for reasoning, users can 
    simply talk to their data to generate professional-grade visualizations and cleaning reports.
    
    #### 🛠️ Tech Stack
    - **Frontend:** Streamlit
    - **Logic:** Pandas & Python
    - **AI Engine:** PandasAI & OpenAI GPT
    - **Visualization:** Matplotlib / Seaborn
    """)
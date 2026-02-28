

import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # Pre-import seaborn
import io
import contextlib
import re

def chat_with_data(df, query):
    
    stats_summary = df.describe(include='all').to_string()
    cols = df.columns.tolist()

    # 2. API CONFIGURATION
    API_URL = "https://router.huggingface.co/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {st.secrets['HF_TOKEN']}",
        "Content-Type": "application/json"
    }

    # Improved System Prompt
    system_prompt = f"""
    You are an Expert Data Analyst. You have a DataFrame 'df' with {len(df)} rows.
    COLUMNS: {cols}
    DATA STATS: {stats_summary}

    TASKS:
    1. First, provide a clear textual explanation or answer to the user's question, using exact numbers from DATA STATS where possible.
    2. If a visualization is appropriate or requested, generate Python code in a single ```python block after the explanation.
    3. AUTO-VIZ RULES: 
       - Categories/Comparison -> Bar Chart (use sns.barplot for better visuals)
       - Distributions -> Histogram (sns.histplot)
       - Relationship/Correlation -> Scatter Plot (sns.scatterplot)
       - Statistical Spread/Outliers -> Boxplot (sns.boxplot)
    4. Use 'plt', 'sns', and 'df' in the code. Do NOT include import statements or plt.show(). Plot directly using sns or plt functions.
    5. If needed, after the code block, add any additional explanation.
    """

    payload = {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        "temperature": 0.1
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()  
        result = response.json()
        ai_message = result['choices'][0]['message']['content']

        # Parse the response to separate explanation and code
        code = None
        code_match = re.search(r"```python(.*?)```", ai_message, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
            # Remove any plt.show() if present
            code = code.replace("plt.show()", "")
            
            # Split the message into text parts (before, after code)
            text_parts = re.split(r"```python.*?```", ai_message, flags=re.DOTALL)
            explanation = "\n".join(part.strip() for part in text_parts if part.strip())
        else:
            explanation = ai_message.strip()

        # 3. EXECUTION ENGINE if code is present
        calc_text = ""
        if code:
            # Redirect stdout for prints
            output_buffer = io.StringIO()
            with contextlib.redirect_stdout(output_buffer):
                try:
                    # Clear previous plots
                    plt.close('all')
                    # Create a fresh figure
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Limited globals for safe exec
                    exec_globals = {
                        "df": df,
                        "pd": pd,
                        "plt": plt,
                        "sns": sns,
                        "fig": fig,
                        "ax": ax
                    }
                    
                    # Execute the code
                    exec(code, exec_globals)
                    
                    # Display the plot if one was created
                    if plt.get_fignums():
                        st.write("### 📊 Auto-Generated Visualization")
                        st.pyplot(fig)
                        plt.close(fig)
                except Exception as e:
                    st.error(f"Visualization Error: {str(e)}")
                    explanation += f"\n\n(Note: There was an error generating the visualization: {str(e)})"
            
            # Capture any printed output (e.g., tables, calculations)
            calc_text = output_buffer.getvalue().strip()
            if calc_text:
                with st.expander("📝 View Detailed Calculations"):
                    st.text(calc_text)

        return explanation

    except requests.exceptions.RequestException as e:
        return f"Error connecting to the AI service: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"
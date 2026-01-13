# Module 2 — The LLM Engine

import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import OpenAI

def chat_with_data(df, query, api_key):
    """
    Core Logic: Turns Natural Language into Data Insights.
    Fulfills: LLM Integration & Prompt Engineering requirements.
    """
    try:
        # 1. Setup the LLM (Large Language Model)
        # We use the API key provided by the user in the UI
        llm = OpenAI(api_token=api_key)
        
        # 2. Initialize the SmartDataframe
        # This handles the 'Prompt Engineering' by automatically sending 
        # the dataframe schema to the AI so it understands the columns.
        smart_df = SmartDataframe(df, config={"llm": llm})
        
        # 3. Process the query and get the result
        response = smart_df.chat(query)
        
        # If the AI fails to generate an answer, return a helpful message
        if response is None:
            return "I couldn't find an answer for that. Try rephrasing your question!"
            
        return response

    except Exception as e:
        return f"System Error: {str(e)}. Please check your API Key and Internet connection."
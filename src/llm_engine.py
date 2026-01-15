# Module 2 — The LLM Engine

import streamlit as st
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
import os

def chat_with_data(df, query):
    api_key = st.secrets["OPENAI_API_KEY"]
    llm = OpenAI(api_token=api_key)
    
    # We create a folder for the charts if it doesn't exist
    if not os.path.exists("exports/charts"):
        os.makedirs("exports/charts")

    sdf = SmartDataframe(df, config={
        "llm": llm,
        "save_charts": True,
        "save_charts_path": "exports/charts/", # This is where the image goes
        "open_charts": False,
        "verbose": True
    })
    
    response = sdf.chat(query)
    return response
# src/ui/streamlit_app.py
from __future__ import annotations

import os

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="AI/ML Governance Demo", layout="centered")

st.title("AI/ML Governance Demo")

st.markdown(
    """
This small demo application lets you:

1. Call a **deployed ML model** (Iris classifier) via FastAPI  
2. Ask a simple **RAG + LLM** endpoint questions about AI governance / ML lifecycle  
"""
)

# --- Prediction Section ---
st.header("1. Supervised Model Prediction")

with st.form("prediction_form"):
    f1 = st.number_input("Feature 1", value=5.1)
    f2 = st.number_input("Feature 2", value=3.5)
    f3 = st.number_input("Feature 3", value=1.4)
    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {"feature1": f1, "feature2": f2, "feature3": f3}
    try:
        resp = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        st.success(f"Prediction: {data['prediction']} (prob={data['probability']:.3f})")
    except Exception as e:
        st.error(f"Error calling prediction API: {e}")

# --- RAG Section ---
st.header("2. RAG Question Answering (LLM + KB)")

question = st.text_input(
    "Ask a question about AI governance or the ML lifecycle:",
    placeholder="e.g., What is model monitoring?",
)

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        try:
            resp = requests.post(f"{API_BASE_URL}/rag", json={"question": question}, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            st.write("**Answer:**")
            st.write(data["answer"])
            if data.get("sources"):
                st.caption(f"Sources: {', '.join(data['sources'])}")
        except Exception as e:
            st.error(f"Error calling RAG API: {e}")

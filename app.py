import streamlit as st
import joblib

try:
    model = joblib.load("model.pkl")
except Exception as e:
    st.error("❌ Failed to load the model. Please check if 'model.pkl' exists and is compatible.")
    st.stop()

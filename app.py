import streamlit as st
import joblib
import pandas as pd

# Load the model
try:
    model = joblib.load("model.pkl")
except Exception as e:
    st.error("❌ Failed to load the model. Please check if 'model.pkl' exists and is compatible.")
    st.stop()

# Title
st.title("🚧 Traffic Accident Severity Predictor")

# Input widgets
speed_limit = st.slider("Speed Limit (km/h)", min_value=0, max_value=120, value=50, step=5)
weather_condition = st.selectbox("Weather Condition", ["Clear", "Adverse"])
vehicle_age = st.number_input("Vehicle Age (years)", min_value=0, max_value=50, value=5)

# Map weather to numeric code
weather_code = 0 if weather_condition == "Clear" else 1

# Predict button
if st.button("Predict Severity"):
    input_data = pd.DataFrame([[speed_limit, weather_code, vehicle_age]],
                              columns=["Speed_Limit", "Weather_Condition", "Vehicle_Age"])
    prediction = model.predict(input_data)[0]

    st.success(f"🧠 Predicted Accident Severity: **{prediction}**")

    severity_map = {
        0: "Low severity",
        1: "Moderate severity",
        2: "High severity"
    }
    st.write(f"📊 Interpretation: {severity_map.get(prediction, 'Unknown')}")

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

@st.cache_resource
def train_model():
    # Dummy dataset (can replace with real data)
    df = pd.DataFrame({
        'Speed_Limit': [30, 50, 80, 60],
        'Weather_Condition': [0, 1, 1, 0],
        'Vehicle_Age': [5, 3, 10, 2],
        'Accident_Severity': [0, 1, 2, 0]
    })

    X = df.drop('Accident_Severity', axis=1)
    y = df['Accident_Severity']

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    return model

# Train model (only once thanks to caching)
model = train_model()

# Title
st.title("🚧 Traffic Accident Severity Predictor (Trained on Host)")

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

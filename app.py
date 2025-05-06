import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

@st.cache_resource
def train_model():
    # Create dummy dataset
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

# Train model once on startup
model = train_model()

# Streamlit interface
st.title("🚧 Accident Severity Predictor (Trained at Runtime)")

speed_limit = st.slider("Speed Limit (km/h)", 0, 120, 50)
weather_condition = st.selectbox("Weather Condition", ["Clear", "Adverse"])
vehicle_age = st.number_input("Vehicle Age (years)", 0, 50, 5)

weather_code = 0 if weather_condition == "Clear" else 1

if st.button("Predict Severity"):
    input_df = pd.DataFrame([[speed_limit, weather_code, vehicle_age]],
                             columns=["Speed_Limit", "Weather_Condition", "Vehicle_Age"])
    prediction = model.predict(input_df)[0]

    severity_text = {
        0: "Low severity",
        1: "Moderate severity",
        2: "High severity"
    }.get(prediction, "Unknown")

    st.success(f"🧠 Predicted Severity: {prediction} — {severity_text}")

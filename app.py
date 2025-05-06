import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load("model.pkl")

# Streamlit App
st.set_page_config(page_title="AI-Driven Traffic Accident Predictor", layout="centered")

st.title("🚦 AI-Driven Traffic Accident Analysis & Prediction")
st.markdown("Upload traffic data or enter details to predict accident severity and enhance safety.")

# Upload CSV data
uploaded_file = st.file_uploader("Upload CSV file for analysis", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data")
    st.write(data.head())

    # Visualize
    st.subheader("📊 Data Visualization")
    selected_col = st.selectbox("Choose column for distribution plot", data.columns)
    fig, ax = plt.subplots()
    sns.histplot(data[selected_col], kde=True, ax=ax)
    st.pyplot(fig)

    # Predict
    st.subheader("🧠 Predict Accident Severity")
    if "Accident_Severity" in data.columns:
        features = data.drop("Accident_Severity", axis=1)
        predictions = model.predict(features)
        data["Predicted_Severity"] = predictions
        st.write(data[["Predicted_Severity"]].head())
        st.success("Prediction completed!")
    else:
        st.warning("Data must include all required feature columns.")
else:
    st.subheader("📝 Manual Input for Prediction")
    feature_1 = st.slider("Speed Limit (km/h)", 0, 150, 50)
    feature_2 = st.selectbox("Weather Condition", [0, 1, 2])  # e.g., encoded values
    feature_3 = st.slider("Vehicle Age (years)", 0, 30, 5)

    input_data = pd.DataFrame([[feature_1, feature_2, feature_3]],
                              columns=["Speed_Limit", "Weather_Condition", "Vehicle_Age"])

    if st.button("Predict Severity"):
        prediction = model.predict(input_data)
        severity_map = {0: "Low", 1: "Medium", 2: "High"}
        st.success(f"Predicted Accident Severity: {severity_map[prediction[0]]}")

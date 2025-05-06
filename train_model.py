import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def train_and_save_model():
    # Create dummy dataset (replace with real data for production)
    df = pd.DataFrame({
        'Speed_Limit': [30, 50, 80, 60],
        'Weather_Condition': [0, 1, 1, 0],
        'Vehicle_Age': [5, 3, 10, 2],
        'Accident_Severity': [0, 1, 2, 0]
    })

    # Split into features and target
    X = df.drop('Accident_Severity', axis=1)
    y = df['Accident_Severity']

    # Train the model
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    # Save the trained model
    model_path = 'model.pkl'
    joblib.dump(model, model_path)

    if os.path.exists(model_path):
        print(f"✅ Model saved successfully as '{model_path}'")
    else:
        print("❌ Failed to save the model.")

if __name__ == "__main__":
    train_and_save_model()

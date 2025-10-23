import streamlit as st
import pandas as pd
import numpy as np
import gdown
import joblib
import os

# -----------------------
# 1. Load the model
# -----------------------
# Replace with your Google Drive file ID
DRIVE_FILE_ID = "1V4OA2T9-6Ya2VkVO3p1rXKGlp-P1uEUs"
MODEL_PATH = "vehicle_price_model.jolib"

# Download the .jolib file from Drive if not already exists
if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

# Load the model
model = jolib.load(MODEL_PATH)

# -----------------------
# 2. Load dataset for widget options
# -----------------------
# Replace this with your actual CSV or dataframe
df = pd.read_csv("vehicle_data.csv")

# -----------------------
# 3. Streamlit App UI
# -----------------------
st.title("🔮 Vehicle Price Predictor")

st.markdown("Fill in the vehicle details below:")

make = st.selectbox("Make:", sorted(df['make'].dropna().unique()))
fuel = st.selectbox("Fuel Type:", sorted(df['fuel'].dropna().unique()))
transmission = st.selectbox("Transmission:", sorted(df['transmission'].dropna().unique()))
body = st.selectbox("Body Type:", sorted(df['body'].dropna().unique()))
drivetrain = st.selectbox("Drivetrain:", sorted(df['drivetrain'].dropna().unique()))
cylinders = st.number_input("Cylinders:", min_value=2, max_value=16, value=4, step=1)
doors = st.number_input("Doors:", min_value=2, max_value=6, value=4, step=1)
mileage = st.number_input("Mileage (in km):", min_value=0, value=30000, step=1000)
car_age = st.number_input("Car Age (in years):", min_value=0, value=2, step=1)

if st.button("Predict Price 🚗"):
    # Create input dataframe
    new_data = pd.DataFrame([{
        'make': make,
        'fuel': fuel,
        'transmission': transmission,
        'body': body,
        'drivetrain': drivetrain,
        'cylinders': cylinders,
        'doors': doors,
        'mileage': mileage,
        'car_age': car_age
    }])

    # Predict
    pred_log = model.predict(new_data)[0]
    pred_price = np.expm1(pred_log)
    st.success(f"Predicted Vehicle Price: ${pred_price:,.2f}")

# -----------------------
# Optional: Show sample predictions
# -----------------------
st.markdown("### 📊 Sample predictions from test set:")
X_test = df.drop(columns=['price'])  # Replace with your actual test set
sample_indices = X_test.sample(3, random_state=42).index
for idx in sample_indices:
    actual_price = df.loc[idx, 'price']
    sample_data = X_test.loc[idx:idx]
    pred_log = model.predict(sample_data)[0]
    pred_price = np.expm1(pred_log)
    error_pct = ((pred_price - actual_price) / actual_price) * 100
    st.write(f"Actual: ${actual_price:,.2f} | Predicted: ${pred_price:,.2f} | Error: {error_pct:+.1f}%")

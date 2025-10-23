import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import io

# ----------------------------
# 🔹 1. Load Model from Google Drive
# ----------------------------
# Replace with your Google Drive file ID for the vehicle price model
MODEL_FILE_ID = "1V4OA2T9-6Ya2VkVO3p1rXKGlp-P1uEUs"
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"

@st.cache_resource
def load_model():
    response = requests.get(MODEL_URL)
    if response.status_code != 200:
        st.error("⚠️ Unable to load model from Google Drive.")
        return None
    model = joblib.load(io.BytesIO(response.content))
    return model

model = load_model()
if model is None:
    st.stop()

# ----------------------------
# 🔹 2. Load Dataset from Google Drive
# ----------------------------
# Replace with your Google Drive file ID for d1.csv
CSV_FILE_ID = "YOUR_CSV_DRIVE_ID_HERE"
CSV_URL = f"https://drive.google.com/uc?id={CSV_FILE_ID}"

@st.cache_data
def load_data():
    response = requests.get(CSV_URL)
    if response.status_code != 200:
        st.error("⚠️ Unable to load CSV from Google Drive.")
        return None
    df = pd.read_csv(io.BytesIO(response.content))
    return df

df = load_data()
if df is None:
    st.stop()

# ----------------------------
# 🔹 3. App Title
# ----------------------------
st.title("🔮 Vehicle Price Predictor")
st.write("Predict the estimated price of a vehicle using its features.")

# ----------------------------
# 🔹 4. Collect Inputs
# ----------------------------
st.sidebar.header("Enter Vehicle Details")

make = st.sidebar.selectbox("Make:", sorted(df['make'].dropna().unique()))
fuel = st.sidebar.selectbox("Fuel Type:", sorted(df['fuel'].dropna().unique()))
transmission = st.sidebar.selectbox("Transmission:", sorted(df['transmission'].dropna().unique()))
body = st.sidebar.selectbox("Body Type:", sorted(df['body'].dropna().unique()))
drivetrain = st.sidebar.selectbox("Drivetrain:", sorted(df['drivetrain'].dropna().unique()))
cylinders = st.sidebar.number_input("Cylinders:", min_value=2, max_value=16, value=4, step=1)
doors = st.sidebar.number_input("Doors:", min_value=2, max_value=6, value=4, step=1)
mileage = st.sidebar.number_input("Mileage (km):", min_value=0, value=30000, step=1000)
car_age = st.sidebar.number_input("Car Age (years):", min_value=0, value=2, step=1)

# ----------------------------
# 🔹 5. Prepare Input
# ----------------------------
input_data = {
    "make": [make],
    "fuel": [fuel],
    "transmission": [transmission],
    "body": [body],
    "drivetrain": [drivetrain],
    "cylinders": [cylinders],
    "doors": [doors],
    "mileage": [mileage],
    "car_age": [car_age]
}

input_df = pd.DataFrame(input_data)

# ----------------------------
# 🔹 6. Prediction
# ----------------------------
if st.button("🔍 Predict Price"):
    pred_log = model.predict(input_df)[0]
    pred_price = np.expm1(pred_log)
    st.success(f"🚗 Predicted Vehicle Price: ${pred_price:,.2f}")

    st.write("### Input Summary:")
    st.dataframe(input_df)

# ----------------------------
# 🔹 7. Footer
# ----------------------------
st.write("---")
st.caption("Developed with ❤️ using Streamlit and Random Forest Regressor")

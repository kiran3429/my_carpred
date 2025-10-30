import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import io
MODEL_FILE_ID = "1V4OA2T9-6Ya2VkVO3p1rXKGlp-P1uEUs"
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
@st.cache_resource
def load_model():
    try:
        response = requests.get(MODEL_URL)
        if response.status_code != 200:
            st.error("⚠️ Unable to load model from Google Drive.")
            return None
        model = joblib.load(io.BytesIO(response.content))
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None
model = load_model()
if model is None:
    st.stop()
CSV_FILE_ID = "19ijlwqFSr7nVCPzPUh4tLuMFKuFFl4q-"  
CSV_URL = f"https://drive.google.com/uc?id={CSV_FILE_ID}"
@st.cache_data
def load_data():
    try:
        response = requests.get(CSV_URL)
        if response.status_code != 200:
            st.error("⚠️ Unable to load CSV from Google Drive.")
            return None
        df = pd.read_csv(io.BytesIO(response.content))
        st.success("✅ Dataset loaded successfully!")
        return df
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")
        return None
df = load_data()
if df is None:
    st.stop()
st.title("🚗 Vehicle Price Predictor")
st.write("Predict the estimated price of a vehicle using its features.")
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
if st.button("🔍 Predict Price"):
    try:
        pred_log = model.predict(input_df)[0]
        pred_price = np.expm1(pred_log)
        st.success(f"💰 **Predicted Vehicle Price: ${pred_price:,.2f}**")
        st.write("Input Summary:")
        st.dataframe(input_df, use_container_width=True)
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
with st.expander("Preview Dataset"):
    st.write(f"Dataset shape: {df.shape}")
    st.dataframe(df.head(), use_container_width=True)
# 🔹 8. Footer
# ----------------------------
st.write("---")
st.caption("Developed with ❤️ using Streamlit and Random Forest Regressor")

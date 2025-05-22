import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("crop_recommendation_model.pkl")

st.set_page_config(page_title="Crop Recommendation", layout="centered")
st.title("Crop Recommendation System")

st.markdown("Enter the soil and climate values below:")

# Input fields
n = st.number_input("Nitrogen (N)", min_value=0.0)
p = st.number_input("Phosphorus (P)", min_value=0.0)
k = st.number_input("Potassium (K)", min_value=0.0)
temp = st.number_input("Temperature (°C)", min_value=0.0)
humidity = st.number_input("Humidity (%)", min_value=0.0)
ph = st.number_input("pH", min_value=0.0)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0)

if st.button("Recommend Crop"):
    features = np.array([[n, p, k, temp, humidity, ph, rainfall]])
    prediction = model.predict(features)
    st.success(f"Recommended Crop: *{prediction[0]}*")
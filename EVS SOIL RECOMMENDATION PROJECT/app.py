import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("crop_recommendation_model.pkl")

# App title
st.title("🌾 AI Crop Recommendation System")

# User inputs
st.header("Enter Soil Report Values")

N = st.number_input("Nitrogen (N)")
P = st.number_input("Phosphorus (P)")
K = st.number_input("Potassium (K)")
temperature = st.number_input("Temperature (°C)")
humidity = st.number_input("Humidity (%)")
ph = st.number_input("pH Level")
rainfall = st.number_input("Rainfall (mm)")

# Predict
if st.button("Recommend Crop"):
    data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                        columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
    prediction = model.predict(data)[0]
    st.success(f"🌿 Recommended Crop: **{prediction.upper()}**")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load trained model
model = joblib.load("logistic_regression_model.pkl")
scaler = joblib.load("scaler_lr.pkl")

st.title("Heart Disease Prediction App")
st.write("Enter the details below to predict if a person has heart disease.")

# User input fields
age = st.number_input("Age", min_value=20, max_value=100, value=50)
sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
cpt = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=250, value=150)
exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

# Prepare input data
input_data = np.array([[age, sex, cpt, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
input_data_scaled = scaler.transform(input_data)

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_data_scaled)
    result = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease"
    st.subheader(f"Prediction: {result}")

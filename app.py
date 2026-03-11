import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model components
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")
selector = joblib.load("selector.pkl")

st.title("❤️ Heart Disease Prediction App")

st.write("Enter patient details to predict heart disease risk.")

# Input fields
age = st.number_input("Age", 20, 100)
sex = st.selectbox("Sex", [0,1])
cp = st.selectbox("Chest Pain Type", [0,1,2,3])
trestbps = st.number_input("Resting Blood Pressure")
chol = st.number_input("Cholesterol")
fbs = st.selectbox("Fasting Blood Sugar > 120", [0,1])
restecg = st.selectbox("Rest ECG", [0,1,2])
thalach = st.number_input("Max Heart Rate")
exang = st.selectbox("Exercise Induced Angina", [0,1])
oldpeak = st.number_input("Oldpeak")
slope = st.selectbox("Slope", [0,1,2])
ca = st.selectbox("Number of Major Vessels", [0,1,2,3])
thal = st.selectbox("Thal", [0,1,2,3])

# Convert to dataframe
data = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,
                  thalach,exang,oldpeak,slope,ca,thal]])

data = scaler.transform(data)
data = selector.transform(data)

# Prediction
if st.button("Predict"):
    
    prediction = model.predict(data)
    
    if prediction[0] == 1:
        st.error("⚠️ High risk of Heart Disease")
    else:
        st.success("✅ Low risk of Heart Disease")

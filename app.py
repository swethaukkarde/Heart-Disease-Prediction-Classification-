import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Load saved model components
# -----------------------------

model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")
selector = joblib.load("selector.pkl")

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")

st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details to predict the risk of heart disease.")

# -----------------------------
# User Inputs
# -----------------------------

age = st.slider("Age", 20, 100, 50)
sex = st.selectbox("Sex", ["Female", "Male"])
cp = st.selectbox("Chest Pain Type", [0,1,2,3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar >120 mg/dl", [0,1])
restecg = st.selectbox("Rest ECG", [0,1,2])
thalach = st.number_input("Maximum Heart Rate", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina", [0,1])
oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope", [0,1,2])
ca = st.selectbox("Number of Major Vessels", [0,1,2,3])
thal = st.selectbox("Thal", [0,1,2,3])

# Convert sex to numeric
sex = 1 if sex == "Male" else 0

# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict Heart Disease Risk"):

    columns = [
        'age','sex','cp','trestbps','chol','fbs','restecg',
        'thalach','exang','oldpeak','slope','ca','thal'
    ]

    input_data = pd.DataFrame(
        [[age,sex,cp,trestbps,chol,fbs,restecg,
          thalach,exang,oldpeak,slope,ca,thal]],
        columns=columns
    )

    # Apply preprocessing
    input_scaled = scaler.transform(input_data)
    input_selected = selector.transform(input_scaled)

    prediction = model.predict(input_selected)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

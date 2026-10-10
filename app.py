
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# -----------------------------
# Light Theme Styling
# -----------------------------
st.markdown("""
<style>
.stApp {
    background-color: #F4F8FC;
    color: #243247;
}

header[data-testid="stHeader"] {
    background-color: #F4F8FC;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1150px;
}

.hero {
    background: linear-gradient(120deg, #FFFFFF, #E7F2FF);
    padding: 35px;
    border-radius: 18px;
    border: 1px solid #DCE7F3;
    margin-bottom: 28px;
}

.hero h1 {
    color: #153E75;
    font-size: 38px;
    margin-bottom: 12px;
}

.hero p {
    color: #43566E;
    font-size: 16px;
    line-height: 1.7;
}

.section-title {
    color: #153E75;
    font-size: 25px;
    font-weight: 700;
    margin-bottom: 8px;
}

.info-card {
    background: #FFFFFF;
    border: 1px solid #DCE7F3;
    border-radius: 14px;
    padding: 22px;
    margin-bottom: 22px;
}

div[data-testid="stForm"] {
    background-color: #FFFFFF;
    border: 1px solid #DCE7F3;
    border-radius: 16px;
    padding: 24px;
}

div[data-testid="stWidgetLabel"] p {
    color: #243247;
    font-weight: 600;
}

div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] div[data-baseweb="select"] {
    background-color: #FFFFFF;
}

.stButton > button,
div[data-testid="stFormSubmitButton"] > button {
    background-color: #1769AA;
    color: white;
    border-radius: 9px;
    border: none;
    padding: 10px 22px;
    font-weight: 600;
    width: 100%;
}

.stButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    background-color: #10528A;
    color: white;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Saved Model Components
# -----------------------------
@st.cache_resource
def load_components():
    model = joblib.load("heart_model.pkl")
    scaler = joblib.load("scaler.pkl")
    selector = joblib.load("selector.pkl")
    return model, scaler, selector

try:
    model, scaler, selector = load_components()
except Exception as e:
    st.error(f"Unable to load the saved model files: {e}")
    st.stop()

# -----------------------------
# Hero Section
# -----------------------------
st.markdown("""
<div class="hero">
    <h1>❤️ Heart Disease Prediction</h1>
    <p>
        Understand potential heart disease risk using a
        machine-learning-based prediction system.
    </p>
    <p>
        The heart continuously pumps blood, delivering oxygen and
        essential nutrients throughout the body.
        Maintaining heart health is vital for overall well-being
        and reducing the risk of cardiovascular disease.
    </p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Patient Information Form
# -----------------------------
st.markdown(
    '<div class="section-title">Patient Health Information</div>',
    unsafe_allow_html=True
)

st.write(
    "Enter the patient's details below to generate a model prediction."
)

with st.form("heart_prediction_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 20, 100, 50)
        sex_choice = st.selectbox("Sex", ["Female", "Male"])
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
        trestbps = st.number_input(
            "Resting Blood Pressure", 80, 200, 120
        )
        chol = st.number_input(
            "Cholesterol", 100, 600, 200
        )

    with col2:
        fbs = st.selectbox("Fasting Blood Sugar >120 mg/dl", [0, 1])
        restecg = st.selectbox("Rest ECG", [0, 1, 2])
        thalach = st.number_input(
            "Maximum Heart Rate", 60, 220, 150
        )
        exang = st.selectbox("Exercise-Induced Angina", [0, 1])

    with col3:
        oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0, step=0.1)
        slope = st.selectbox("Slope", [0, 1, 2])
        ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])
        thal = st.selectbox("Thal", [0, 1, 2, 3])

    submitted = st.form_submit_button(
        "Predict Heart Disease Risk"
    )

# -----------------------------
# Prediction
# -----------------------------
if submitted:

    sex = 1 if sex_choice == "Male" else 0

    columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak",
        "slope", "ca", "thal"
    ]

    input_data = pd.DataFrame(
        [[
            age, sex, cp, trestbps, chol, fbs,
            restecg, thalach, exang, oldpeak,
            slope, ca, thal
        ]],
        columns=columns
    )

    try:
        input_scaled = scaler.transform(input_data)
        input_selected = selector.transform(input_scaled)
        prediction = model.predict(input_selected)

        st.markdown("---")
        st.subheader("Prediction Result")

        if prediction[0] == 1:
            st.error(
                "Potential Heart Disease Risk Detected. "
                "Please consult a qualified healthcare professional."
            )
        else:
            st.success(
                "The model did not identify a positive result "
                "for heart disease in this input."
            )

        st.caption(
            "This prediction is for educational purposes only. "
            "It is not a medical diagnosis or a substitute for "
            "professional medical evaluation."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption(
    "Heart Disease Prediction | Machine Learning Project"
)

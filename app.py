
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# -----------------------------
# Light Theme and Styling
# -----------------------------
st.markdown("""
<style>
.stApp {
    background-color: #F3F7FC;
    color: #243247;
}

header[data-testid="stHeader"] {
    background-color: #F3F7FC;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Main headings */
h1, h2, h3 {
    color: #153E75 !important;
}

/* General text */
.stApp p,
.stApp label,
.stApp [data-testid="stWidgetLabel"] p,
.stApp [data-testid="stMarkdownContainer"] p {
    color: #243247 !important;
}

/* Hero section */
.hero-title {
    color: #153E75;
    font-size: 38px;
    font-weight: 750;
    line-height: 1.2;
    margin-bottom: 18px;
}

.hero-description {
    color: #43566E;
    font-size: 16px;
    line-height: 1.8;
}

/* Form container */
div[data-testid="stForm"] {
    background-color: #FFFFFF !important;
    border: 1px solid #D6E2F0;
    border-radius: 16px;
    padding: 24px;
}

/* Input labels */
div[data-testid="stWidgetLabel"] p {
    color: #243247 !important;
    font-size: 15px !important;
    font-weight: 650 !important;
}

/* Dropdown fields */
div[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 8px !important;
}

div[data-baseweb="select"] *,
div[data-baseweb="select"] input {
    color: #243247 !important;
}

/* Number inputs */
div[data-testid="stNumberInput"] input {
    background-color: #FFFFFF !important;
    color: #243247 !important;
    border: 1px solid #CBD5E1 !important;
}

/* Slider labels and values */
div[data-testid="stSlider"] label,
div[data-testid="stSlider"] [data-testid="stMarkdownContainer"] {
    color: #243247 !important;
}

/* Buttons */
.stButton > button,
div[data-testid="stFormSubmitButton"] > button {
    background-color: #1769AA !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 9px !important;
    font-weight: 650 !important;
    min-height: 44px;
}

.stButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    background-color: #10528A !important;
    color: #FFFFFF !important;
}

/* Result panels */
div[data-testid="stAlert"] {
    border-radius: 10px;
}

/* Footer */
footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Load Saved ML Components
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
    st.error(f"Unable to load model files: {e}")
    st.stop()


# -----------------------------
# Hero Section
# -----------------------------
hero_text, hero_image = st.columns(
    [1.15, 1],
    vertical_alignment="center"
)

with hero_text:
    st.markdown("""
    <div style="padding: 20px 5px;">
        <div class="hero-title">
            ❤️ Heart Disease Prediction
        </div>
        <div class="hero-description">
            Your heart continuously pumps blood, supplying
            oxygen and essential nutrients throughout your body.
            <br><br>
            Maintaining heart health through healthy habits
            and timely medical checkups can help reduce
            cardiovascular risks.
        </div>
    </div>
    """, unsafe_allow_html=True)

with hero_image:
    image_path = Path(__file__).parent / "heart_banner.png"

    if image_path.exists():
        st.image(
            str(image_path),
            use_container_width=True
        )
    else:
        st.warning(
            "Heart image not found. Upload heart_banner.png "
            "to the same folder as app.py."
        )


st.divider()


# -----------------------------
# Patient Information
# -----------------------------
st.markdown("## Patient Health Information")

st.write(
    "Enter the patient details below to generate a "
    "machine-learning prediction."
)

with st.form("heart_prediction_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider(
            "Age",
            min_value=20,
            max_value=100,
            value=50
        )

        sex_choice = st.selectbox(
            "Sex",
            ["Female", "Male"]
        )

        cp = st.selectbox(
            "Chest Pain Type",
            [0, 1, 2, 3]
        )

        trestbps = st.number_input(
            "Resting Blood Pressure",
            min_value=80,
            max_value=200,
            value=120
        )

        chol = st.number_input(
            "Cholesterol",
            min_value=100,
            max_value=600,
            value=200
        )

    with col2:
        fbs = st.selectbox(
            "Fasting Blood Sugar >120 mg/dl",
            [0, 1]
        )

        restecg = st.selectbox(
            "Rest ECG",
            [0, 1, 2]
        )

        thalach = st.number_input(
            "Maximum Heart Rate",
            min_value=60,
            max_value=220,
            value=150
        )

        exang = st.selectbox(
            "Exercise-Induced Angina",
            [0, 1]
        )

    with col3:
        oldpeak = st.slider(
            "Oldpeak",
            min_value=0.0,
            max_value=6.0,
            value=1.0,
            step=0.1
        )

        slope = st.selectbox(
            "Slope",
            [0, 1, 2]
        )

        ca = st.selectbox(
            "Number of Major Vessels",
            [0, 1, 2, 3]
        )

        thal = st.selectbox(
            "Thal",
            [0, 1, 2, 3]
        )

    submitted = st.form_submit_button(
        "Predict Heart Disease Risk",
        use_container_width=True
    )


# -----------------------------
# Prediction
# -----------------------------
if submitted:

    sex = 1 if sex_choice == "Male" else 0

    columns = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal"
    ]

    input_data = pd.DataFrame(
        [[
            age,
            sex,
            cp,
            trestbps,
            chol,
            fbs,
            restecg,
            thalach,
            exang,
            oldpeak,
            slope,
            ca,
            thal
        ]],
        columns=columns
    )

    try:
        # Scale the input
        input_scaled = scaler.transform(input_data)

        # Apply feature selection
        input_selected = selector.transform(input_scaled)

        # Generate prediction
        prediction = model.predict(input_selected)

        st.divider()
        st.markdown("## Prediction Result")

        if prediction[0] == 1:
            st.error(
                "Potential Heart Disease Risk Detected. "
                "Please consult a qualified healthcare professional."
            )
        else:
            st.success(
                "The model did not identify a positive "
                "heart disease result for these inputs."
            )

        st.caption(
            "Educational use only. This model does not provide "
            "a medical diagnosis. Consult a qualified healthcare "
            "professional for medical evaluation."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")

        st.info(
            "Check whether the input features and their order "
            "match those used to train scaler.pkl and selector.pkl."
        )


# -----------------------------
# Footer
# -----------------------------
st.divider()

st.markdown(
    """
    <div style="
        text-align: center;
        color: #64748B;
        padding: 10px;
        font-size: 13px;
    ">
        Heart Disease Prediction | Machine Learning Project
    </div>
    """,
    unsafe_allow_html=True
)

import warnings
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

warnings.simplefilter(action="ignore", category=FutureWarning)

# =========================
# Page Setup & Base Config
# =========================
st.set_page_config(
    page_title="Medical Insurance Predictor",
    page_icon="💳",
    layout="centered",
)

# Explicit path to your Deployment folder
DEPLOY_FOLDER = Path(r"D:\Data science\Qasim Elasyed Gadelkareem Ali\Deployment")

# =========================
# Artifact Loaders
# =========================
@st.cache_resource
def load_ml_artifacts():
    model_path = DEPLOY_FOLDER / "insurance_rf_model.pkl"
    input_path = DEPLOY_FOLDER / "input.h5"
    metadata_path = DEPLOY_FOLDER / "model_metadata.joblib"

    # Fallback to local script folder if running from a different root
    if not model_path.exists():
        base_dir = Path(__file__).resolve().parent
        model_path = base_dir / "insurance_rf_model.pkl"
        input_path = base_dir / "input.h5"
        metadata_path = base_dir / "model_metadata.joblib"

    model = joblib.load(model_path) if model_path.exists() else None

    # Load input feature order either from input.h5 or metadata dict
    input_features = None
    if input_path.exists():
        input_features = joblib.load(input_path)
    elif metadata_path.exists():
        metadata = joblib.load(metadata_path)
        input_features = metadata.get("features", None)

    return model, input_features


model, input_features = load_ml_artifacts()

# =========================================================
# Machine Learning Prediction Tool
# =========================================================
st.title("💳 Predict Medical Insurance Charges")
st.markdown("Enter demographic information below to compute predicted charges using your Random Forest model.")

if model is None:
    st.error(f"⚠️ Could not locate `insurance_rf_model.pkl` inside `{DEPLOY_FOLDER}`.")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", min_value=18, max_value=100, value=30, step=1)
    bmi = st.slider("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    children = st.slider("Number of Children", min_value=0, max_value=10, value=0, step=1)

with col2:
    sex = st.selectbox("Sex", options=["female", "male"])
    smoker = st.selectbox("Smoker", options=["no", "yes"])
    region = st.selectbox("Region", options=["northeast", "northwest", "southeast", "southwest"])

if st.button("🚀 Predict Charges"):
    if model is None:
        st.error("Cannot predict: Model file is not loaded.")
    else:
        # Map user choices to binary encoding matching model training
        sex_binary = 1 if sex == "male" else 0
        smoker_binary = 1 if smoker == "yes" else 0

        input_data = {
            "age": age,
            "sex": sex_binary,
            "bmi": bmi,
            "children": children,
            "smoker": smoker_binary,
            "region_northwest": 1 if region == "northwest" else 0,
            "region_southeast": 1 if region == "southeast" else 0,
            "region_southwest": 1 if region == "southwest" else 0,
        }

        # Match exact column schema and ordering
        features_order = input_features if input_features is not None else list(input_data.keys())
        test_df = pd.DataFrame([input_data])[features_order]

        try:
            prediction = model.predict(test_df)[0]

            st.balloons()
            st.success(f"### Estimated Medical Insurance Charge: **${prediction:,.2f}**")
        except Exception as e:
            st.error(f"Error during model prediction: {e}")

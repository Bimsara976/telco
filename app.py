from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Telco Churn Risk Explorer",
    page_icon="📉",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "churn_pipeline.joblib"
METADATA_PATH = ROOT / "artifacts" / "model_metadata.json"


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Model artifacts are missing. Run the notebook through Section 18 and commit the artifacts folder."
        )
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, encoding="utf-8") as file:
        metadata = json.load(file)
    return model, metadata


try:
    model, metadata = load_artifacts()
except Exception as exc:
    st.error(f"Application startup failed: {exc}")
    st.stop()

st.title("Telecommunications Customer Churn Risk Explorer")
st.caption(
    "COM763 Portfolio Task 1 demonstration — estimates churn probability for one customer profile."
)

with st.sidebar:
    st.header("Model information")
    st.write(f"**Estimator:** {metadata['model_type']}")
    st.write(f"**Decision threshold:** {metadata['decision_threshold']:.3f}")
    metrics = metadata.get("test_metrics_tuned_threshold", {})
    if metrics:
        st.metric("Held-out ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
        st.metric("Held-out PR-AUC", f"{metrics.get('pr_auc', 0):.3f}")
        st.metric("Held-out recall", f"{metrics.get('recall', 0):.3f}")
    st.divider()
    st.info(
        "This educational system supports human review. It should not be used for pricing, denial of service, or fully automated decisions."
    )

st.subheader("Customer profile")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_label = st.selectbox("Senior citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)
        phone_service = st.selectbox("Phone service", ["Yes", "No"])
        multiple_lines = st.selectbox(
            "Multiple lines", ["No", "Yes", "No phone service"]
        )

    with col2:
        internet_service = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
        service_options = ["No", "Yes", "No internet service"]
        online_security = st.selectbox("Online security", service_options)
        online_backup = st.selectbox("Online backup", service_options)
        device_protection = st.selectbox("Device protection", service_options)
        tech_support = st.selectbox("Tech support", service_options)
        streaming_tv = st.selectbox("Streaming TV", service_options)
        streaming_movies = st.selectbox("Streaming movies", service_options)

    with col3:
        contract = st.selectbox(
            "Contract", ["Month-to-month", "One year", "Two year"]
        )
        paperless_billing = st.selectbox("Paperless billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        monthly_charges = st.number_input(
            "Monthly charges", min_value=0.0, max_value=200.0, value=70.0, step=0.05
        )
        total_charges = st.number_input(
            "Total charges", min_value=0.0, max_value=20000.0,
            value=float(round(tenure * monthly_charges, 2)), step=0.05
        )

    submitted = st.form_submit_button("Estimate churn risk", width='stretch')

if submitted:
    row = {
        "gender": gender,
        "SeniorCitizen": 1 if senior_label == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    input_df = pd.DataFrame([row])
    input_df = input_df[metadata["feature_order"]]

    probability = float(model.predict_proba(input_df)[:, 1][0])
    threshold = float(metadata["decision_threshold"])
    predicted_churn = probability >= threshold

    result_col, gauge_col = st.columns([1, 2])
    with result_col:
        st.metric("Estimated churn probability", f"{probability:.1%}")
        st.metric("Decision threshold", f"{threshold:.1%}")

    with gauge_col:
        st.progress(min(max(probability, 0.0), 1.0), text=f"Risk score: {probability:.1%}")
        if predicted_churn:
            st.warning(
                "The profile is above the selected screening threshold. A retention specialist should review the account and contact history."
            )
        else:
            st.success(
                "The profile is below the selected screening threshold. Continue normal monitoring rather than treating this as a guarantee of retention."
            )

    with st.expander("Prediction input record"):
        st.dataframe(input_df.T.rename(columns={0: "Value"}).astype(str), width='stretch')

st.divider()
st.caption(
    "Dataset: IBM Telco Customer Churn sample. Model outputs are probabilistic estimates and may not generalise to another provider or time period."
)
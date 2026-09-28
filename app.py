
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# Load the trained model only once
@st.cache_resource
def load_model():
    return joblib.load("churn_model.pkl")

model = load_model()
THRESHOLD = 0.30

# Reset service selections when internet changes
internet_keys = [
    "online_security", "online_backup", "device_protection",
    "tech_support", "streaming_tv", "streaming_movies"
]

def reset_internet_services():
    if st.session_state["internet_service"] == "No":
        for key in internet_keys:
            st.session_state[key] = "No internet service"
    else:
        for key in internet_keys:
            if st.session_state.get(key) == "No internet service":
                st.session_state[key] = "No"

def reset_multiple_lines():
    if st.session_state["phone_service"] == "No":
        st.session_state["multiple_lines"] = "No phone service"
    elif st.session_state.get("multiple_lines") == "No phone service":
        st.session_state["multiple_lines"] = "No"

# Display-friendly names for SHAP
labels = {
    "gender": "Gender",
    "SeniorCitizen": "Senior citizen",
    "Partner": "Partner",
    "Dependents": "Dependents",
    "PhoneService": "Phone service",
    "MultipleLines": "Multiple lines",
    "InternetService": "Internet service",
    "OnlineSecurity": "Online security",
    "OnlineBackup": "Online backup",
    "DeviceProtection": "Device protection",
    "TechSupport": "Technical support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming movies",
    "Contract": "Contract",
    "PaperlessBilling": "Paperless billing",
    "PaymentMethod": "Payment method",
    "tenure": "Tenure (months)",
    "MonthlyCharges": "Monthly charges",
    "TotalCharges": "Total charges"
}

def readable_name(name):
    if name.startswith("num__"):
        raw = name.removeprefix("num__")
        return labels.get(raw, raw)

    raw = name.removeprefix("cat__")
    for column, label in labels.items():
        prefix = column + "_"
        if raw.startswith(prefix):
            category = raw[len(prefix):]
            return f"{label}: {category}"

    return raw


# -------------------------
# Page heading
# -------------------------
st.title("📊 Customer Churn Predictor")
st.caption(
    "Estimate customer churn risk using a trained machine learning model."
)
st.info(
    "This is a demonstration based on historical telecom data, "
    "not a guarantee of future customer behaviour."
)

# -------------------------
# Customer profile
# -------------------------
st.subheader("Customer profile")

c1, c2, c3 = st.columns(3)

with c1:
    gender = st.selectbox("Gender", ["Female", "Male"], key="gender")

    senior_citizen = st.selectbox(
        "Senior citizen",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes",
        key="senior_citizen"
    )

with c2:
    partner = st.selectbox("Partner", ["No", "Yes"], key="partner")
    dependents = st.selectbox(
        "Dependents", ["No", "Yes"], key="dependents"
    )

with c3:
    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=72,
        value=12,
        step=1,
        key="tenure"
    )

# -------------------------
# Services
# -------------------------
st.divider()
st.subheader("Services")

left, right = st.columns(2)

with left:
    st.markdown("**Phone and internet**")

    phone_service = st.selectbox(
        "Phone service",
        ["Yes", "No"],
        key="phone_service",
        on_change=reset_multiple_lines
    )

    multiple_options = (
        ["No phone service"]
        if phone_service == "No"
        else ["No", "Yes"]
    )

    multiple_lines = st.selectbox(
        "Multiple lines",
        multiple_options,
        key="multiple_lines"
    )

    internet_service = st.selectbox(
        "Internet service",
        ["DSL", "Fiber optic", "No"],
        key="internet_service",
        on_change=reset_internet_services
    )

    service_options = (
        ["No internet service"]
        if internet_service == "No"
        else ["No", "Yes"]
    )

    online_security = st.selectbox(
        "Online security",
        service_options,
        key="online_security"
    )

    online_backup = st.selectbox(
        "Online backup",
        service_options,
        key="online_backup"
    )

    device_protection = st.selectbox(
        "Device protection",
        service_options,
        key="device_protection"
    )

with right:
    st.markdown("**Other services and contract**")

    tech_support = st.selectbox(
        "Technical support",
        service_options,
        key="tech_support"
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        service_options,
        key="streaming_tv"
    )

    streaming_movies = st.selectbox(
        "Streaming movies",
        service_options,
        key="streaming_movies"
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"],
        key="contract"
    )

    paperless_billing = st.selectbox(
        "Paperless billing",
        ["Yes", "No"],
        key="paperless_billing"
    )

    payment_method = st.selectbox(
        "Payment method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ],
        key="payment_method"
    )

# -------------------------
# Charges
# -------------------------
st.divider()
st.subheader("Charges")

charge1, charge2 = st.columns(2)

with charge1:
    monthly_charges = st.number_input(
        "Monthly charges",
        min_value=0.0,
        max_value=120.0,
        value=70.0,
        step=0.5
    )

with charge2:
    total_charges = st.number_input(
        "Total charges",
        min_value=0.0,
        value=800.0,
        step=10.0
    )

# -------------------------
# Prediction
# -------------------------
st.divider()

if st.button("Estimate churn risk", type="primary"):

    if tenure == 0 and total_charges > 0:
        st.error(
            "For this dataset, zero-tenure customers have "
            "zero total charges. Please check your inputs."
        )

    else:
        customer = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior_citizen,
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
            "TotalCharges": total_charges
        }])

        probability = float(model.predict_proba(customer)[0, 1])
        prediction = probability >= THRESHOLD

        st.subheader("Prediction")

        metric_col, status_col = st.columns(2)

        with metric_col:
            st.metric(
                "Estimated churn probability",
                f"{probability:.1%}"
            )
            st.progress(probability)

        with status_col:
            if prediction:
                st.error("Higher predicted churn risk")
            else:
                st.success("Lower predicted churn risk")

            st.caption(
                f"Classification threshold: {THRESHOLD:.0%}"
            )

        st.caption(
            "The risk label is determined by the experimental "
            "30% threshold. It is not a guaranteed outcome."
        )

        # -------------------------
        # Individual SHAP explanation
        # -------------------------
        with st.expander(
            "Why did the model make this prediction?",
            expanded=True
        ):
            preprocessor = model.named_steps["preprocessor"]
            classifier = model.named_steps["model"]

            transformed = preprocessor.transform(customer)

            explainer = shap.TreeExplainer(classifier)
            explanation = explainer(transformed)

            values = np.asarray(explanation.values)

            if values.ndim == 3:
                contributions = values[0, :, 1]
            else:
                contributions = values[0]

            feature_names = preprocessor.get_feature_names_out()

            explanation_df = pd.DataFrame({
                "Feature": [
                    readable_name(name) for name in feature_names
                ],
                "Contribution": contributions
            })

            explanation_df["Magnitude"] = (
                explanation_df["Contribution"].abs()
            )

            top_features = (
                explanation_df.nlargest(10, "Magnitude")
                .sort_values("Contribution")
            )

            st.caption(
                "Positive contributions push the model score "
                "toward churn; negative contributions push it away. "
                "They are not probability percentage points."
            )

            st.bar_chart(
                top_features.set_index("Feature")["Contribution"],
                horizontal=True
            )

            st.markdown("**Largest contributions toward churn**")
            toward = explanation_df.nlargest(3, "Contribution")

            for _, row in toward.iterrows():
                if row["Contribution"] > 0:
                    st.write(
                        f"• {row['Feature']}: "
                        f"+{row['Contribution']:.3f}"
                    )

            st.markdown("**Largest contributions away from churn**")
            away = explanation_df.nsmallest(3, "Contribution")

            for _, row in away.iterrows():
                if row["Contribution"] < 0:
                    st.write(
                        f"• {row['Feature']}: "
                        f"{row['Contribution']:.3f}"
                    )
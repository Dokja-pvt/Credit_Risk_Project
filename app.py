import os
import re
import pickle
import pandas as pd
import numpy as np
import streamlit as st

# Configure professional dashboard architecture
st.set_page_config(
    page_title="Enterprise Credit Risk Core",
    page_icon="🛡️",
    layout="wide"
)

MODEL_PATH = "data/processed/model.pkl"
FEATURES_PATH = "data/processed/trained_features.pkl"

@st.cache_resource
def load_production_artifacts():
    """Safely load and cache serialized model weights and feature schemas."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
        return None, None
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(FEATURES_PATH, 'rb') as f:
        features = pickle.load(f)
    return model, features

def calculate_credit_score(p, base_score=600, base_odds=50, pdo=20):
    """Applies log-odds calibration to convert default probability into a 300-850 score."""
    p = np.clip(p, 0.0001, 0.9999)
    odds = (1 - p) / p
    factor = pdo / np.log(2)
    offset = base_score - (factor * np.log(base_odds))
    score = offset + (factor * np.log(odds))
    return np.clip(score, 300, 850).astype(int)

# Initialize application header
st.title("🛡️ Enterprise Credit Risk Decisioning System")
st.markdown("---")

model, trained_features = load_production_artifacts()

if model is None or trained_features is None:
    st.error("❌ Production pipeline artifacts missing. Please execute Phase 3 and Phase 4 before serving traffic.")
    st.stop()

# Segment the interface into multi-column layout blocks
st.subheader("📋 Applicant Telemetry Input Profile")
col1, col2, col3 = st.columns(3)

with col1:
    amt_income = st.number_input("Total Annual Income (Currency)", min_value=10000, max_value=5000000, value=150000, step=5000)
    amt_credit = st.number_input("Requested Loan Amount (Currency)", min_value=10000, max_value=10000000, value=500000, step=10000)
    amt_annuity = st.number_input("Expected Annual Loan Repayment (Annuity)", min_value=1000, max_value=500000, value=25000, step=1000)

with col2:
    age_years = st.slider("Applicant Age (Years)", min_value=18, max_value=90, value=35)
    employment_years = st.slider("Verified Employment Length (Years)", min_value=0, max_value=60, value=5)
    income_type = st.selectbox("Income Classification Type", ["Working", "Commercial associate", "State servant", "Pensioner"])

with col3:
    contract_type = st.selectbox("Loan Contract Architecture", ["Cash loans", "Revolving loans"])
    education_type = st.selectbox("Highest Educational Level Attained", ["Secondary / special education", "Higher education", "Incomplete higher", "Lower secondary"])
    # Relational historical data inputs derived from Phase 2 aggregation
    hist_credit_mean = st.number_input("Historical Portfolio Mean Credit (Phase 2)", min_value=0, max_value=5000000, value=300000)

# Execution Trigger
if st.button("Run Credit Decision Optimization Pipeline", type="primary"):
    
    # 1. Map interactive values back to foundational raw training features
    raw_input_dict = {
        'AMT_INCOME_TOTAL': amt_income,
        'AMT_CREDIT': amt_credit,
        'AMT_ANNUITY': amt_annuity,
        'DAYS_BIRTH': -int(age_years * 365.25),
        'DAYS_EMPLOYED': -int(employment_years * 365.25),
        'NAME_INCOME_TYPE': income_type,
        'NAME_CONTRACT_TYPE': contract_type,
        'NAME_EDUCATION_TYPE': education_type,
        'AMT_CREDIT_MEAN': hist_credit_mean
    }
    
    # 2. Build scoring DataFrame matching training alignment
    input_df = pd.DataFrame([raw_input_dict])
    input_df = pd.get_dummies(input_df, drop_first=True)
    
    # Sanitize feature name syntax
    input_df.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', str(col)) for col in input_df.columns]
    
    # Create production array structure filled with 0s for missing sparse columns
    scoring_matrix = pd.DataFrame(0, index=[0], columns=trained_features)
    
    # Overlay overlap columns to preserve feature tracking state
    for col in input_df.columns:
        if col in scoring_matrix.columns:
            scoring_matrix[col] = input_df[col].values

    # 3. Compute continuous model inference probability
    prob_default = model.predict_proba(scoring_matrix)[0, 1]
    
    # 4. Process calibration scorecard conversions
    credit_score = calculate_credit_score(prob_default)

    # 5. Resolve Portfolio Risk Tiers
    if credit_score >= 720:
        tier, color, status = "Tier 1 (Prime)", "#2ecc71", "SUCCESS: High Probability of Repayment. Auto-Approve authorized."
    elif credit_score >= 640:
        tier, color, status = "Tier 2 (Good)", "#3498db", "NOTICE: Standard risk metrics. Pass to regular underwriting loops."
    elif credit_score >= 540:
        tier, color, status = "Tier 3 (Subprime)", "#f39c12", "WARNING: Elevated default pattern detected. Requires structural collateral."
    else:
        tier, color, status = "Tier 4 (High Risk)", "#e74c3c", "CRITICAL: Underwriting failure. High expected loss. Auto-Decline enforced."

    # 6. Render Quantitative Outputs to User Interface
    st.markdown("---")
    st.subheader("📊 Operational Risk Underwriting Report")
    
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric(label="Calibrated Credit Score", value=f"{credit_score} Pts")
    with metric_col2:
        st.metric(label="Inference Default Probability", value=f"{prob_default * 100:.2f}%")
    with metric_col3:
        st.markdown(f"**Assigned Allocation Status Bracket:** <span style='color:{color}; font-size:20px; font-weight:bold;'>{tier}</span>", unsafe_allow_html=True)
        
    st.info(f"**Underwriting Action Directive:** {status}")
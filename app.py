# ==========================================
# 💻 STREAMLIT APP: Marketing Response Predictor
# ==========================================

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier

# ✅ Load trained model
@st.cache_resource
def load_model():
    model = joblib.load("xgb_marketing_model_v3.pkl")
    return model

model = load_model()

# ✅ Page configuration
st.set_page_config(page_title="Marketing Campaign Response Predictor", layout="centered")

st.title("🎯 Marketing Campaign Response Predictor")
st.markdown("Predict whether a customer is likely to **respond to your marketing campaign** based on product, pricing, and advertisement parameters.")

# ==========================================
# 🔹 User Inputs
# ==========================================
col1, col2 = st.columns(2)

with col1:
    ad_intensity_input = st.selectbox("Ad Intensity", ["Low", "Medium", "High"])
    product_type_input = st.selectbox("Product Type", ["Necessity", "Luxury"])
    discount = st.slider("Discount Offered (%)", 0, 50, 10)
    credit_score = st.slider("Credit Score", 300, 850, 600)
    age = st.slider("Age", 18, 65, 30)

with col2:
    price_min = st.number_input("Min Product Price (₹)", 500, 50000, 1000, step=500)
    price_max = st.number_input("Max Product Price (₹)", 500, 50000, 10000, step=500)
    income_min = st.number_input("Min Annual Income (₹)", 20000, 150000, 50000, step=5000)
    income_max = st.number_input("Max Annual Income (₹)", 20000, 150000, 120000, step=5000)

# ==========================================
# 🔹 Derived Features
# ==========================================
# Encode product type
product_type = 0 if product_type_input == "Necessity" else 1

# Convert ad intensity to numeric
ad_intensity_map = {"Low": 25, "Medium": 75, "High": 150}
ad_intensity = ad_intensity_map[ad_intensity_input]

# Derived variables
price_avg = (price_min + price_max) / 2
income_avg = (income_min + income_max) / 2
affordability_ratio = round(price_avg / income_avg, 2)
normalized_ad_intensity = ad_intensity / 150

discount_to_afford = discount * (1 - affordability_ratio)
credit_afford_interaction = credit_score * (1 - affordability_ratio)
income_to_credit = income_avg / credit_score
discount_intensity = discount * ad_intensity
price_to_income = price_avg / income_avg

# Feature vector (same order as training)
features = np.array([[age, income_avg, credit_score, product_type, price_avg,
                      discount, affordability_ratio, ad_intensity,
                      normalized_ad_intensity, np.log1p(price_avg),
                      np.log1p(income_avg), discount_to_afford,
                      credit_afford_interaction, income_to_credit,
                      discount_intensity, price_to_income]])

# ==========================================
# 🔹 Prediction
# ==========================================
if st.button("🔍 Predict Response"):
    prob = model.predict_proba(features)[0][1]
    result = "✅ Likely to Respond" if prob >= 0.5 else "❌ Unlikely to Respond"

    st.subheader(result)
    st.progress(int(prob * 100))
    st.metric(label="Predicted Response Probability", value=f"{prob*100:.2f}%")

    st.markdown("---")
    st.caption("Model: XGBoost v3 | Dataset: Improved Balanced Synthetic | Accuracy ≈ 0.64 | ROC-AUC ≈ 0.70")

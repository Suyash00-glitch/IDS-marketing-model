# ==========================================
# 💼 MARKETING CAMPAIGN RESPONSE PREDICTOR
# ==========================================
# Inputs:
#   1. Ad Intensity (Low / Medium / High)
#   2. Product Type (Luxury / Necessity)
#   3. Annual Income (ranges)
#   4. Product Price (ranges)
# Output:
#   - Response Probability
# ==========================================

import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ---------- Page Setup ----------
st.set_page_config(page_title="Marketing Response Predictor", layout="centered")

st.markdown(
    """
    <div style="text-align:center;">
        <h1 style="color:#0077b6; font-size:38px;">📊 Marketing Campaign Response Predictor</h1>
        <p style="font-size:18px; color:#555;">
            Estimate the probability of a customer responding to a campaign using key business insights.
        </p>
    </div>
    <hr style="border:1px solid #eee;">
    """,
    unsafe_allow_html=True
)

# ---------- Load Model ----------
@st.cache_resource
def load_model():
    return joblib.load("xgb_marketing_model_final_colab.pkl")

model = load_model()

# ---------- Dropdown Definitions ----------

ad_intensity_options = ["Low", "Medium", "High"]
product_type_options = ["Necessity", "Luxury"]

annual_income_ranges = {
    "₹1,00,000 - ₹3,00,000": (100000, 300000),
    "₹3,00,001 - ₹5,00,000": (300001, 500000),
    "₹5,00,001 - ₹10,00,000": (500001, 1000000),
    "₹10,00,001 - ₹25,00,000": (1000001, 2500000),
    "₹25,00,001 - ₹50,00,000": (2500001, 5000000)
}

product_price_ranges = {
    "₹10,000 - ₹20,000": (10000, 20000),
    "₹20,001 - ₹50,000": (20001, 50000),
    "₹50,001 - ₹1,00,000": (50001, 100000),
    "₹1,00,001 - ₹5,00,000": (100001, 500000),
    "₹5,00,001 - ₹10,00,000": (500001, 1000000)
}

# ---------- Input Section ----------
st.subheader("Enter Campaign Parameters")

col1, col2 = st.columns(2)

with col1:
    ad_intensity = st.selectbox("📢 Ad Intensity", ad_intensity_options)
    product_type = st.selectbox("🛍️ Product Type", product_type_options)

with col2:
    income_label = st.selectbox("💰 Annual Income Range", list(annual_income_ranges.keys()))
    price_label = st.selectbox("🏷️ Product Price Range", list(product_price_ranges.keys()))

# ---------- Compute Derived Inputs ----------
def midpoint(bounds):
    return (bounds[0] + bounds[1]) / 2

annual_income = midpoint(annual_income_ranges[income_label])
product_price = midpoint(product_price_ranges[price_label])

ad_intensity_map = {"Low": 0, "Medium": 1, "High": 2}
product_type_map = {"Necessity": 0, "Luxury": 1}

ad_intensity_num = ad_intensity_map[ad_intensity]
product_type_num = product_type_map[product_type]

# Derived engineered features
affordability_ratio = product_price / annual_income
log_price = np.log1p(product_price)
log_income = np.log1p(annual_income)
discount_offered = 15  # fixed example, business can adjust
credit_score = 700     # assume avg user for simplicity
ad_intensity_value = {0: 100, 1: 200, 2: 350}[ad_intensity_num]  # scaled numeric proxy

# Derived engineered columns
normalized_ad_intensity = ad_intensity_value / 400
discount_to_afford = discount_offered * (1 - affordability_ratio)
credit_afford_interaction = credit_score * (1 - affordability_ratio)
income_to_credit = annual_income / credit_score
discount_intensity = discount_offered * ad_intensity_value
price_to_income = product_price / annual_income

# Create dataframe for prediction
X_input = pd.DataFrame([{
    "Age": 30,
    "Annual_Income": annual_income,
    "Credit_Score": credit_score,
    "Product_Type_enc": product_type_num,
    "Product_Price": product_price,
    "Discount_Offered(%)": discount_offered,
    "Affordability_Ratio": affordability_ratio,
    "Ad_Calls": 10,
    "Ad_SMS": 100,
    "Ad_Social": 50,
    "Ad_Display": 10,
    "Ad_Intensity_Num": ad_intensity_num,
    "Normalized_Ad_Intensity": normalized_ad_intensity,
    "Log_Price": log_price,
    "Log_Income": log_income,
    "Discount_to_Afford": discount_to_afford,
    "Credit_Afford_Interaction": credit_afford_interaction,
    "Income_to_Credit": income_to_credit,
    "Discount_Intensity": discount_intensity,
    "Price_to_Income": price_to_income
}])

# ---------- Prediction ----------
if st.button("🎯 Predict Response Probability"):
    probability = model.predict_proba(X_input)[:, 1][0]
    st.markdown(
        f"""
        <div style="text-align:center; margin-top:30px;">
            <h2 style="color:#023047;">Predicted Campaign Response Probability:</h2>
            <h1 style="color:#06d6a0; font-size:50px;">{probability*100:.2f}%</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------- Footer ----------
st.markdown(
    """
    <hr style="border:1px solid #eee;">
    <p style="text-align:center; color:gray; font-size:14px;">
    Built for marketing analytics by <b>Infrst</b> | XGBoost Model (v5)
    </p>
    """,
    unsafe_allow_html=True
)

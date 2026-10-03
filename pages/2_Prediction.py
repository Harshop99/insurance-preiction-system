from __future__ import annotations

import pandas as pd
import streamlit as st

from model_utils import append_history, load_app_data, prediction_record, train_model

st.set_page_config(page_title="Insurance Prediction", page_icon="🔮", layout="wide")
st.title("🔮 Make a prediction")
st.caption("Enter customer details to estimate annual insurance charges.")

data = load_app_data()
model, _ = train_model(data)

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
        bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=27.5, step=0.1)
    with col2:
        sex = st.selectbox("Sex", sorted(data["sex"].dropna().unique()))
        children = st.number_input("Children", min_value=0, max_value=10, value=0, step=1)
    with col3:
        smoker = st.selectbox("Smoker", sorted(data["smoker"].dropna().unique()))
        region = st.selectbox("Region", sorted(data["region"].dropna().unique()))
    submitted = st.form_submit_button("Predict insurance charge", type="primary", use_container_width=True)

if submitted:
    input_data = pd.DataFrame(
        [{"age": age, "sex": sex, "bmi": bmi, "children": children, "smoker": smoker, "region": region}]
    )
    predicted_charge = float(model.predict(input_data)[0])
    append_history(prediction_record(input_data, predicted_charge))
    st.session_state["latest_prediction"] = predicted_charge
    st.success(f"Estimated annual charge: ${predicted_charge:,.2f}")
    st.caption("Your prediction has been saved on the History page.")

if "latest_prediction" in st.session_state:
    st.metric("Latest estimate", f"${st.session_state['latest_prediction']:,.2f}")

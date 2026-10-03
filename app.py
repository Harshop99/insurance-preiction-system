from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from model_utils import load_app_data, train_model

st.set_page_config(page_title="Insurance Analysis", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 3rem; }
    [data-testid="stMetricValue"] { color: #0f766e; }
    .hero { padding: 1.5rem 1.75rem; border-radius: 18px; background: linear-gradient(120deg, #ecfeff, #f0fdf4); border: 1px solid #ccfbf1; margin-bottom: 1.25rem; }
    .hero h1 { margin: 0; color: #134e4a; }
    .hero p { margin: .45rem 0 0; color: #475569; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("💙 HealthCover AI")
st.sidebar.caption("Insurance charge prediction workspace")

data = load_app_data()
_, metrics = train_model(data)

st.markdown(
    '<div class="hero"><h1>Insurance Analysis</h1><p>Explore the dataset and review model quality.</p></div>',
    unsafe_allow_html=True,
)

metric_cols = st.columns(4)
metric_cols[0].metric("Records", f"{len(data):,}")
metric_cols[1].metric("R² score", f"{metrics['r2']:.3f}")
metric_cols[2].metric("Mean absolute error", f"${metrics['mae']:,.0f}")
metric_cols[3].metric("Average charge", f"${data['charges'].mean():,.0f}")

left, right = st.columns(2)
with left:
    st.subheader("Charge distribution")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(data["charges"], kde=True, color="#0f766e", ax=ax)
    ax.set_xlabel("Annual charges ($)")
    ax.set_ylabel("People")
    st.pyplot(fig, clear_figure=True)
with right:
    st.subheader("Charges by smoking status")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=data, x="smoker", y="charges", hue="smoker", palette="Set2", legend=False, ax=ax)
    ax.set_xlabel("Smoker")
    ax.set_ylabel("Annual charges ($)")
    st.pyplot(fig, clear_figure=True)

st.subheader("Dataset preview")
st.dataframe(data.head(10), use_container_width=True, hide_index=True)
with st.expander("About the model"):
    st.write("A linear regression model uses standardized numeric features and one-hot encoded categorical features. The holdout metrics above are calculated on 20% of the data.")
    st.write(f"Root mean squared error: **${metrics['rmse']:,.0f}**")

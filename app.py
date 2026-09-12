from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


APP_DIR = Path(__file__).parent
HISTORY_FILE = APP_DIR / "prediction_history.csv"
REQUIRED_COLUMNS = ["age", "sex", "bmi", "children", "smoker", "region", "charges"]
NUMERIC_FEATURES = ["age", "bmi", "children"]
CATEGORICAL_FEATURES = ["sex", "smoker", "region"]

st.set_page_config(
    page_title="HealthCover AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

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


def locate_dataset() -> Path | None:
    candidates = [APP_DIR / "insurance.csv", APP_DIR.parent / "insurance.csv", Path.home() / "Desktop" / "insurance.csv"]
    return next((path for path in candidates if path.exists()), None)


@st.cache_data(show_spinner=False)
def read_dataset(source: str | bytes) -> pd.DataFrame:
    csv_source = io.BytesIO(source) if isinstance(source, bytes) else source
    data = pd.read_csv(csv_source)
    missing = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    data = data[REQUIRED_COLUMNS].copy()
    data[NUMERIC_FEATURES + ["charges"]] = data[NUMERIC_FEATURES + ["charges"]].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=REQUIRED_COLUMNS)
    return data


@st.cache_resource(show_spinner="Training the insurance model…")
def train_model(data: pd.DataFrame) -> tuple[Pipeline, dict[str, float]]:
    features = data.drop(columns="charges")
    target = data["charges"]
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    model = Pipeline([("preprocessor", preprocessor), ("regressor", LinearRegression())])
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = {
        "r2": r2_score(y_test, predictions),
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": mean_squared_error(y_test, predictions) ** 0.5,
    }
    return model, metrics


def append_history(record: dict[str, object]) -> None:
    history = pd.DataFrame([record])
    history.to_csv(HISTORY_FILE, mode="a", header=not HISTORY_FILE.exists(), index=False)


st.sidebar.title("💙 HealthCover AI")
st.sidebar.caption("Insurance charge prediction workspace")

uploaded_file = st.sidebar.file_uploader("Upload insurance.csv", type="csv")
dataset_path = locate_dataset()
source = uploaded_file if uploaded_file is not None else dataset_path

if source is None:
    st.markdown('<div class="hero"><h1>HealthCover AI</h1><p>Estimate medical insurance charges with a transparent machine-learning model.</p></div>', unsafe_allow_html=True)
    st.info("Upload an insurance CSV to get started. The file must contain: age, sex, bmi, children, smoker, region, and charges.")
    st.stop()

try:
    data = read_dataset(uploaded_file.getvalue() if uploaded_file is not None else str(source))
except ValueError as error:
    st.error(str(error))
    st.stop()

model, metrics = train_model(data)

st.markdown('<div class="hero"><h1>Insurance Prediction & Analysis</h1><p>Explore the dataset, review model quality, and estimate an individual\'s annual insurance charge.</p></div>', unsafe_allow_html=True)

metric_cols = st.columns(4)
metric_cols[0].metric("Records", f"{len(data):,}")
metric_cols[1].metric("R² score", f"{metrics['r2']:.3f}")
metric_cols[2].metric("Mean absolute error", f"${metrics['mae']:,.0f}")
metric_cols[3].metric("Average charge", f"${data['charges'].mean():,.0f}")

analysis_tab, prediction_tab = st.tabs(["📊 Analysis", "🔮 Make a prediction"])

with analysis_tab:
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

with prediction_tab:
    st.subheader("Enter customer details")
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
        input_data = pd.DataFrame([{"age": age, "sex": sex, "bmi": bmi, "children": children, "smoker": smoker, "region": region}])
        predicted_charge = float(model.predict(input_data)[0])
        record = {**input_data.iloc[0].to_dict(), "predicted_charge": round(predicted_charge, 2), "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        append_history(record)
        st.session_state["latest_prediction"] = predicted_charge
        st.success(f"Estimated annual charge: ${predicted_charge:,.2f}")
        st.caption("Your prediction has been saved on the History page.")

    if "latest_prediction" in st.session_state:
        st.metric("Latest estimate", f"${st.session_state['latest_prediction']:,.2f}")

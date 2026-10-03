from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

APP_DIR = Path(__file__).parent
DATASET_FILE = APP_DIR / "insurance.csv"
HISTORY_FILE = APP_DIR / "prediction_history.csv"
REQUIRED_COLUMNS = ["age", "sex", "bmi", "children", "smoker", "region", "charges"]
NUMERIC_FEATURES = ["age", "bmi", "children"]
CATEGORICAL_FEATURES = ["sex", "smoker", "region"]


@st.cache_data(show_spinner=False)
def read_dataset(source: str) -> pd.DataFrame:
    data = pd.read_csv(source)
    missing = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    data = data[REQUIRED_COLUMNS].copy()
    data[NUMERIC_FEATURES + ["charges"]] = data[NUMERIC_FEATURES + ["charges"]].apply(pd.to_numeric, errors="coerce")
    return data.dropna(subset=REQUIRED_COLUMNS)


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
    pd.DataFrame([record]).to_csv(HISTORY_FILE, mode="a", header=not HISTORY_FILE.exists(), index=False)


def load_app_data() -> pd.DataFrame:
    if not DATASET_FILE.exists():
        st.error(f"Dataset not found: {DATASET_FILE.name}")
        st.stop()
    try:
        return read_dataset(str(DATASET_FILE))
    except (OSError, ValueError) as error:
        st.error(str(error))
        st.stop()


def prediction_record(input_data: pd.DataFrame, predicted_charge: float) -> dict[str, object]:
    return {
        **input_data.iloc[0].to_dict(),
        "predicted_charge": round(predicted_charge, 2),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

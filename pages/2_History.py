from pathlib import Path

import pandas as pd
import streamlit as st

HISTORY_FILE = Path(__file__).parent.parent / "prediction_history.csv"

st.set_page_config(page_title="Prediction History", page_icon="🧾", layout="wide")

st.title("🧾 Prediction history")
st.caption("Every estimate created on the prediction page is stored locally in this workspace.")

if not HISTORY_FILE.exists():
    st.info("No predictions yet. Go to **Insurance Prediction & Analysis** and make your first estimate.")
    st.stop()

history = pd.read_csv(HISTORY_FILE)
if history.empty:
    st.info("No predictions yet.")
    st.stop()

history["predicted_charge"] = pd.to_numeric(history["predicted_charge"], errors="coerce")

summary = st.columns(3)
summary[0].metric("Total predictions", f"{len(history):,}")
summary[1].metric("Average estimate", f"${history['predicted_charge'].mean():,.2f}")
summary[2].metric("Highest estimate", f"${history['predicted_charge'].max():,.2f}")

st.subheader("Saved estimates")
display_columns = ["created_at", "age", "sex", "bmi", "children", "smoker", "region", "predicted_charge"]
display_history = history[[column for column in display_columns if column in history.columns]].copy()
display_history = display_history.sort_values("created_at", ascending=False)
st.dataframe(
    display_history.style.format({"predicted_charge": "${:,.2f}", "bmi": "{:.1f}"}),
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    "Download history as CSV",
    data=history.to_csv(index=False).encode("utf-8"),
    file_name="prediction_history.csv",
    mime="text/csv",
)

with st.expander("Manage history"):
    st.warning("This permanently deletes the local prediction history file.")
    if st.button("Clear all history", type="secondary"):
        HISTORY_FILE.unlink(missing_ok=True)
        st.success("Prediction history cleared.")
        st.rerun()

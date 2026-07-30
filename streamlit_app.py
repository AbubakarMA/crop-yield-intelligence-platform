"""Streamlit interface for a locally running Crop Yield Intelligence API."""

from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Crop Yield Intelligence",
    page_icon="🌾",
    layout="wide",
)
st.title("Crop Yield Intelligence Platform")
st.caption(
    "Estimate historical crop yield and inspect the model's intended scope."
)

try:
    metadata = httpx.get(f"{API_URL}/metadata", timeout=10).json()
except (httpx.HTTPError, ValueError):
    st.error(
        "The prediction API is not ready. Start the API and confirm the model "
        "artifacts exist."
    )
    st.stop()

left, right = st.columns(2)
with left:
    area = st.selectbox("Country", metadata["countries"])
    item = st.selectbox("Crop", metadata["crops"])
    year = st.slider(
        "Historical year",
        metadata["supported_years"][0],
        metadata["supported_years"][1],
        metadata["supported_years"][1],
    )
with right:
    rainfall = st.number_input(
        "Annual rainfall (mm)",
        min_value=1.0,
        value=1200.0,
    )
    pesticides = st.number_input(
        "National pesticide use (tonnes)",
        min_value=0.0,
        value=100.0,
    )
    temperature = st.number_input(
        "Average temperature (°C)",
        min_value=-30.0,
        max_value=50.0,
        value=25.0,
    )

if st.button("Estimate yield", type="primary"):
    response = httpx.post(
        f"{API_URL}/predict",
        json={
            "area": area,
            "item": item,
            "year": year,
            "average_rainfall_mm_per_year": rainfall,
            "pesticides_tonnes": pesticides,
            "average_temperature_c": temperature,
        },
        timeout=30,
    )
    if response.is_success:
        result = response.json()
        st.metric(
            "Estimated crop yield",
            f"{result['prediction_tonnes_per_ha']:.2f} t/ha",
        )
        st.warning(result["warning"])
    else:
        st.error(response.json().get("detail", "Prediction failed."))

with st.expander("Model scope and limitations"):
    for limitation in metadata["limitations"]:
        st.write(f"- {limitation}")

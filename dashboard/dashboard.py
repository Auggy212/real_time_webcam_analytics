import streamlit as st
import requests
import time

st.set_page_config(page_title="Real-Time Webcam Analytics", layout="centered")

API_URL = "http://localhost:8000/metrics"  # Update to deployed API if hosted

st.title("📷 Real-Time Webcam Monitoring Dashboard")
st.markdown("Live feedback on user presence, posture, and gaze direction.")

refresh_rate = st.slider("⏱️ Refresh Interval (seconds)", 1, 10, 2)

placeholder = st.empty()

while True:
    try:
        response = requests.get(API_URL, timeout=2)
        if response.status_code == 200:
            data = response.json()
        else:
            data = {"error": "No metrics available"}

        with placeholder.container():
            st.subheader("📡 Live Analysis")
            col1, col2 = st.columns(2)

            with col1:
                st.metric("🧍 Face Visible", "✅" if data.get("face_visible") else "❌")
                st.metric("🎯 Centered Posture", "✅" if data.get("body_centered") else "❌")

            with col2:
                st.progress(min(data.get("posture_alignment_percent", 0)/100, 1.0),
                            text=f"Posture Alignment: {data.get('posture_alignment_percent', 0)}%")

        time.sleep(refresh_rate)

    except Exception as e:
        st.error(f"Error fetching data: {e}")
        time.sleep(refresh_rate)

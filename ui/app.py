import pandas as pd
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Chest X-ray Classifier", layout="centered")
st.title("Chest X-ray Classifier")
st.warning("Educational project only. Not a medical device and not for diagnosis.")

uploaded = st.file_uploader("Upload a chest X-ray (PNG or JPG)", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    st.image(uploaded, caption="Uploaded image", width=350)

    if st.button("Analyze"):
        with st.spinner("Running model..."):
            try:
                resp = requests.post(
                    API_URL,
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                    timeout=60,
                )
            except requests.exceptions.ConnectionError:
                st.error("Can't reach the API. Is uvicorn running on port 8000?")
                st.stop()

        if resp.status_code != 200:
            st.error(f"API error {resp.status_code}: {resp.text}")
        else:
            preds = resp.json()["predictions"]
            df = pd.DataFrame({"Disease": list(preds.keys()), "Score": list(preds.values())})
            df = df.sort_values("Score", ascending=False).reset_index(drop=True)

            st.subheader("Model confidence scores")
            st.caption("Relative model scores, not clinical probabilities.")
            st.bar_chart(df.set_index("Disease"))
            st.dataframe(df, hide_index=True, use_container_width=True)

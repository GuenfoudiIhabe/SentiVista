# streamlit_app.py
import streamlit as st
import requests

st.set_page_config(page_title="🧐 SentiVista Sentiment Analysis", layout="wide")
st.title("🧐 SentiVista Sentiment Analysis")

# --- 1) MODEL SELECTION ---
MODEL_OPTIONS = ["lr", "nb", "roberta"]
MODEL_NAMES   = {
    "lr":      "Logistic Regression",
    "nb":      "Naive Bayes",
    "roberta": "RoBERTa (deep)"
}

model = st.sidebar.selectbox(
    "Select model:",
    options=MODEL_OPTIONS,
    format_func=lambda x: MODEL_NAMES[x]
)

# --- 2) TEXT INPUT ---
raw = st.text_area(
    "Enter one sentence per line:",
    value="I love this app!\nThis product is terrible…",
    height=150,
)
texts = [t for t in raw.split("\n") if t.strip()]

# --- 3) SEND TO API ON BUTTON PRESS ---
if st.button("Analyze"):
    if not texts:
        st.warning("Please enter at least one line of text.")
    else:
        st.markdown(f"### Results using **{MODEL_NAMES[model]}**")
        try:
            # wrap in a spinner, since RoBERTa can be slow on first load
            with st.spinner(f"Contacting API for {MODEL_NAMES[model]}…"):
                resp = requests.post(
                    "https://senti-api-24294949938.europe-west1.run.app/predict",
                    json={"texts": texts, "model": model},
                    # no short timeout here
                )

            # inspect HTTP status first
            if resp.status_code != 200:
                st.error(f"🔴 API returned {resp.status_code}:")
                st.write(resp.text)   # raw error from Flask
            else:
                data = resp.json()

                # sanity-check the JSON shape
                if "sentiment_labels" not in data:
                    st.error("❌ Unexpected API response:")
                    st.write(data)
                else:
                    # display each sentence
                    for i, txt in enumerate(texts):
                        lbl   = data["sentiment_labels"][i]
                        color = "green" if lbl == "Positive" else "red"

                        # pull out confidence (traditional uses 'confidence', RoBERTa might use 'score')
                        detail = data.get("detailed_results", [{}])[i]
                        conf  = detail.get("confidence", detail.get("score", None))
                        conf_txt = f" ({conf*100:.1f}% confidence)" if conf is not None else ""

                        st.markdown(
                            f"**Text:** {txt}  \n"
                            f"**Sentiment:** "
                            f"<span style='color:{color};'>{lbl}</span>"
                            f"{conf_txt}",
                            unsafe_allow_html=True
                        )
        except requests.exceptions.RequestException as e:
            st.error(f"🔴 Network error: {e}")

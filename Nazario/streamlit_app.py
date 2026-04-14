import re
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Spam Detection", layout="centered")

model = joblib.load("Nazario/best_nb_model.pkl")
vectorizer = joblib.load("Nazario/tfidf_vectorizer.pkl")


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " ", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


st.title("Email Spam & Phishing Detection")

subject = st.text_input("Email Subject")
body = st.text_area("Email Body", height=200)

if st.button("Predict"):
    text = clean_text(subject + " " + body)

    if text == "":
        st.warning("Please enter some text.")
    else:
        X = vectorizer.transform([text])
        pred = model.predict(X)[0]

        # Safer probability handling
        if len(model.classes_) == 2:
            spam_class_index = list(model.classes_).index(1)
            prob = model.predict_proba(X)[0][spam_class_index]
        else:
            prob = 1.0 if pred == 1 else 0.0

        st.subheader("Result")
        st.write(f"Spam/Phishing Probability: {prob:.2%}")

        if pred == 1:
            st.error("Spam / Phishing Email")
        else:
            st.success("Legitimate Email")

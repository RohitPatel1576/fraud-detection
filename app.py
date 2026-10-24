import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fraud Detector", page_icon="💳")
st.title("💳 Credit Card Fraud Detector")
st.write("Select a transaction, and the model will tell you whether it is fraudulent or not—and explain why.")

@st.cache_resource
def load_artifacts():
    art = joblib.load("models/model.joblib")
    data = pd.read_csv("sample_transactions.csv")
    return art["model"], art["threshold"], art["features"], data

model, thr, feats, data = load_artifacts()

idx = st.slider("Transaction number", 0, len(data) - 1, 0)
row = data[feats].iloc[[idx]]

st.subheader("Details of Transaction")
st.dataframe(row)

if st.button("Do check"):
    p = model.predict_proba(row)[0, 1]

    if p >= thr:
        st.error(f"🚨 FRAUD. Probability: {p:.1%}")
    else:
        st.success(f"✅ Normal. Fraud probability: {p:.1%}")

    actual = "Fraud" if data["Class"].iloc[idx] == 1 else "Normal"
    st.caption(f"Real Label: {actual} | Threshold: {thr:.2f}")

    st.subheader("Why did model decide this?")
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(row)
    exp = shap.Explanation(
        values=sv[0],
        base_values=explainer.expected_value,
        data=row.iloc[0].values,
        feature_names=list(row.columns),
    )
    plt.figure()
    shap.plots.waterfall(exp, show=False)

    st.pyplot(plt.gcf())
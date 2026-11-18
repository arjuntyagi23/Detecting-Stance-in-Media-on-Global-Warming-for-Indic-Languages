import os
import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
st.set_page_config(page_title="Climate Stance Detector", page_icon="🌍", layout="centered")
LABELS = ["FAVOUR", "AGAINST", "NONE"]
INFO = {"FAVOUR": ("🟢", "Agrees that climate change is a serious concern"), "AGAINST": ("🔴", "Says climate change is not serious or is fake"), "NONE": ("⚪", "No opinion about it, or off topic")}
EXAMPLES = ["जलवायु परिवर्तन हमारे भविष्य के लिए बहुत बड़ा खतरा है।", "ग्लोबल वार्मिंग गंभीर नहीं है।", "आज मौसम बहुत अच्छा है।", "জলবায়ু পরিবর্তন আমাদের জন্য একটি গুরুতর হুমকি।", "বৈশ্বিক উষ্ণায়ন মোটেও গুরুতর সমস্যা নয়।", "বাংলাদেশ আজ ম্যাচ জিতেছে।", "Global warming bas drama hai."]
@st.cache_resource
def load_model(path):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    return tok, model
def get_probs(texts, tok, model):
    all_probs = []
    for i in range(0, len(texts), 32):
        batch = texts[i:i + 32]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).numpy()
        for p in probs:
            all_probs.append(p)
    return all_probs
def label_of(model, p):
    return model.config.id2label[int(p.argmax())]
st.title("🌍 Climate Stance Detector")
st.write("Type a **Hindi** or **Bengali** sentence. The model tells you if it agrees that *climate change and global warming is a serious concern*.")
path = st.sidebar.text_input("Model folder", "stance_model")
st.sidebar.markdown("**Labels**")
for lab in LABELS:
    st.sidebar.write(INFO[lab][0], "**" + lab + "**", "-", INFO[lab][1])
if not os.path.isdir(path):
    st.error("Model folder '" + path + "' not found.")
    st.write("Train the model in the Colab notebook, download `stance_model.zip`, unzip it next to `app.py`, then refresh this page.")
    st.stop()
with st.spinner("Loading model..."):
    tok, model = load_model(path)
tab1, tab2, tab3 = st.tabs(["Single sentence", "Many sentences (CSV)", "About"])
with tab1:
    pick = st.selectbox("Try an example (optional)", ["-"] + EXAMPLES)
    text = st.text_area("Your sentence", value="" if pick == "-" else pick, height=100)
    if st.button("Check stance", type="primary"):
        if text.strip() == "":
            st.warning("Please type a sentence first.")
        else:
            p = get_probs([text.strip()], tok, model)[0]
            lab = label_of(model, p)
            st.subheader(INFO[lab][0] + " " + lab)
            st.write(INFO[lab][1])
            st.write("Confidence:", str(round(float(p.max()) * 100, 1)) + "%")
            if p.max() < 0.6:
                st.info("The model is not very sure about this one.")
            chart = {}
            for i in range(len(p)):
                chart[model.config.id2label[i]] = float(p[i])
            st.bar_chart(pd.Series(chart))
with tab2:
    st.write("Upload a CSV file that has a column with text. You will get a table with the predicted label for each row.")
    f = st.file_uploader("CSV file", type=["csv"])
    if f is not None:
        df = pd.read_csv(f)
        col = st.selectbox("Which column has the text?", list(df.columns))
        if st.button("Predict all"):
            texts = df[col].astype(str).tolist()
            bar = st.progress(0.0, text="Working...")
            labs = []
            confs = []
            for i in range(0, len(texts), 32):
                probs = get_probs(texts[i:i + 32], tok, model)
                for p in probs:
                    labs.append(label_of(model, p))
                    confs.append(round(float(p.max()), 3))
                bar.progress(min(1.0, (i + 32) / len(texts)), text="Working...")
            bar.empty()
            df["predicted_label"] = labs
            df["confidence"] = confs
            st.dataframe(df)
            st.bar_chart(df["predicted_label"].value_counts())
            st.download_button("Download results", df.to_csv(index=False).encode("utf-8"), "predictions.csv", "text/csv")
with tab3:
    st.markdown("""
**Model:** IndicBERT v2 (`ai4bharat/IndicBERTv2-MLM-only`), fine-tuned for 3 labels. It knows 23 Indian languages, including Hindi and Bengali.

**Training data:** the English GWSD dataset (8 people voted on each sentence), translated into Hindi and Bengali with NLLB-200.

**Good to know**
- The Hindi and Bengali training text is machine translated, so it can be a little unnatural.
- Text written in English letters (like "Global warming bas drama hai") is harder for the model.
- Very short or sarcastic sentences can be wrong. Always check the confidence.
""")

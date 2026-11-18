import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

st.set_page_config(page_title="Climate Stance Detector", page_icon="🌍", layout="centered")

@st.cache_resource
def load_model():
    model_path = "./stance_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

tokenizer, model = load_model()

st.title("🌍 Climate Stance Detector")
st.markdown("Type a **Hindi**, **Bengali**, or **English** sentence. The model predicts if it agrees that *climate change and global warming is a serious concern*.")

# Navigation tabs
tab1, tab2 = st.tabs(["Single sentence", "Many sentences (CSV)"])

with tab1:
    st.subheader("Test a Single Sentence")
    
    # Text input area
    user_input = st.text_area("Enter sentence:", height=100)

    if st.button("Predict Stance", type="primary"):
        if user_input.strip() == "":
            st.warning("Please enter a sentence first.")
        else:
            with st.spinner("Analyzing stance..."):
                inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=128)
                with torch.no_grad():
                    outputs = model(**inputs)
                
                probs = torch.softmax(outputs.logits, dim=-1)
                pred_label = torch.argmax(probs, dim=-1).item()
                confidence = probs[0][pred_label].item() * 100
                
                # Label mapping
                labels = {0: "Against / Skeptical", 1: "Neutral", 2: "In Favor / Serious Concern"}
                result_text = labels.get(pred_label, "Unknown")
                
                st.success(f"**Prediction:** {result_text} ({confidence:.2f}% confidence)")

with tab2:
    st.subheader("Batch Prediction via CSV")
    st.markdown("Upload a CSV file containing a column with text to get predictions for each row.")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file is not None:
        import pandas as pd
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:", df.head())
        
        # Let user pick which column contains the text
        text_column = st.selectbox("Select the column containing sentences:", df.columns)
        
        if st.button("Run Batch Predictions"):
            results = []
            with st.spinner("Running predictions on CSV..."):
                for text in df[text_column].astype(str):
                    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
                    with torch.no_grad():
                        outputs = model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)
                    pred_label = torch.argmax(probs, dim=-1).item()
                    labels = {0: "Against / Skeptical", 1: "Neutral", 2: "In Favor / Serious Concern"}
                    results.append(labels.get(pred_label, "Unknown"))
                
                df["Predicted Stance"] = results
                st.success("Batch prediction complete!")
                st.dataframe(df)
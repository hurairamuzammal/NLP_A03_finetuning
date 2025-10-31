import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

@st.cache_resource
def load_model():
    model_path = "./bert_sentiment/final"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

st.set_page_config(page_title="Sentiment Analysis", page_icon="🎭", layout="wide")

st.title("Sentiment Analysis with BERT")
st.write("Analyze the sentiment of your text using a fine-tuned BERT model")

tokenizer, model, device = load_model()
# tab1 = st.tabs(["Single Text"])
st.sidebar.title("About")
st.sidebar.info("This app uses a fine-tuned BERT model for sentiment analysis. Upload your model checkpoint to the './bert_sentiment/final' directory.")
st.sidebar.markdown("**Model:** bert-base-uncased")
st.sidebar.markdown(f"**Device:** {device}")

user_input = st.text_area("Enter your text here:", height=150, placeholder="Type or paste your text here...")

if st.button("Analyze Sentiment", type="primary"):
    if user_input.strip():
        with st.spinner("Analyzing..."):
            inputs = tokenizer(user_input, padding="max_length", max_length=128, truncation=True, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            model.eval()
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                prediction = torch.argmax(probs, dim=-1)
                confidence = probs.max().item()
            label = model.config.id2label[prediction.item()]
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted Sentiment", label)
            with col2:
                st.metric("Confidence", f"{confidence:.2%}")
            st.subheader("Probability Distribution")
            prob_data = {model.config.id2label[i]: probs[0][i].item() for i in range(len(probs[0]))}
            st.bar_chart(prob_data)
    else:
        st.warning("Please enter some text to analyze")

# Sidebar options for model selection
st.sidebar.title("Model Type")
model_option = st.sidebar.radio(
    "Choose model architecture:",
    ("Encoder: BERT", "Decoder: GPT-2", "Encoder-Decoder: T5")
)

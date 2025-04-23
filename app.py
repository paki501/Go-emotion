import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import plotly.express as px

# Set up page
st.set_page_config(page_title="🎭 GoEmotions Classifier", layout="centered")

st.title("🎭 GoEmotions Classifier")
st.markdown("Classify multiple emotions from a given text using **BERT** trained on the **GoEmotions** dataset by Google.")

# Load tokenizer and model
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained("monologg/bert-base-cased-goemotions-original")
    model = BertForSequenceClassification.from_pretrained("monologg/bert-base-cased-goemotions-original")
    return tokenizer, model

tokenizer, model = load_model()
model.eval()

# Load emotion labels
EMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
    'curiosity', 'desire', 'disappointment', 'disapproval', 'embarrassment', 'excitement',
    'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride',
    'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

# Input area
text = st.text_area("Enter your text below 👇", height=150, placeholder="E.g., I'm so happy today!")

if st.button("🔍 Analyze"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        # Tokenization
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits)[0].cpu().numpy()

        # Apply threshold to detect relevant emotions
        threshold = 0.3
        predictions = [(label, float(p)) for label, p in zip(EMOTION_LABELS, probs) if p > threshold]

        st.subheader("🧠 Predicted Emotions")
        if predictions:
            for emotion, score in sorted(predictions, key=lambda x: x[1], reverse=True):
                st.markdown(f"- **{emotion.capitalize()}**: `{score:.2f}`")
        else:
            st.info("No strong emotions detected above the threshold.")

        # Plotting
        fig = px.bar(
            x=[e[0] for e in predictions],
            y=[e[1] for e in predictions],
            labels={'x': 'Emotion', 'y': 'Confidence'},
            title="Emotion Confidence Scores",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

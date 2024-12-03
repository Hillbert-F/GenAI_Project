# sentiment_analysis.py
import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

def sentiment_analysis_app():
    # Load pre-trained BERT model for sentiment analysis
    tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

    # User input
    st.header("Sentiment Analysis")
    st.markdown("Enter a text to analyze its sentiment:")
    user_text = st.text_area("Enter your text here:")

    if st.button("Analyze Sentiment"):
        if user_text:
            # Tokenize input text and get model predictions
            inputs = tokenizer(user_text, return_tensors='pt', truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Convert logits to probabilities
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1).squeeze().tolist()

            # Define sentiment labels
            labels = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']

            # Display results
            st.subheader("Sentiment Analysis Results")
            st.write(f"Predicted Sentiment: {labels[torch.argmax(logits)]}")
            
            st.subheader("Sentiment Scores:")
            for label, score in zip(labels, probabilities):
                st.write(f"{label}: {score:.2f}")
        else:
            st.write("Please enter some text for analysis.")

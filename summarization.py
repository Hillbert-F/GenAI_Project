import streamlit as st
from transformers import pipeline

def display_summarization():
    st.title("Text Summarization")
    st.markdown("""
        Use this tool to generate concise summaries of long texts. 
        Simply enter your text in the input box below, and the model will provide a summary.
    """)

def summarization_app():
    # Load summarization pipeline
    summarizer = pipeline("summarization")

    # User input
    st.header("Text Summarization")
    st.markdown("Enter a piece of text to generate its summary:")
    user_text = st.text_area("Enter your text here:")

    if st.button("Summarize Text"):
        if user_text:
            # Generate summary
            summary = summarizer(user_text, max_length=150, min_length=30, do_sample=False)
            st.subheader("Summary")
            st.write(summary[0]['summary_text'])
        else:
            st.write("Please enter some text to summarize.")
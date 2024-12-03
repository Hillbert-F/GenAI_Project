# app.py
import streamlit as st
from text_prediction_visualization import text_prediction_visualization_app
from text_prediction_visualization_llama import text_prediction_visualization_llama_app
from text_stepwise_generation_llama import text_stepwise_generation_llama_app
from text_stepwise_generation import text_stepwise_generation_app
from summarization import summarization_app
from sentiment_analysis import sentiment_analysis_app
from interactive_text_input import interactive_text_input_app
from code_qa import generate_code_with_params, load_code_qa_model

# Homepage content
def homepage():
    st.title("Welcome to Transformers in Action")
    st.markdown("""
        ### Explore the Power of Transformers!
        This application demonstrates the versatility of Transformer models in various tasks:
        - **Text Summarization**: Create concise summaries of long texts.
        - **Text Generation**: Generate coherent text, step-by-step or via visualization.
        - **Sentiment Analysis**: Identify the sentiment in any given text.
        - **Interactive Text Input**: Experiment with Transformer-based text generation.
        - **Code Q&A**: Solve coding problems and understand Transformer concepts.
        
        Please feel free to explore! This user-friendly platform is designed for you to learn Transformers from zero by applications showcasing and AI Helper!

        Use the sidebar to navigate through the functionalities and the Help page for detailed explanations.
    """)

# Help page content
def help_page():
    st.title("Help & Instructions")
    st.markdown("""
        ## How to Use This Application
        - **Text Summarization**: Enter a long piece of text, and the model will generate a concise summary.
        - **Text Generation**: Start with a prompt and generate text step-by-step or visualize token predictions.
        - **Sentiment Analysis**: Input text to get its sentiment (e.g., positive, negative, or neutral).
        - **Interactive Text Input**: Generate text interactively by selecting tokens.
        - **Code Q&A**: Ask coding-related questions and explore Transformer functionalities.

        ### Parameter Explanations
        - **Temperature**: Controls randomness in generation. Lower values make output deterministic; higher values increase creativity.
        - **Top-K Sampling**: Limits token selection to the top K most probable options. Smaller K keeps output focused; larger K adds variability.
        - **Max Tokens**: The maximum number of tokens (words) generated.

        ### Tips
        - Experiment with parameters to see their effect on the output.
        - Use shorter inputs for quicker results, especially with large models.
    """)

# Function to render Code Q&A section
def display_code_qa():
    st.header("Ask Coding and Conceptual Questions to Your AI Assistant!")
    st.markdown("""
    Welcome to the Transformer Code Q&A section! Here are some example prompts to get started:
    
    - **Basic Coding Tasks**:
      - "Write a Python function to calculate the factorial of a number."
      - "Provide a Python implementation of quick sort."
    
    - **Transformer Concepts**:
      - "Explain the concept of multi-head attention with code."
      - "Generate Python code for a Transformer encoder layer."
    
    - **Advanced Topics**:
      - "Write a custom loss function for training a Transformer."
      - "Implement positional encoding in Transformers."
    """)

    load_code_qa_model()

    # User input for prompt
    prompt = st.text_area("Enter your coding question or task description:", key="qa_prompt")

    # Parameter adjustment
    st.subheader("Adjust Generation Parameters")
    temperature = st.slider("Temperature (creativity level):", 0.1, 1.0, 0.7, key="qa_temp")
    max_tokens = st.slider("Max Tokens (length of output):", 50, 1024, 512, key="qa_tokens")
    top_k = st.slider("Top-K Sampling (limits randomness):", 1, 100, 50, key="qa_top_k")

    if st.button("Generate Code", key="qa_button"):
        if prompt.strip():
            with st.spinner("Generating code..."):
                result = generate_code_with_params(prompt, temperature, max_tokens, top_k)
            st.subheader("Generated Code")
            st.code(result, language="python")
        else:
            st.warning("Please enter a valid prompt!")

# Main application function
def main():
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Homepage", "Text Summarization", "Text Generation", "Sentiment Analysis", 
         "Interactive Text Input", "Ask Transformer Code Q&A", "Help Page"]
    )

    # Page routing
    if page == "Homepage":
        homepage()
    elif page == "Text Summarization":
        summarization_app()
    elif page == "Text Generation":
        st.subheader("Text Generation Options")
        gen_mode = st.selectbox(
            "Choose a text generation method",
            ["Next Word Prediction and Visualization", "Stepwise Text Generation"]
        )
        model_choice = st.radio(
            "Choose a model for text generation:",
            ["GPT-2", "LLaMA 3.2"]
        )
        if gen_mode == "Next Word Prediction and Visualization":
            if model_choice == "GPT-2":
                text_prediction_visualization_app()
            elif model_choice == "LLaMA 3.2":
                text_prediction_visualization_llama_app()
        elif gen_mode == "Stepwise Text Generation":
            if model_choice == "GPT-2":
                text_stepwise_generation_app()
            elif model_choice == "LLaMA 3.2":
                text_stepwise_generation_llama_app()
    elif page == "Sentiment Analysis":
        sentiment_analysis_app()
    elif page == "Interactive Text Input":
        interactive_text_input_app()
    elif page == "Ask Transformer Code Q&A":
        display_code_qa()
    elif page == "Help Page":
        help_page()

if __name__ == "__main__":
    main()
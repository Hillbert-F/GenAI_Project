# text_prediction_visualization.py
import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import matplotlib.pyplot as plt
import seaborn as sns

def text_prediction_visualization_app():
    # Load pre-trained GPT-2 model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", output_attentions=True)

    # User input
    st.header("Text Prediction and Attention Visualization")
    st.markdown("Enter an incomplete sentence to predict the next words:")
    user_input = st.text_area("Enter your text here:")
    temperature = st.slider("Set the temperature for text generation:", 0.1, 1.5, 1.0, 0.1)

    if st.button("Generate and Visualize"):
        if user_input:
            # Tokenize input and generate output
            inputs = tokenizer(user_input, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits

                # Apply temperature scaling to logits
                scaled_logits = logits[:, -1, :] / temperature
                probabilities = torch.softmax(scaled_logits, dim=-1)

                # Sample next token based on probabilities
                next_token = torch.multinomial(probabilities, num_samples=1).item()  # Convert tensor to scalar
                predicted_word = tokenizer.decode(next_token)

                attentions = outputs.attentions

            # Display prediction
            st.subheader("Predicted Next Word")
            st.write(predicted_word)

            # Attention Visualization (last layer for simplicity)
            attention = attentions[-1][0]  # (num_heads, seq_len, seq_len)
            attention_mean = attention.mean(0).numpy()  # Average over all heads

            # Extract token labels without special characters
            token_ids = inputs["input_ids"][0]
            tokens = [tokenizer.decode([t]).strip().replace("Ġ", "") for t in token_ids]

            # Plot attention heatmap
            st.subheader("Attention Heatmap")
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(attention_mean, annot=False, cmap="Blues", cbar=True, ax=ax,
                        xticklabels=tokens,
                        yticklabels=tokens)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            ax.set_yticklabels(ax.get_yticklabels())
            st.pyplot(fig)

        else:
            st.write("Please enter a text for prediction.")

""" # text_prediction_visualization.py
import streamlit as st
import requests
import matplotlib.pyplot as plt
import seaborn as sns

def text_prediction_visualization_app():
    # Model Selection
    st.sidebar.header("Model Selection")
    model_choice = st.sidebar.radio("Choose a model for text prediction:", ["GPT-2", "LLaMA 3.2"])

    # Set up API for LLaMA 3.2 if selected
    api_token = None
    api_url = None
    if model_choice == "LLaMA 3.2":
        api_token = st.sidebar.text_input("Enter your Hugging Face API Token here:")
        api_url = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-1B"

    # User input
    st.header("Text Prediction and Attention Visualization")
    st.markdown("Enter an incomplete sentence to predict the next words:")
    user_input = st.text_area("Enter your text here:")
    temperature = st.slider("Set the temperature for text generation:", 0.1, 1.5, 1.0, 0.1)

    if st.button("Generate and Visualize"):
        if user_input and model_choice == "LLaMA 3.2" and not api_token:
            st.error("Please enter your Hugging Face API Token to proceed with LLaMA 3.2.")
        elif user_input:
            if model_choice == "GPT-2":
                # Tokenize input and generate output using GPT-2 (local model)
                from transformers import GPT2Tokenizer, GPT2LMHeadModel
                import torch

                tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
                model = GPT2LMHeadModel.from_pretrained("gpt2", output_attentions=True)
                inputs = tokenizer(user_input, return_tensors="pt")

                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits

                    # Apply temperature scaling to logits
                    scaled_logits = logits[:, -1, :] / temperature
                    probabilities = torch.softmax(scaled_logits, dim=-1)

                    # Sample next token based on probabilities
                    next_token = torch.multinomial(probabilities, num_samples=1).item()
                    predicted_word = tokenizer.decode([next_token])

                    attentions = outputs.attentions

            elif model_choice == "LLaMA 3.2":
                # Use Hugging Face API for LLaMA 3.2
                headers = {"Authorization": f"Bearer {api_token}"}
                payload = {
                    "inputs": user_input,
                    "parameters": {
                        "temperature": temperature,
                        "top_k": 5,
                        "return_full_text": False
                    }
                }

                response = requests.post(api_url, headers=headers, json=payload)
                if response.status_code == 200:
                    output = response.json()
                    predicted_word = output[0]["generated_text"]
                    attentions = None  # Inference API does not provide attentions

                else:
                    st.error(f"Error: {response.status_code}, {response.text}")
                    return

            # Display prediction
            st.subheader("Predicted Next Word")
            st.write(predicted_word)

            # Attention Visualization (only available for GPT-2)
            if model_choice == "GPT-2" and attentions is not None:
                attention = attentions[-1][0]  # (num_heads, seq_len, seq_len)
                attention_mean = attention.mean(0).numpy()  # Average over all heads

                # Extract token labels without special characters
                token_ids = inputs["input_ids"][0]
                tokens = [tokenizer.decode([t]).strip().replace("Ġ", "") for t in token_ids]

                # Plot attention heatmap
                st.subheader("Attention Heatmap")
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(attention_mean, annot=False, cmap="Blues", cbar=True, ax=ax,
                            xticklabels=tokens,
                            yticklabels=tokens)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
                ax.set_yticklabels(ax.get_yticklabels())
                st.pyplot(fig)

        else:
            st.write("Please enter a text for prediction.") """

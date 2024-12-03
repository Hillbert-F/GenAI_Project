## text_stepwise_generation_llama.py
import requests
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Replace this with your Hugging Face API token
API_TOKEN = "" # Replace with your API token

# Define API URL for LLaMA 3.2-1B
API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-1B"

# Function to query the model using Hugging Face Inference API
def query_llama(payload):
    headers = {
        "Authorization": f"Bearer {API_TOKEN}"
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        return None
    return response.json()

# Stepwise text generation with LLaMA 3.2-1B
def text_stepwise_generation_llama_app():
    st.title("Stepwise Text Generation with Ollama (LLaMA 3.2-1B)")
    user_input = st.text_input("Enter the beginning of your text here:", "")

    if "generated_text" not in st.session_state:
        st.session_state.generated_text = user_input
        st.session_state.step = 0

    if st.button("Start Generation"):
        st.session_state.generated_text = user_input
        st.session_state.step = 0

    if st.button("Generate Next Word"):
        if st.session_state.generated_text.strip():
            payload = {
                "inputs": st.session_state.generated_text,
                "parameters": {
                    "max_new_tokens": 1,  # Generate one word at a time
                    "return_full_text": False,
                    "output_attentions": True
                }
            }
            output = query_llama(payload)

            # Extract generated token and update the generated text
            if output and isinstance(output, list) and len(output) > 0:
                generated_token = output[0].get('generated_text', '').strip()
                if generated_token:
                    st.session_state.generated_text += f" {generated_token}"

                    # Extract attention weights if available
                    attentions = output[0].get('attentions')
                    if attentions:
                        attention_matrix = attentions[-1]  # Get the last layer's attention matrix
                        visualize_attention(attention_matrix)
                else:
                    st.error("No new token generated. Please try again.")
            else:
                st.error("Error in response from model. Please try again.")

    # Display the generated text so far
    st.subheader("Generated Text So Far")
    st.write(st.session_state.generated_text)

# Function to visualize attention weights
def visualize_attention(attention_matrix):
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_matrix, annot=False, cmap="Blues", cbar=True)
    plt.xlabel("Generated Words")
    plt.ylabel("Input Words")
    plt.title(f"Attention Heatmap for Step {st.session_state.step + 1}")
    st.pyplot(plt)
    st.session_state.step += 1

# Ensure the script runs correctly when executed
if __name__ == "__main__":
    text_stepwise_generation_llama_app()
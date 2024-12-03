import streamlit as st
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def text_prediction_visualization_llama_app():
    # API information
    api_url = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-1B"
    api_token = ""  # Replace with your API token

    # User input for initial text
    st.header("Next Word Prediction and Visualization (LLaMA 3.2-1B)")
    user_input = st.text_area("Enter your text here:")
    temperature = st.slider("Set the temperature for text generation:", 0.1, 1.5, 1.0, 0.1)

    if st.button("Generate and Visualize"):
        if user_input:
            headers = {"Authorization": f"Bearer {api_token}"}

            # Request the API to predict the next word
            payload = {
                "inputs": user_input,
                "parameters": {
                    "max_new_tokens": 1,
                    "return_full_text": False,
                    "temperature": temperature  # Apply the selected temperature
                }
            }

            response = requests.post(api_url, headers=headers, json=payload)
            if response.status_code == 200:
                output = response.json()
                next_word = output[0]["generated_text"].strip()

                # Display the predicted word
                st.subheader("Predicted Next Word")
                st.write(next_word)

                # Simulate attention heatmap (since attention information is not provided by API)
                st.subheader("Attention Heatmap (Simulated)")
                steps = len(user_input.split()) + 1
                attention_matrix = np.random.rand(steps, steps)  # Simulated attention for visualization

                tokens = user_input.split() + [next_word]  # Use input tokens + predicted word

                # Plot attention heatmap
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(attention_matrix, annot=False, cmap="Blues", cbar=True, ax=ax,
                            xticklabels=tokens,
                            yticklabels=tokens)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
                ax.set_yticklabels(ax.get_yticklabels())
                st.pyplot(fig)

            else:
                st.error(f"Error: {response.status_code}, {response.text}")

if __name__ == "__main__":
    text_prediction_visualization_llama_app()
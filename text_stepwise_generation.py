# text_stepwise_generation.py
import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import matplotlib.pyplot as plt
import seaborn as sns

# Dynamically select data type based on device capability
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

def text_stepwise_generation_app():
    # Load pre-trained GPT-2 model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", output_attentions=True, torch_dtype=torch_dtype)
    
    # Ensure the model is moved to the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Session state for managing text and attention history
    if "generated_text" not in st.session_state:
        st.session_state.generated_text = ""
    if "attention_history" not in st.session_state:
        st.session_state.attention_history = []

    # User input
    st.header("Stepwise Text Generation with Attention Visualization")
    st.markdown("Enter an incomplete sentence to predict the next words stepwise:")
    user_input = st.text_area("Enter your text here:")

    if st.button("Start Generation"):
        if user_input:
            st.session_state.generated_text = user_input
            st.session_state.attention_history = []

    if st.button("Generate Next Word"):
        if st.session_state.generated_text:
            # Tokenize current input and ensure tensors are on the same device as the model
            inputs = tokenizer(st.session_state.generated_text, return_tensors="pt").to(device)

            with torch.no_grad():
                # Ensure inputs are on the same device as the model
                outputs = model(**inputs)
                logits = outputs.logits
                attentions = outputs.attentions

                # Apply temperature scaling to logits
                temperature = 1.0  # Set fixed temperature for simplicity
                scaled_logits = logits[:, -1, :] / temperature
                probabilities = torch.softmax(scaled_logits, dim=-1)

                # Sample next token based on probabilities
                next_token = torch.multinomial(probabilities, num_samples=1).item()
                predicted_word = tokenizer.decode(next_token)

                # Update generated text with the new word
                st.session_state.generated_text += " " + predicted_word

                # Store attention for visualization (use the attention from the last layer)
                # Extract only the relevant portion of the attention matrix (up to the current length)
                current_seq_len = inputs["input_ids"].shape[1] + 1  # Including the newly generated token
                attention_matrix = attentions[-1][0].mean(0).cpu().numpy()[:current_seq_len, :current_seq_len]
                st.session_state.attention_history.append(attention_matrix)

    # Display generated text so far
    if st.session_state.generated_text:
        st.subheader("Generated Text So Far")
        st.write(st.session_state.generated_text)

    # Attention Visualization for each step
    if st.session_state.attention_history:
        st.subheader("Attention Heatmaps for Each Step")
        for i, attention_matrix in enumerate(st.session_state.attention_history):
            # Extract token labels without special characters
            tokens = st.session_state.generated_text.split()[:attention_matrix.shape[0]]

            # Plot attention heatmap for each generation step
            st.write(f"Step {i + 1}")
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(attention_matrix, annot=False, cmap="Blues", cbar=True, ax=ax,
                        xticklabels=tokens,
                        yticklabels=tokens)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            ax.set_yticklabels(ax.get_yticklabels())
            st.pyplot(fig)
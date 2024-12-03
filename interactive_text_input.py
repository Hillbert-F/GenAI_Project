# interactive_text_input.py
import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import matplotlib.pyplot as plt

# Dynamically select data type based on device capability
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

def interactive_text_input_app():
    # Load pre-trained GPT-2 model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", output_attentions=True, torch_dtype=torch_dtype)
    
    # Move the model to the appropriate device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # Session state to manage generated text and current input tokens
    if "generated_text" not in st.session_state:
        st.session_state.generated_text = ""
    if "input_ids" not in st.session_state:
        st.session_state.input_ids = None

    # User input to start text generation
    st.header("Interactive Text Input Experience")
    user_input = st.text_area("Enter the beginning of your text here:")

    if st.button("Start Generation"):
        if user_input:
            # Tokenize user input and move to the correct device
            st.session_state.generated_text = user_input.strip()
            st.session_state.input_ids = tokenizer.encode(user_input, return_tensors="pt").to(device)

    if st.session_state.input_ids is not None:
        # Generate next possible tokens
        with torch.no_grad():
            outputs = model(
                st.session_state.input_ids,
                return_dict=True,
                output_attentions=False
            )
            # Get the logits (unnormalized probabilities) of the next token
            logits = outputs.logits[:, -1, :]
            # Apply top-k sampling to get the top 5 tokens
            top_k = 8
            probs = torch.softmax(logits, dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, top_k)

            # Decode the top k tokens to words
            options = [tokenizer.decode([idx]) for idx in top_k_indices[0]]
            probabilities = top_k_probs[0].tolist()

        # Display options to the user for the next word
        st.subheader("Choose the next word")
        selected_word = st.selectbox("Select one of the options:", options)

        # Plot the probabilities of the top k tokens
        st.subheader("Token Probabilities")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(options, probabilities, color='skyblue')
        ax.set_xlabel("Probability")
        ax.set_title("Top 8 Token Probabilities")
        ax.invert_yaxis()  # To display the highest probability on top
        st.pyplot(fig)

        # Add the selected word to the generated text
        if st.button("Add Word"):
            if selected_word:
                # Update generated text with selected word
                st.session_state.generated_text += " " + selected_word
                # Update input_ids to include the selected word
                new_input_ids = tokenizer.encode(selected_word, return_tensors="pt").to(device)
                st.session_state.input_ids = torch.cat([st.session_state.input_ids, new_input_ids], dim=1)

    # Display the generated text so far
    if st.session_state.generated_text:
        st.subheader("Generated Text So Far")
        st.write(st.session_state.generated_text)
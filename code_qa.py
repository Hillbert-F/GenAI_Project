import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Function to load the Q&A model
def load_code_qa_model():
    # Check if GPU is available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    st.write(f"Using device: {device}")

    # Define the local model path
    model_path = r"D:\OneDrive - Vanderbilt\Desktop\Hillbert's PC Files\My Grad Life\Fall 2024\DS5690_GenAI_Models\Final\Qwen2.5-Coder-0.5B-Instruct"   # Original Model #
    # model_path = r"D:\OneDrive - Vanderbilt\Desktop\Hillbert's PC Files\My Grad Life\Fall 2024\DS5690_GenAI_Models\Final\tuned_qwen2.5-coder"         # Fine-tuned Model #

    # Check if the model is already loaded in the session state
    if "code_qa_model" not in st.session_state:
        st.write("Loading Code Q&A model from local directory...")

        # Load tokenizer and model from the local path
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32
        )
        model.to(device)

        # Store the model and tokenizer in session state for reuse
        st.session_state["code_qa_model"] = model
        st.session_state["code_qa_tokenizer"] = tokenizer
    else:
        st.write("Code Q&A model already loaded.")

# Function to generate code based on the user's input prompt and parameters
def generate_code_with_params(prompt, temperature, max_tokens, top_k):
    # Retrieve the model and tokenizer from session state
    model = st.session_state["code_qa_model"]
    tokenizer = st.session_state["code_qa_tokenizer"]

    # Move to GPU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Perform inference with the model
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=max_tokens,  # Limit the number of tokens generated
            temperature=temperature,  # Adjust creativity in generation
            top_k=top_k  # Limit the sampling to the top K options
        )
    
    # Decode the generated output into readable text
    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_code
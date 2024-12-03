# Interactive Transformer Learning Hub: Exploring AI Models in Practice

## Project Overview
Welcome to the Interactive Transformer Learning Hub, a user-friendly platform designed to help users explore and understand Transformer models through hands-on applications. This project integrates various functionalities of Transformer-based models, including text summarization, text generation, sentiment analysis, and more, to provide a comprehensive learning experience.

## Features
1. **Homepage**: An introduction to the platform, outlining the functionalities and their real-world applications.
2. **Help Page**: A detailed guide on how to use the platform, with explanations of adjustable parameters.
3. **Applications**:
    - **Text Summarization**: Generate concise summaries of lengthy texts.
    - **Text Generation**: Stepwise Text Generation: Generate text one word at a time with attention visualization.
    - **Next Word Prediction**: Visualize attention weights while generating the next word.
    - **Sentiment Analysis**: Predict the sentiment (positive, negative, neutral) of any given text.
    - **Interactive Text Input**: Experiment with Transformer-based text generation interactively.
    - **Code Q&A**: Ask coding-related questions to explore concepts and practical Transformer applications.

## Technologies Used
- Streamlit: Interactive web application framework.
- Hugging Face Transformers: Pre-trained Transformer models, `GPT-2`, `BERT`, `LLaMA3.2-1B`, and `Qwen2.5-Coder-0.5B-Instruct`.
- PyTorch: Backend for Transformer model operations.

## Setup Instructions
**1. Prerequisites**
- Python 3.8+
- GPU support is optional but recommended for faster performance.

**2. Clone the Repository**
```bash
git clone https://github.com/Hillbert-F/GenAI_Project.git
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Models**

Ensure the required models are downloaded locally (`GPT-2` and `BERT` will automatically download by executing `app.py`)
- `GPT-2`
- `BERT` for Sentiment Analysis
- `LLaMA3.2`, for scripts `text_stepwise_generation_llama.py` and `text_prediction_visualization_llama.py`
    - Fill in your Hugging Face API if you have, or
    - Deploy `LLaMa3.2` locally and modify above two scripts

**5. Run the Application**
Launch the Streamlit app:

```bash
streamlit run app.py
```

## Fine-Tuning Qwen2.5-Coder
### Goal
The fine-tuning focuses on adapting `Qwen2.5-Coder` to address coding and conceptual questions for transformer beginners. This task involves demonstrating how a pre-trained model can be specialized for a unique use case.

### Data Preparation
The dataset is derived from the [codeparrot-clean-train dataset](https://huggingface.co/datasets/codeparrot/codeparrot-clean-train). Data processing is handled using the `data_processing.ipynb notebook`, which performs the following steps:

**1. Filtering**: Extracts entries relevant to transformers or general coding tasks.

**2. Prompt Engineering**: Formats data into "prompt-completion" pairs. For example **(not actual data)**:
- **Prompt**: "What is self-attention in transformers?"
- **Completion**: "Self-attention computes dependencies between all words in a sequence to determine which words should be focused on."

**3. Splitting**: Divides the dataset into training (`train_data.json`) and validation (`val_data.json`) subsets.

### Fine-Tuning Process
The fine-tuning workflow is detailed in the `Qwen2.5_Fine_Tuning.ipynb` notebook. But as there are temporarily no datasets designated for our goal, the tuned model **does not perform well** and is not recommended for use. The notebook was kept simply **to show the idea and logic to fine-tune a pre-trained model** to be used in a particular situation.

## Acknowledgments
This project was developed as part of a course on Generative AI Models in Theory and Practice, leveraging cutting-edge Transformer-based models and APIs from Hugging Face.
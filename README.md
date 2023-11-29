# Personal RAG QA Advanced Chatbot

**Creator:** Subham Thirani

## Technology Stack
- Python
- Langchain
- LLM (Local Language Model)
- Generative AI
- Streamlit

## Description
This repository houses a sophisticated Personal RAG (Retrieval-Augmented Generation) QA (Question-Answering) chatbot, built on top of LLM (Local Language Model) using Langchain. Unlike traditional chatbots, this system not only processes queries with context but also modifies the query using conversation history, significantly enhancing the quality of responses.

## Setup Instructions

### Local LLM (Using Ollama for Llama2 on Mac)
1. Use Ollama to obtain Llama2 on your Mac.

### Google Palm Model
1. Obtain the API key for the Google Palm Model and set it as an environment variable.

### Installation
1. Install the required libraries by running:
    ```bash
    pip install -r requirements.txt
    ```

### Usage
1. Load documents, split, and embed them in vector space before using the UI.
2. Utilize the `langchain_helper.py` script for this purpose (ensure to update the path of the directory).
    ```bash
    python langchain_helper.py
    ```

3. Run the main application using Streamlit:
    ```bash
    streamlit run main.py
    ```

Now, you can interact with the chatbot through the provided UI. Enjoy the advanced capabilities of the Personal RAG QA chatbot!

Feel free to contribute, report issues, or suggest improvements. Happy chatting!

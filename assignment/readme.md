# Chat With Your Documents

A simple Streamlit application for chatting with your PDF documents using a Retrieval-Augmented Generation (RAG) pipeline. The app can also function as a standard chatbot without RAG.

This project supports both **Hugging Face** and **Azure OpenAI** for embeddings and language models, with **Pinecone** serving as the vector database for persistent, per-document storage.

## Features

- **Toggle RAG Mode**: Easily switch between a standard LLM chat and a RAG-powered chat.
- **PDF Upload**: Upload your PDF documents to create a searchable knowledge base.
- **Persistent Storage**: Each uploaded document gets its own persistent index in Pinecone, created automatically from the filename. No re-processing is needed.
- **Multi-Provider Support**:
  - **LLMs**: Hugging Face Inference API (`Mistral-7B`) or Azure OpenAI (`gpt-4o`).
  - **Embeddings**: Hugging Face Inference API or Azure OpenAI.

## Setup

Follow these steps to run the application locally.

### 1. Prerequisites

- Python 3.9+
- API keys for:
  - [Hugging Face](https://huggingface.co/settings/tokens)
  - [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service)
  - [Pinecone](https://www.pinecone.io/)

### 2. Installation

Clone this repository or download the files.

Create a virtual environment:```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

Install the required Python packages:
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a file named `.env` in the root of your project directory and add your API keys. Use the `.env.example` below as a template.

**.env.example**```env
# Azure OpenAI Credentials
AZURE_OPENAI_ENDPOINT="YOUR_AZURE_ENDPOINT" # e.g., https://your-resource-name.openai.azure.com/
AZURE_OPENAI_KEY="YOUR_AZURE_API_KEY"
DEPLOYMENT_NAME="YOUR_CHAT_MODEL_DEPLOYMENT_NAME" # e.g., gpt-4o

# Hugging Face API Key
HUGGINGFACE_API_KEY="hf_..."

# Pinecone API Key
PINECONE_API_KEY="YOUR_PINECONE_API_KEY"
```

### 4. Create `requirements.txt`

Create a `requirements.txt` file with the following content:

```txt
streamlit
langchain
langchain-community
pypdf
langchain-huggingface
sentence-transformers
huggingface_hub
langchain-text-splitters
langchain-pinecone
langchain-openai
pinecone-client
python-dotenv
```

## How to Run

Launch the Streamlit application from your terminal:

```bash
streamlit run app.py
```

The application will open in your web browser.

## Usage

1.  **Select Mode**: Use the "Enable RAG" toggle in the sidebar.
    - **RAG On**: Choose an embedding model provider (Hugging Face or Azure OpenAI) and upload a PDF. The app will process the file once and store it in a dedicated Pinecone index. You can then ask questions about the document.
    - **RAG Off**: Choose an LLM provider and chat directly with the model without any document context.
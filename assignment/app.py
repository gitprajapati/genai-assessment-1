# Import necessary libraries
import os
import uuid  # Used for generating unique IDs for document chunks
import time  # Used to add a delay after creating a Pinecone index
import streamlit as st  # The framework for building the web app
from dotenv import load_dotenv  # For loading environment variables from a .env file
from pinecone import Pinecone, ServerlessSpec  # Pinecone's Python client for vector database management
from langchain_community.document_loaders import PyPDFLoader  # For loading PDF files
from langchain_text_splitters import RecursiveCharacterTextSplitter  # For splitting text into smaller chunks
from langchain_pinecone import PineconeVectorStore  # LangChain's integration for using Pinecone as a vector store
from langchain_huggingface import HuggingFaceEndpointEmbeddings, ChatHuggingFace, HuggingFaceEndpoint  # LangChain integrations for Hugging Face models
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI  # LangChain integrations for Azure OpenAI models
from langchain_core.messages import HumanMessage, SystemMessage  # LangChain's schema for representing chat messages

# Load environment variables from a .env file into the environment
# This is useful for keeping sensitive information like API keys out of the code
load_dotenv()

# --- Constants ---

# Default system prompt for the RAG assistant. It instructs the model on how to behave.
DEFAULT_USER_PROMPT = "You are a helpful assistant. Your task is to answer the user's question based only on the following context. If the answer is not in the context, say 'I do not have enough information to answer that question.'"
# Suffix to be added to the user-defined prompt, which will be populated with the retrieved context.
CONTEXT_SUFFIX = "\n\nContext:\n{context}"

# --- Helper Functions ---

def sanitize_filename_for_pinecone(filename: str) -> str:
    """
    Converts a filename into a valid Pinecone index name.
    Pinecone has restrictions on index names (e.g., lowercase, no special characters).
    """
    # Get the filename without its extension (e.g., "document.pdf" -> "document")
    name_without_ext = os.path.splitext(filename)[0]
    lower_name = name_without_ext.lower()
    return lower_name

def initialize_huggingface_components(api_key):
    """Initializes Hugging Face embeddings and the text-generation model."""

    embeddings = HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-mpnet-base-v2", huggingfacehub_api_token=api_key)
    repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
    llm = HuggingFaceEndpoint(
        repo_id=repo_id, task="text-generation", temperature=0.5,
        huggingfacehub_api_token=api_key, max_new_tokens=512
    )
    
    return embeddings, ChatHuggingFace(llm=llm)

def initialize_azure_openai_components(endpoint, api_key, deployment_name):
    """Initializes Azure OpenAI embeddings and chat model."""
    embeddings = AzureOpenAIEmbeddings(azure_deployment='text-embedding-3-small', api_key=api_key, azure_endpoint=endpoint)
    chat_model = AzureChatOpenAI(azure_deployment=deployment_name, temperature=0.5, api_key=api_key, api_version='2023-03-15-preview', azure_endpoint=endpoint)
    
    return embeddings, chat_model

def setup_pinecone(api_key, index_name, dimension):
    """Connects to a Pinecone index, creating it if it doesn't exist."""
    # Initialize the Pinecone client with the API key.
    pc = Pinecone(api_key=api_key)
    
    # Check if an index with the given name already exists.
    if index_name not in pc.list_indexes().names():
        # If it doesn't exist, inform the user and create it.
        st.info(f"First time seeing this document. Creating a new Pinecone index: '{index_name}'")
        pc.create_index(
            name=index_name, 
            dimension=dimension,  # The size of the vectors (e.g., 768 for all-mpnet-base-v2).
            metric="cosine",      # The metric used for similarity search (cosine is common for text).
            spec=ServerlessSpec(cloud="aws", region="us-east-1") # Specifies a serverless deployment.
        )
        # Wait a moment to ensure the index is ready before connecting.
        time.sleep(1)
        
    # Return a client object connected to the specified index.
    return pc.Index(index_name)

def process_and_store_pdf(uploaded_file, text_splitter, vector_store):
    """Loads a PDF, splits it into chunks, and stores the chunks in the vector store."""
    # Create a temporary local path to save the uploaded file.
    temp_file_path = f"./{uploaded_file.name}"
    
    # Write the contents of the uploaded file to the temporary path.
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    # Use PyPDFLoader to load the text content from the PDF.
    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()
    
    # Split the loaded documents into smaller text chunks using the provided text splitter.
    texts = text_splitter.split_documents(documents)
    
    # Generate a unique ID for each text chunk.
    uuids = [str(uuid.uuid4()) for _ in range(len(texts))]
    
    # Add the text chunks (and their generated embeddings) to the vector store.
    vector_store.add_documents(documents=texts, ids=uuids)
    
    # Clean up by removing the temporary file.
    os.remove(temp_file_path)
    
    # Return the number of chunks processed.
    return len(texts)

def ask_document(question, vector_store, chat_model, user_prompt):
    """
    Asks a question to the document using the RAG (Retrieval-Augmented Generation) pipeline.
    """
    # Create a retriever from the vector store to find relevant document chunks.
    retriever = vector_store.as_retriever()
    
    # 1. Retrieve: Find documents relevant to the user's question.
    retrieved_docs = retriever.invoke(question)
    
    # Combine the content of the retrieved documents into a single context string.
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # Construct the full system prompt by adding the retrieved context.
    full_prompt_template = user_prompt + CONTEXT_SUFFIX
    final_system_prompt = full_prompt_template.format(context=context)
    
    # Create the message list for the chat model.
    messages = [
        SystemMessage(content=final_system_prompt),  # The instruction and context.
        HumanMessage(content=question),             # The user's question.
    ]
    
    # 2. Generate: Get the final answer from the chat model.
    return chat_model.invoke(messages).content

# --- Streamlit App ---

st.title("Chat with Your Documents")

# --- Session State Initialization ---
# Session state is used to store variables that persist across user interactions.

# Initialize a list to store the chat history.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize a dictionary to store RAG components (vector store, chat model).
if "rag_components" not in st.session_state:
    st.session_state.rag_components = {}

# Initialize the user-editable system prompt.
if "user_prompt" not in st.session_state:
    st.session_state.user_prompt = DEFAULT_USER_PROMPT

# --- Sidebar UI ---
with st.sidebar:
    st.header("Settings")
    
    # A toggle switch to turn the RAG functionality on or off.
    rag_on = st.toggle("Enable RAG", value=True)
    
    if rag_on:
        # If RAG is on, show options for model provider and file upload.
        embedding_model_provider = st.selectbox("Select Embedding Model", ["Hugging Face", "Azure OpenAI"])
        uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
        
        st.header("Assistant Instructions")
        # Allow the user to edit the system prompt for the RAG assistant.
        st.session_state.user_prompt = st.text_area(
            "Edit the instructions for the RAG assistant",
            value=st.session_state.user_prompt,
            height=150
        )
    else:
        # If RAG is off, only show an option to select the LLM provider for a standard chat.
        llm_provider = st.selectbox("Select LLM Provider", ["Hugging Face", "Azure OpenAI"])

# --- Main App Logic for RAG ---
# This block runs only if RAG is enabled and a file has been uploaded.
if rag_on and uploaded_file is not None:
    # Sanitize the filename to use as the Pinecone index name.
    index_name = sanitize_filename_for_pinecone(uploaded_file.name)
    
    # Check if the current file is different from the one already processed.
    # This prevents re-initializing everything on every app rerun if the file is the same.
    if st.session_state.rag_components.get("index_name") != index_name:
        with st.spinner(f"Connecting to knowledge base for '{uploaded_file.name}'..."):
            try:
                # Initialize models and set vector dimension based on the selected provider.
                if embedding_model_provider == "Hugging Face":
                    embeddings, chat_model = initialize_huggingface_components(os.getenv("HUGGINGFACE_API_KEY"))
                    dimension = 768  # Dimension for 'all-mpnet-base-v2'
                else: # Azure OpenAI
                    embeddings, chat_model = initialize_azure_openai_components(os.getenv("AZURE_OPENAI_ENDPOINT"), os.getenv("AZURE_OPENAI_KEY"), os.getenv("DEPLOYMENT_NAME"))
                    dimension = 1536 # Dimension for 'text-embedding-3-small'
                
                # Check if the index already exists in Pinecone to determine if processing is needed.
                pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
                needs_processing = index_name not in pc.list_indexes().names()

                # Set up the Pinecone index (creates it if it's new).
                index = setup_pinecone(os.getenv("PINECONE_API_KEY"), index_name, dimension)
                # Initialize the LangChain vector store object.
                vector_store = PineconeVectorStore(index=index, embedding=embeddings)

                if needs_processing:
                    # If this is a new document, it needs to be processed and stored.
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                    with st.spinner(f"Processing '{uploaded_file.name}' for the first time..."):
                        num_chunks = process_and_store_pdf(uploaded_file, text_splitter, vector_store)
                    st.sidebar.success(f"Successfully added {num_chunks} sections to the knowledge base.")
                else:
                    # If the index already exists, just connect to it.
                    st.sidebar.success(f"Connected to existing knowledge base for '{uploaded_file.name}'.")
                
                # Store the initialized components in the session state for later use.
                st.session_state.rag_components = {
                    "vector_store": vector_store,
                    "chat_model": chat_model,
                    "index_name": index_name
                }
            except Exception as e:
                # Handle any errors during initialization.
                st.error(f"An error occurred: {e}")
                st.session_state.rag_components = {}

# --- Chat History Display ---
# Display previous messages from the chat history.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User Input and Response Generation ---
# Get new user input from the chat input box.
if prompt := st.chat_input("Ask your question here..."):
    # Add the user's message to the chat history.
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display the user's message in the chat.
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display the assistant's response.
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                if rag_on:
                    # If RAG is enabled...
                    if st.session_state.rag_components:
                        # Get the RAG components from session state.
                        vs = st.session_state.rag_components["vector_store"]
                        cm = st.session_state.rag_components["chat_model"]
                        # Generate an answer using the RAG pipeline.
                        answer = ask_document(prompt, vs, cm, st.session_state.user_prompt)
                    else:
                        # Handle the case where a document hasn't been uploaded yet.
                        answer = "Please upload a document to begin the RAG chat."
                else:
                    # If RAG is disabled, use the selected LLM as a standard chatbot.
                    if llm_provider == "Hugging Face":
                        _, chat_model = initialize_huggingface_components(os.getenv("HUGGINGFACE_API_KEY"))
                    else: # Azure OpenAI
                        _, chat_model = initialize_azure_openai_components(os.getenv("AZURE_OPENAI_ENDPOINT"), os.getenv("AZURE_OPENAI_KEY"), os.getenv("DEPLOYMENT_NAME"))
                    
                    # Get a direct response from the chat model without any context.
                    answer = chat_model.invoke([HumanMessage(content=prompt)]).content
                
                # Display the generated answer.
                st.markdown(answer)
                # Add the assistant's response to the chat history.
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"An error occurred: {e}")

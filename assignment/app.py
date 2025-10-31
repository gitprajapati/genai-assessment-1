# Import necessary libraries
import os
import uuid
import time
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEndpointEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables from the .env file
load_dotenv()

DEFAULT_USER_PROMPT = "You are a helpful assistant. Your task is to answer the user's question based only on the following context. If the answer is not in the context, say 'I do not have enough information to answer that question.'"
CONTEXT_SUFFIX = "\n\nContext:\n{context}"

# --- Helper Functions ---

def sanitize_filename_for_pinecone(filename: str) -> str:
    """Converts a filename into a valid Pinecone index name."""
    name_without_ext = os.path.splitext(filename)[0]
    lower_name = name_without_ext.lower()
    return lower_name

def initialize_huggingface_components(api_key):
    """Initializes Hugging Face embeddings and the correct text-generation model."""
    embeddings = HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-mpnet-base-v2", huggingfacehub_api_token=api_key)
    repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
    llm = HuggingFaceEndpoint(
        repo_id=repo_id, task="text-generation", temperature=0.5,
        huggingfacehub_api_token=api_key, max_new_tokens=512
    )
    return embeddings, ChatHuggingFace(llm=llm)

def initialize_azure_openai_components(endpoint, api_key, deployment_name):
    embeddings = AzureOpenAIEmbeddings(azure_deployment='text-embedding-3-small', api_key=api_key, azure_endpoint=endpoint)
    chat_model = AzureChatOpenAI(azure_deployment=deployment_name, temperature=0.5, api_key=api_key, api_version='2023-03-15-preview', azure_endpoint=endpoint)
    return embeddings, chat_model

def setup_pinecone(api_key, index_name, dimension):
    """Connects to a Pinecone index, creating it if it doesn't exist."""
    pc = Pinecone(api_key=api_key)
    if index_name not in pc.list_indexes().names():
        st.info(f"First time seeing this document. Creating a new Pinecone index: '{index_name}'")
        pc.create_index(name=index_name, dimension=dimension, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))
        time.sleep(1)
    return pc.Index(index_name)

def process_and_store_pdf(uploaded_file, text_splitter, vector_store):
    """Loads, splits, and stores a PDF in the vector store."""
    temp_file_path = f"./{uploaded_file.name}"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()
    texts = text_splitter.split_documents(documents)
    uuids = [str(uuid.uuid4()) for _ in range(len(texts))]
    vector_store.add_documents(documents=texts, ids=uuids)
    os.remove(temp_file_path)
    return len(texts)

def ask_document(question, vector_store, chat_model, user_prompt):
    """Asks a question to the document using the RAG pipeline."""
    retriever = vector_store.as_retriever()
    retrieved_docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    full_prompt_template = user_prompt + CONTEXT_SUFFIX
    final_system_prompt = full_prompt_template.format(context=context)
    
    messages = [
        SystemMessage(content=final_system_prompt),
        HumanMessage(content=question),
    ]
    return chat_model.invoke(messages).content

# --- Streamlit App ---

st.title("Chat with Your Documents")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_components" not in st.session_state:
    st.session_state.rag_components = {}
if "user_prompt" not in st.session_state:
    st.session_state.user_prompt = DEFAULT_USER_PROMPT

with st.sidebar:
    st.header("Settings")
    rag_on = st.toggle("Enable RAG", value=True)
    
    if rag_on:
        embedding_model_provider = st.selectbox("Select Embedding Model", ["Hugging Face", "Azure OpenAI"])
        uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
        
        st.header("Assistant Instructions")
        st.session_state.user_prompt = st.text_area(
            "Edit the instructions for the RAG assistant",
            value=st.session_state.user_prompt,
            height=150
        )
        
    else:
        llm_provider = st.selectbox("Select LLM Provider", ["Hugging Face", "Azure OpenAI"])

if rag_on and uploaded_file is not None:
    index_name = sanitize_filename_for_pinecone(uploaded_file.name)
    if st.session_state.rag_components.get("index_name") != index_name:
        with st.spinner(f"Connecting to knowledge base for '{uploaded_file.name}'..."):
            try:
                if embedding_model_provider == "Hugging Face":
                    embeddings, chat_model = initialize_huggingface_components(os.getenv("HUGGINGFACE_API_KEY"))
                    dimension = 768
                else:
                    embeddings, chat_model = initialize_azure_openai_components(os.getenv("AZURE_OPENAI_ENDPOINT"), os.getenv("AZURE_OPENAI_KEY"), os.getenv("DEPLOYMENT_NAME"))
                    dimension = 1536

                pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
                needs_processing = index_name not in pc.list_indexes().names()

                index = setup_pinecone(os.getenv("PINECONE_API_KEY"), index_name, dimension)
                vector_store = PineconeVectorStore(index=index, embedding=embeddings)

                if needs_processing:
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                    with st.spinner(f"Processing '{uploaded_file.name}' for the first time..."):
                        num_chunks = process_and_store_pdf(uploaded_file, text_splitter, vector_store)
                    st.sidebar.success(f"Successfully added {num_chunks} sections to the knowledge base.")
                else:
                    st.sidebar.success(f"Connected to existing knowledge base for '{uploaded_file.name}'.")
                
                st.session_state.rag_components = {
                    "vector_store": vector_store,
                    "chat_model": chat_model,
                    "index_name": index_name
                }
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.session_state.rag_components = {}

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                if rag_on:
                    if st.session_state.rag_components:
                        vs = st.session_state.rag_components["vector_store"]
                        cm = st.session_state.rag_components["chat_model"]
                        answer = ask_document(prompt, vs, cm, st.session_state.user_prompt)
                    else:
                        answer = "Please upload a document to begin the RAG chat."
                else:
                    if llm_provider == "Hugging Face":
                        _, chat_model = initialize_huggingface_components(os.getenv("HUGGINGFACE_API_KEY"))
                    else:
                        _, chat_model = initialize_azure_openai_components(os.getenv("AZURE_OPENAI_ENDPOINT"), os.getenv("AZURE_OPENAI_KEY"), os.getenv("DEPLOYMENT_NAME"))
                    answer = chat_model.invoke([HumanMessage(content=prompt)]).content
                
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"An error occurred: {e}")
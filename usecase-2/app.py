import os
import re
from dotenv import load_dotenv
from serpapi import GoogleSearch
from pinecone import Pinecone, ServerlessSpec

from flask import Flask, request, jsonify, render_template

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from pydantic import BaseModel, Field

# --- 1. INITIALIZATION & SETUP ---
# This section handles loading configurations, initializing all external services (Flask, OpenAI, Pinecone),
# and ensuring the environment is ready.

# Load environment variables from the .env file.
# This keeps secret keys out of the code.
load_dotenv()

# Check that all necessary API keys have been set in the .env file.
# The application will exit with an error if any are missing.
api_keys = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "SERPAPI_API_KEY", "PINECONE_API_KEY"]
if not all(os.getenv(key) for key in api_keys):
    raise ValueError("One or more required API keys are missing from the .env file.")

# Define a constant for the Pinecone index name to avoid typos.
PINECONE_INDEX_NAME = "career-guides"

# Initialize the Flask web application.
app = Flask(__name__)

# Initialize the Azure OpenAI Chat Model (LLM) for generating text.
# We set temperature=0 for more deterministic and factual responses.
model = AzureChatOpenAI(
    azure_deployment='gpt-4o',
    api_version="2024-08-01-preview",
    temperature=0,
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)
# Initialize the Azure OpenAI Embeddings Model.
# This model converts text into numerical vectors for semantic search in Pinecone.
embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-3-small", # Make sure this is your exact deployment name in Azure AI Studio
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# Initialize the Pinecone client with the API key.
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Check if the Pinecone index already exists. If not, create it.
# This prevents errors on subsequent runs of the application.
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    print(f"Creating Pinecone index: {PINECONE_INDEX_NAME}...")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,  # The dimension must match the output of the embeddings model (1536 for text-embedding-3-small)
        metric="cosine", # Cosine similarity is effective for semantic text comparison.
        spec=ServerlessSpec(cloud='aws', region='us-east-1') # Use a serverless spec for cost-effectiveness.
    )
    print("Index created successfully.")

# Get a handle to the specific Pinecone index we'll be working with.
pinecone_index = pc.Index(PINECONE_INDEX_NAME)
# Create a LangChain vector store object, which simplifies interacting with the Pinecone index.
vector_store = PineconeVectorStore(index=pinecone_index, embedding=embeddings, text_key='text')

# --- 2. LANGCHAIN LOGIC & HELPER FUNCTIONS ---
# This section contains the core AI logic, broken down into reusable functions.

# Define a Pydantic model to enforce a structured JSON output from the LLM.
# This ensures we reliably get the user's role and required skills.
class CareerResponse(BaseModel):
    user_role: str = Field(..., description="The user's desired future job role")
    skills: list[str] = Field(..., description="A list of key skills required for that role")

def normalize_text(text: str) -> str:
    """A helper function to create a consistent, URL-safe ID for Pinecone vectors from a text string."""
    return re.sub(r'\s+', '-', text).lower()

def extract_career_goal(user_input: str) -> CareerResponse:
    """
    Uses the LLM with structured output to analyze the user's query and extract
    their target job role and the skills they are interested in.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert career counselor. Analyze the user's request and identify their desired future job role and the key skills required for it."),
        ("user", "{input}")
    ])
    # The .with_structured_output method forces the LLM's response into the CareerResponse schema.
    structured_chain = prompt | model.with_structured_output(CareerResponse)
    return structured_chain.invoke({"input": user_input})

def generate_roadmap(role: str, skills: list) -> str:
    """Generates a personalized, step-by-step learning roadmap using the LLM."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a world-class career coach. Create a detailed, step-by-step roadmap for a user trying to transition into a new role. The roadmap must be practical, actionable, and provide clear guidance."),
        ("user", "Please provide a step-by-step learning roadmap for me to become a {user_role}. I already know Python, and I need to learn the following skills: {skills}. Start with the most fundamental skills first and suggest key topics for each.")
    ])
    chain = prompt | model | StrOutputParser() # StrOutputParser extracts just the text content from the LLM's response.
    return chain.invoke({"user_role": role, "skills": ", ".join(skills)})

def get_market_insights(role: str) -> str:
    """
    Fetches real-time news about a job role via the SerpApi Google News engine
    and then uses the LLM to summarize the findings into a market analysis.
    """
    print(f"Searching for market news on '{role}'...")
    try:
        # Define the search parameters for the SerpApi Google News API.
        params = {
            "engine": "google_news",
            "q": f"'{role}' job market trends skills demand salary",
            "gl": "us", "hl": "en",
            "api_key": os.getenv("SERPAPI_API_KEY")
        }
        # Execute the search and get the results as a dictionary.
        results = GoogleSearch(params).get_dict()
        news_items = results.get("news_results", [])[:10] # Take the top 10 articles.

        if not news_items:
            return "No recent market news could be found for this role."

        # Format the news articles into a single string to pass to the LLM.
        formatted_articles = "".join(
            f"Article {i+1}: {item.get('title')}\nSnippet: {item.get('snippet')}\n\n"
            for i, item in enumerate(news_items)
        )
        
        # Create a specific prompt for the summarization task.
        summarization_prompt = ChatPromptTemplate.from_template(
            """You are a senior market analyst. Based on the following news articles, generate a concise summary of the current market trends for a **{user_role}**.
            Focus on overall demand, job outlook, emerging skills, and general sentiment.
            
            News Articles:\n{news_articles}
            
            Provide your analysis as a brief, insightful summary."""
        )
        chain = summarization_prompt | model | StrOutputParser()
        return chain.invoke({"user_role": role, "news_articles": formatted_articles})
    except Exception as e:
        print(f"Error fetching/summarizing news: {e}")
        return "There was an error retrieving market insights."

# --- 3. FLASK API ROUTES ---
# This section defines the web endpoints that the user's browser will interact with.

@app.route("/")
def index():
    """Serves the main chat page (index.html)."""
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    """
    This is the main endpoint that orchestrates the entire AI response workflow.
    It receives a user's message, processes it, and returns the AI's guidance.
    """
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    try:
        # Step 1: Extract the user's career goal from their raw input.
        goal = extract_career_goal(user_input)
        role_id = normalize_text(goal.user_role)
        
        # Step 2: Check Pinecone for a cached guide. This saves API calls and provides a faster response.
        print(f"Checking cache in Pinecone for role_id: {role_id}")
        fetched_vectors = pinecone_index.fetch(ids=[role_id])
       
        # Check if the vector for the given role_id exists in the response.
        if fetched_vectors.vectors and role_id in fetched_vectors.vectors:
            print("Cache hit! Retrieving from Pinecone.")
            # If it exists, retrieve the full text from the metadata and return it immediately.
            cached_response = fetched_vectors.vectors[role_id].metadata['text']
            return jsonify({"response": cached_response})

        # Step 3: (CACHE MISS) If no guide is found, generate a new one from scratch.
        print("Cache miss. Generating new career guide...")
        roadmap = generate_roadmap(goal.user_role, goal.skills)
        insights = get_market_insights(goal.user_role)
        
        # Combine all generated parts into a single, well-formatted response document.
        full_response = (
            f"### Your Personalized Career Guide to Becoming a {goal.user_role}\n\n"
            f"**Skills to Focus On:** {', '.join(goal.skills)}\n\n"
            "--- \n\n"
            "### Step-by-Step Learning Roadmap\n\n"
            f"{roadmap}\n\n"
            "--- \n\n"
            "### Current Market Insights\n\n"
            f"{insights}"
        )

        # Step 4: Store the newly generated guide in Pinecone for future requests.
        print(f"Storing new guide in Pinecone with id: {role_id}")
        vector_store.add_texts(
            texts=[full_response],
            # We store the role name in metadata for potential filtering later.
            metadatas=[{"role": goal.user_role}],
            # We use the normalized role_id as the unique identifier for the vector.
            ids=[role_id]
        )
        
        # Return the newly generated response to the user.
        return jsonify({"response": full_response})

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc() 
        return jsonify({"error": "An internal error occurred. Please try again."}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5001)

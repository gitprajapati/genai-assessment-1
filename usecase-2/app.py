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

# Load environment variables
load_dotenv()

# Check for necessary API keys
api_keys = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "SERPAPI_API_KEY", "PINECONE_API_KEY"]
if not all(os.getenv(key) for key in api_keys):
    raise ValueError("One or more required API keys are missing from the .env file.")

PINECONE_INDEX_NAME = "career-guides"

# Initialize Flask App
app = Flask(__name__)

# Initialize LLM and Embeddings Model
model = AzureChatOpenAI(
    azure_deployment='gpt-4o',
    api_version="2024-08-01-preview",
    temperature=0,
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)
embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-3-small",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Create Pinecone index if it doesn't exist
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    print(f"Creating Pinecone index: {PINECONE_INDEX_NAME}...")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,  # Dimension for text-embedding-ada-002
        metric="cosine",
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )
    print("Index created successfully.")

pinecone_index = pc.Index(PINECONE_INDEX_NAME)
vector_store = PineconeVectorStore(index=pinecone_index, embedding=embeddings, text_key='text')

# --- 2. LANGCHAIN LOGIC & HELPER FUNCTIONS ---

# Pydantic model for structured output
class CareerResponse(BaseModel):
    user_role: str = Field(..., description="The user's desired future job role")
    skills: list[str] = Field(..., description="A list of key skills required for that role")

def normalize_text(text):
    """A helper function to create a consistent ID for Pinecone."""
    return re.sub(r'\s+', '-', text).lower()

def extract_career_goal(user_input: str) -> CareerResponse:
    """Uses LLM to extract the user's target role and skills."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert career counselor. Analyze the user's request and identify their desired future job role and the key skills required for it."),
        ("user", "{input}")
    ])
    structured_chain = prompt | model.with_structured_output(CareerResponse)
    return structured_chain.invoke({"input": user_input})

def generate_roadmap(role: str, skills: list) -> str:
    """Generates a personalized learning roadmap."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a world-class career coach. Create a detailed, step-by-step roadmap for a user trying to transition into a new role. The roadmap must be practical, actionable, and provide clear guidance."),
        ("user", "Please provide a step-by-step learning roadmap for me to become a {user_role}. I already know Python, and I need to learn the following skills: {skills}. Start with the most fundamental skills first and suggest key topics for each.")
    ])
    chain = prompt | model | StrOutputParser()
    return chain.invoke({"user_role": role, "skills": ", ".join(skills)})

def get_market_insights(role: str) -> str:
    """Fetches and summarizes market news for a given role."""
    print(f"Searching for market news on '{role}'...")
    try:
        params = {
            "engine": "google_news",
            "q": f"'{role}' job market trends skills demand salary",
            "gl": "us", "hl": "en",
            "api_key": os.getenv("SERPAPI_API_KEY")
        }
        results = GoogleSearch(params).get_dict()
        news_items = results.get("news_results", [])[:5]

        if not news_items:
            return "No recent market news could be found for this role."

        formatted_articles = "".join(
            f"Article {i+1}: {item.get('title')}\nSnippet: {item.get('snippet')}\n\n"
            for i, item in enumerate(news_items)
        )
        
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

@app.route("/")
def index():
    """Serves the main chat page."""
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    """Main endpoint to handle user messages and orchestrate the AI response."""
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    try:
        # Part 1: Extract the core career goal
        goal = extract_career_goal(user_input)
        role_id = normalize_text(goal.user_role)
        
        # Part 2: Check Pinecone for a cached response
        print(f"Checking cache in Pinecone for role_id: {role_id}")
        fetched_vectors = pinecone_index.fetch(ids=[role_id])
        
        # --- FIX IS HERE ---
        # Access the response using dot notation for the modern pinecone-client
        if fetched_vectors.vectors and role_id in fetched_vectors.vectors:
            print("Cache hit! Retrieving from Pinecone.")
            # Also access metadata using dot notation
            cached_response = fetched_vectors.vectors[role_id].metadata['text']
            return jsonify({"response": cached_response})

        # Part 3: If not cached, generate the full response
        print("Cache miss. Generating new career guide...")
        roadmap = generate_roadmap(goal.user_role, goal.skills)
        insights = get_market_insights(goal.user_role)
        
        # Combine into a single response document
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

        # Part 4: Store the new response in Pinecone
        print(f"Storing new guide in Pinecone with id: {role_id}")
        vector_store.add_texts(
            texts=[full_response],
            metadatas=[{"role": goal.user_role}],
            ids=[role_id]
        )
        
        return jsonify({"response": full_response})

    except Exception as e:
        print(f"An error occurred: {e}")
        # Add this to get more detailed error logs in your terminal
        import traceback
        traceback.print_exc()
        return jsonify({"error": "An internal error occurred. Please try again."}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5001)
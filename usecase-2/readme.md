# AI Career Advisor Chatbot

An intelligent, conversational AI assistant designed to provide personalized, data-driven career guidance. This chatbot helps users explore new career paths, generates custom learning roadmaps, and provides real-time insights into job market trends.


*(Replace with a screenshot of your application)*

## 📖 Table of Contents
- [Problem Statement](#-problem-statement)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Technology Stack](#-technology-stack)
- [Setup and Installation](#-setup-and-installation)
- [Running the Application](#-running-the-application)
- [Project Structure](#-project-structure)
- [Future Enhancements](#-future-enhancements)

## 🎯 Problem Statement
Students and professionals often lack personalized, data-driven career guidance. Traditional counseling methods struggle to keep pace with dynamic market trends, evolving skill demands, and the rapid emergence of new job roles. This project aims to bridge that gap with a 24/7 AI-powered career mentor.

## ✨ Key Features
- **Conversational Interface:** Engage in a natural conversation to define your career aspirations.
- **Personalized Roadmaps:** Receive a detailed, step-by-step learning plan tailored to your target role and existing skills.
- **Live Market Analysis:** Get an AI-generated summary of the latest job market trends, in-demand skills, and salary expectations for your desired role, powered by real-time news data.
- **Intelligent Caching:** Utilizes a Pinecone vector database to cache generated career guides, ensuring near-instantaneous responses for previously requested roles.
- **Scalable Architecture:** Built with modern tools to be extensible and robust.

## 🏗️ System Architecture
The application follows a modular architecture where the user interface is decoupled from the core logic, which in turn leverages specialized engines for reasoning and data retrieval.

```
                ┌──────────────────────────────────────────┐
                │            User Interface                │
                │         (Flask Web Application)          │
                └──────────────────────────────────────────┘
                                    │
                                    ▼
                ┌──────────────────────────────────────────┐
                │      LangChain Conversation Engine       │
                │  • Role & Skill Extraction (Pydantic)    │
                │  • LLMChain (Azure GPT-4o)               │
                │  • Caching Logic                         │
                └──────────────────────────────────────────┘
                                        │
                            ┌───────────┴───────────┐
                            ▼                       ▼
            ┌───────────────────────────┐     ┌──────────────────────────┐
            │     Pinecone Vector DB    │     │   Market Insight Engine  │
            │ • Stores/retrieves guides │     │  • SerpApi (Google News) │
            │ • Semantic Caching        │     │  • Summarization Chain   │
            └───────────────────────────┘     └──────────────────────────┘
```

**Data Flow:**
1. A user sends a career goal query via the Flask web UI.
2. The LangChain engine extracts the target `user_role`.
3. The system checks Pinecone for a cached guide for this role.
4. **Cache Hit:** The pre-generated guide is retrieved instantly from Pinecone and sent to the user.
5. **Cache Miss:**
   - The system generates a new learning roadmap.
   - The Market Insight Engine fetches and summarizes the latest news.
   - The complete guide is assembled.
   - The new guide is stored in Pinecone for future requests.
   - The response is sent to the user.

## 💻 Technology Stack
- **Backend Framework:** **Flask**
- **LLM Orchestration:** **LangChain**
- **LLM & Embeddings:** **Azure OpenAI** (GPT-4o, text-embedding-3-small)
- **Vector Database:** **Pinecone**
- **Real-time News API:** **SerpApi** (for Google News)
- **Environment Management:** `python-dotenv`

## 🚀 Setup and Installation

Follow these steps to get the application running on your local machine.

### Prerequisites
- Python 3.10 or higher
- An active virtual environment (recommended)
- API keys for Azure OpenAI, Pinecone, and SerpApi

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/ai-career-advisor.git
cd ai-career-advisor```

### Step 2: Create and Activate a Virtual Environment
- **On Windows:**
  ```bash
  python -m venv venv
  .\venv\Scripts\activate
  ```
- **On macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### Step 3: Install Dependencies
Install all the required libraries from the `requirements.txt` file.
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
1.  Create a file named `.env` in the root of the project directory.
2.  Copy the content from `.env.example` (or the block below) into your new `.env` file.
3.  Replace the placeholder values with your actual API keys and endpoints.

**File: `.env`**
```
# Azure OpenAI Credentials
# Find these in the "Keys and Endpoint" section of your Azure OpenAI resource
AZURE_OPENAI_API_KEY="YOUR_AZURE_OPENAI_API_KEY"
AZURE_OPENAI_ENDPOINT="https://YOUR_AZURE_RESOURCE_NAME.openai.azure.com/"

# SerpApi Key for Google News (Market Insight Engine)
# Get this from your SerpApi dashboard
SERPAPI_API_KEY="YOUR_SERPAPI_API_KEY"

# Pinecone API Key for Vector Database
# Get this from your Pinecone dashboard under "API Keys"
PINECONE_API_KEY="YOUR_PINECONE_API_KEY"
```**Important:** Ensure your `.gitignore` file includes `.env` to prevent accidentally committing your secret keys.

## ▶️ Running the Application

1.  **Start the Flask Backend Server:**
    ```bash
    python app.py
    ```
    The server will start, typically on `http://127.0.0.1:5001`.

2.  **Access the Chatbot:**
    Open your web browser and navigate to the address provided in the terminal:
    [http://127.0.0.1:5001](http://127.0.0.1:5001)

You can now start chatting with your AI Career Advisor!

## 📂 Project Structure
```
.
├── .env                  # Stores all secret API keys and endpoints
├── app.py                # Main Flask application, contains all backend logic
├── requirements.txt      # List of Python dependencies for the project
├── README.md             # This file
└── templates/
    └── index.html        # The HTML/CSS/JS frontend for the chat interface
```

---
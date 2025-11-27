# weather-rag-langgraph
🌦️📚 LangGraph Weather + PDF RAG Pipeline
AI Engineer Assignment — Complete Implementation

This project demonstrates a complete AI pipeline built using LangChain, LangGraph, LangSmith, Qdrant, and Streamlit.
The system performs two intelligent tasks:

Fetch real-time weather data using the OpenWeatherMap API

Answer questions from a PDF using RAG (Retrieval-Augmented Generation)

A LangGraph decision node chooses which branch to execute based on the user query.

🚀 Features
1. LangGraph Agentic Pipeline

A graph-based pipeline built using LangGraph

Decision node detects whether the user is asking about:

Weather

PDF content (RAG)

Routes the query to the correct node

2. Weather API Integration

Uses OpenWeatherMap for real-time weather

Handles rate limits, retries, API errors

Returns structured data to the LLM

3. PDF RAG System

PDF loading with pypdf

Text chunking using LangChain splitters

Embeddings generated using Azure/OpenAI

Stored in Qdrant vector DB

Similarity search + summarization using LLM

4. LLM Processing

Unified wrapper for all LLM usage

Includes summarization helper

Clean modular LangChain implementation

5. LangSmith Evaluation

Pipeline fully instrumented with LangSmith

All graph runs traceable

Response quality evaluated

Includes screenshots + logs (below)

6. Streamlit Chat UI

Simple interactive chat app

Allows switching between weather + RAG

Perfect for demoing the pipeline

7. Unit Tests

Covers:

Weather API handling

RAG fetching + retrieval

Decision node logic

Ensures >80% coverage

📁 Project Structure
langgraph-weather-rag/
├─ src/
│  ├─ __init__.py
│  ├─ pipeline.py                 
│  ├─ weather.py                  
│  ├─ pdf_rag.py                  
│  ├─ llm_utils.py                
│  ├─ qdrant_utils.py             
│  ├─ langsmith_eval.py           
│  └─ streamlit_app.py            
├─ tests/
│  ├─ test_weather.py
│  ├─ test_pdf_rag.py
│  └─ test_pipeline_decision.py
├─ sample_data/
│  └─ sample_doc.pdf
├─ requirements.txt
└─ README.md

🔧 Installation
1. Clone the Repo
git clone https://github.com/<your-username>/langgraph-weather-rag.git
cd langgraph-weather-rag

2. Create Virtual Environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt

🔑 Environment Variables

Create a .env file:

OPENWEATHER_API_KEY=your_key
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_key_if_cloud
AZURE_OPENAI_ENDPOINT=https://xxx.openai.azure.com/
AZURE_OPENAI_API_KEY=xxxx
AZURE_OPENAI_API_VERSION=2024-02-01
LANGSMITH_API_KEY=xxxx
LANGSMITH_TRACING=true

▶️ Running the Pipeline
Start Streamlit UI
streamlit run src/streamlit_app.py

🎥 Streamlit Demo Video

📌 Loom demo link (replace with your actual link):
👉 https://www.loom.com/share/your-demo-link

🖼️ Streamlit UI Screenshots

(Add your actual screenshots here)

Example placeholders:

Chat Interface

Weather Response

RAG Response

📊 LangSmith Evaluation

This project supports LangSmith tracing + evaluation.

Example Traces Screenshot

(Replace with real screenshots)

Evaluation Result Screenshot

🧪 Running Unit Tests

Run all tests:

pytest -vv


Check coverage:

pytest --cov=src --cov-report=term-missing


Expected: 80–90% coverage

🧱 Qdrant Setup
Option 1: Local Docker
docker run -p 6333:6333 qdrant/qdrant

Option 2: Cloud Qdrant
Sign up at qdrant.tech and update env vars accordingly.

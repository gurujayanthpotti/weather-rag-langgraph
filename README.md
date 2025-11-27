# 🌦️📚 LangGraph Weather + PDF RAG Pipeline  
## AI Engineer Assignment — Complete Implementation

This project demonstrates a complete **AI pipeline** built using **LangChain**, **LangGraph**, **LangSmith**, **Qdrant**, and **Streamlit**. The system performs two intelligent tasks:

1. **Fetch real-time weather data** using the OpenWeatherMap API  
2. **Answer questions from a PDF** using RAG (Retrieval-Augmented Generation)  

A **LangGraph decision node** chooses which branch to execute based on the user query.

---

## 🚀 Features

### **LangGraph Agentic Pipeline**
- A graph-based pipeline built using LangGraph  
- Decision node detects whether the user is asking about:
  - **Weather**
  - **PDF content (RAG)**
- Routes the query to the correct node

### **Weather API Integration**
- Uses OpenWeatherMap for real-time weather  
- Handles rate limits, retries, and API errors  
- Returns structured data to the LLM

### **PDF RAG System**
- PDF loading with `pypdf`  
- Text chunking using LangChain splitters  
- Embeddings generated using Azure/OpenAI  
- Stored in **Qdrant vector DB**  
- Similarity search + summarization using LLM

### **LLM Processing**
- Unified wrapper for all LLM usage  
- Includes summarization helper  
- Clean modular LangChain implementation

### **LangSmith Evaluation**
- Pipeline fully instrumented with LangSmith  
- All graph runs traceable  
- Response quality evaluated  
- Includes screenshots + logs (below)

### **Streamlit Chat UI**
- Simple interactive chat app  
- Allows switching between Weather + RAG  
- Perfect for demoing the pipeline

### **Unit Tests**
Covers:
- Weather API handling  
- RAG fetching + retrieval  
- Decision node logic  
- Ensures >80% coverage

---

## 📁 Project Structure

langgraph-weather-rag/
├─ src/
│ ├─ init.py
│ ├─ pipeline.py
│ ├─ weather.py
│ ├─ pdf_rag.py
│ ├─ llm_utils.py
│ ├─ qdrant_utils.py
│ ├─ langsmith_eval.py
│ └─ streamlit_app.py
├─ tests/
│ ├─ test_weather.py
│ ├─ test_pdf_rag.py
│ └─ test_pipeline_decision.py
├─ sample_data/
│ └─ sample_doc.pdf
├─ requirements.txt
└─ README.md

---

## 🔧 Installation

### 1. Clone the Repo

git clone https://github.com/<your-username>/langgraph-weather-rag.git
cd langgraph-weather-rag 

###2. Create Virtual Environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

###3. Install Dependencies
pip install -r requirements.txt


###Running the Pipeline
####Start Streamlit UI
streamlit run src/streamlit_app.py

###🎥 Streamlit Demo Video
###📌 Loom demo link (replace with your actual link):
https://www.loom.com/share/04c805ac18944ab7859fd7ad80dc7cfb

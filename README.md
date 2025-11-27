# 🌦️📚 LangGraph Weather + PDF RAG Pipeline  
## AI Engineer Assignment — Complete Implementation

This project demonstrates a complete **AI pipeline** built using **LangChain**, **LangGraph**, **LangSmith**, **Qdrant**, and **Streamlit**. The system performs two intelligent tasks:

1. **Fetch real-time weather data** using the OpenWeatherMap API  
2. **Answer questions from a PDF** using RAG (Retrieval-Augmented Generation)  

A **LangGraph decision node** chooses which branch to execute based on the user query.

## 🔗 Live Deployment & Demo

- **Deployed Streamlit App:** [Open in Browser](https://weather-rag-langgraph-nxjmfm7guy2ethwstdy4eh.streamlit.app/)  
- **Demo Video (Loom):** [Watch Demo](https://www.loom.com/share/04c805ac18944ab7859fd7ad80dc7cfb)

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

## 🔧 Installation

### 1. Clone the Repo

git clone https://github.com/<your-username>/langgraph-weather-rag.git
cd langgraph-weather-rag 

### 2. Create Virtual Environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

### 3. Install Dependencies
pip install -r requirements.txt

--- 

### Running the Pipeline
### Start Streamlit UI
streamlit run src/streamlit_app.py

---
### LangSmith logs/screenshots

<img width="1100" height="558" alt="image" src="https://github.com/user-attachments/assets/f885b0fd-43e5-4cfd-a71b-1e8e9fadefe3" />

<img width="1100" height="561" alt="image" src="https://github.com/user-attachments/assets/4685524b-f0ab-418a-a494-63c25008563b" />

 ###Comparing results

 <img width="1100" height="561" alt="image" src="https://github.com/user-attachments/assets/f23eb0c5-6bef-4683-8162-371ba46c6d32" />

 ### Track results over time
<img width="1100" height="562" alt="image" src="https://github.com/user-attachments/assets/fca2d20e-ae53-491c-970a-9b6262160d1e" />

---

### Streamlit app screenshots
<img width="2859" height="1524" alt="image" src="https://github.com/user-attachments/assets/b6eb07b7-3e6c-44b3-93ef-70d004395655" />
### Response from chatbot
<img width="2879" height="1693" alt="image" src="https://github.com/user-attachments/assets/db6c4e90-cd9a-4a55-8a1d-8cfe03023c68" />




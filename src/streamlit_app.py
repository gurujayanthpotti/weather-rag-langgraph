import os
import time
import streamlit as st
from dotenv import load_dotenv
from graph import build_pipeline_graph
from pdf_rag import build_embeddings_and_upsert

# ------------------- Load .env and Setup LangSmith -------------------
load_dotenv()  # Load environment variables first

# Enable LangSmith tracing
os.environ["LANGSMITH_TRACING"] = "true"

# Set API key from environment
LS_API_KEY = os.getenv("LANGSMITH_API_KEY")
if not LS_API_KEY:
    st.warning("LangSmith API key not found. Tracing and ingestion will be disabled.")
else:
    os.environ["LANGSMITH_API_KEY"] = LS_API_KEY

# ------------------- Streamlit Page Setup -------------------
st.set_page_config(page_title="Weather + PDF RAG (LangGraph)", layout="wide")

st.markdown("""
<style>
.user-bubble { background: #c9f7ff; padding:1rem; border-radius:12px; margin-bottom:0.5rem; font-size:1.05rem; color:#000; }
.bot-bubble { background: #fff3d8; padding:1rem; border-radius:12px; margin-bottom:1rem; font-size:1.05rem; color:#333; }
</style>
""", unsafe_allow_html=True)

st.title(" Weather + PDF RAG (LangGraph) Chatbot")

# ------------------- Session State -------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "graph" not in st.session_state:
    st.session_state.graph = build_pipeline_graph(
        openweather_key=os.getenv("OPENWEATHER_API_KEY")
    )

if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False

if "allow_question" not in st.session_state:
    st.session_state.allow_question = True

# ------------------- Reset Chat -------------------
if st.button("Reset Conversation", key="reset", help="Clear chat history"):
    st.session_state.history = []
    st.session_state.allow_question = True
    st.rerun()

# ------------------- PDF Upload -------------------
st.markdown("### Upload PDF to Re-Embed Into Vector DB")

if st.session_state.pdf_uploaded:
    if st.button(" Upload New PDF"):
        st.session_state.pdf_uploaded = False
        st.session_state.pdf_ready_for_new_upload = True
        st.rerun()

uploaded = None
if not st.session_state.pdf_uploaded:
    uploaded = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded and not st.session_state.pdf_uploaded:
    
    count = build_embeddings_and_upsert(uploaded)

    st.toast(f" PDF embedded successfully! ({count} chunks)", icon="🎉")
    st.success(" PDF upload completed!")
    st.info(" Refreshing page shortly…")

    st.session_state.pdf_uploaded = True
    st.session_state.pdf_ready_for_new_upload = True

    countdown = st.empty()
    for i in range(3, 0, -1):
        countdown.info(f"Auto-refreshing in **{i}** seconds...")
        time.sleep(1)
    countdown.empty()
    st.rerun()
# ------------------- Chat UI -------------------
st.markdown("###  Ask a Question")

if st.session_state.allow_question:
    user_input = st.text_input("Ask here...", key="input_text")
else:
    user_input = None
    st.info("Ask another question when you're ready.")

if st.session_state.allow_question:
    if st.button("Send", key="send_btn") and user_input.strip():
        with st.spinner("Thinking..."):
            import langsmith as ls
            # Wrap the graph invocation in a LangSmith trace context
            # with ls.traceable(run_type="tool", name="Graph Invoke"):
            result = st.session_state.graph.invoke({"user_input": user_input})

            # Save chat history
            st.session_state.history.append({
                "input": user_input,
                "output": result.get("summary"),
                "action": result.get("action"),
                "raw": result.get("raw")
            })

        st.session_state.allow_question = False
        st.rerun()
else:
    if st.button("Ask Another Question"):
        st.session_state.allow_question = True
        st.rerun()

# ------------------- Render Chat Bubbles -------------------
for chat in reversed(st.session_state.history):
    st.markdown(
        f"<div class='user-bubble'><b>You:</b> {chat['input']}</div>",
        unsafe_allow_html=True
    )

    rag_score = ""
    if chat["action"] == "pdf_rag" and isinstance(chat["raw"], list):
        rag_score = f"<br><small>📊 RAG retrieved {len(chat['raw'])} docs</small>"

    st.markdown(
        f"<div class='bot-bubble'><b>Bot ({chat['action']}):</b> {chat['output']}{rag_score}</div>",
        unsafe_allow_html=True
    )

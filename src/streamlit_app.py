# streamlit_app.py
import os
import time
import streamlit as st
from dotenv import load_dotenv

from graph import build_pipeline_graph
from pdf_rag import build_embeddings_and_upsert
from evaluations import greeting_eval, guardrail_eval, correctness_eval

import langsmith as ls
from langsmith import Client

# ------------------- Load .env and Setup LangSmith -------------------
load_dotenv()

os.environ["LANGSMITH_TRACING"] = "true"

LS_API_KEY = os.getenv("LANGSMITH_API_KEY")
if not LS_API_KEY:
    st.warning("LangSmith API key not found. Tracing and evaluation logging disabled.")
else:
    os.environ["LANGSMITH_API_KEY"] = LS_API_KEY

ls_client = Client() if LS_API_KEY else None

# ------------------- Wrapper Classes -------------------
class ExampleWrapper:
    def __init__(self, inputs, expected=None):
        self.inputs = inputs
        self.expected = expected


class RunWrapper:
    def __init__(self, outputs):
        self.outputs = outputs

# ------------------- Streamlit Page Setup -------------------
st.set_page_config(page_title="Weather + PDF RAG (LangGraph)", layout="wide")

st.markdown("""
<style>
.user-bubble {
    background: #c9f7ff;
    color: #000000;          /* ✅ force black text */
    padding: 1rem;
    border-radius: 12px;
    margin-bottom: 0.5rem;
    font-size: 1.05rem;
}

.bot-bubble {
    background: #fff3d8;
    color: #1a1a1a;          /* ✅ dark text */
    padding: 1rem;
    border-radius: 12px;
    margin-bottom: 1rem;
    font-size: 1.05rem;
}

.bot-bubble small {
    color: #2e7d32;          /* ✅ green for evaluations */
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


st.title("Weather + PDF RAG (LangGraph) Chatbot")

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

# ------------------- Reset -------------------
if st.button("Reset Conversation"):
    st.session_state.history = []
    st.session_state.allow_question = True
    st.rerun()

# ------------------- PDF Upload -------------------
st.markdown("### Upload PDF")

if st.session_state.pdf_uploaded:
    if st.button("Upload New PDF"):
        st.session_state.pdf_uploaded = False
        st.rerun()

uploaded = None
if not st.session_state.pdf_uploaded:
    uploaded = st.file_uploader("Choose a PDF", type=["pdf"])

if uploaded and not st.session_state.pdf_uploaded:
    count = build_embeddings_and_upsert(uploaded)
    st.success(f"PDF embedded successfully ({count} chunks)")
    st.session_state.pdf_uploaded = True
    time.sleep(1)
    st.rerun()

# ------------------- Chat -------------------
st.markdown("### Ask a Question")

if st.session_state.allow_question:
    user_input = st.text_input("Ask here...")
else:
    user_input = None

if st.session_state.allow_question and st.button("Send") and user_input.strip():

    with st.spinner("Thinking..."):

        invoke_fn = st.session_state.graph.invoke

        if LS_API_KEY:
            invoke_fn = ls.traceable(
                run_type="chain",
                name="LangGraph Pipeline"
            )(invoke_fn)

        # 🔥 CAPTURE RUN ID
        with ls.trace(name="Graph Run", run_type="chain") as run_ctx:
            result = invoke_fn({"user_input": user_input})
            run_id = run_ctx.id if LS_API_KEY else None

        # ------------------- Wrap -------------------
        example = ExampleWrapper(
            inputs={"input": user_input},
            expected={"output": result.get("summary")}
        )

        run_wrapper = RunWrapper(outputs=result)

        # ------------------- Evaluations -------------------
        eval_results = {}
        evaluators = [greeting_eval, guardrail_eval, correctness_eval]

        for evaluator in evaluators:
            try:
                score = evaluator(run_wrapper, example)
                eval_results[evaluator.__name__] = score

                # 🔥 LOG TO LANGSMITH
                if ls_client and run_id:
                    ls_client.create_feedback(
                        run_id=run_id,
                        key=evaluator.__name__,
                        score=1 if score else 0
                    )

            except Exception as e:
                eval_results[evaluator.__name__] = f"Error: {e}"

        # ------------------- Save History -------------------
        st.session_state.history.append({
            "input": user_input,
            "output": result.get("summary"),
            "action": result.get("action"),
            "raw": result.get("raw"),
            "evaluation": eval_results
        })

    st.session_state.allow_question = False
    st.rerun()

if not st.session_state.allow_question:
    if st.button("Ask Another Question"):
        st.session_state.allow_question = True
        st.rerun()

# ------------------- Render Chat -------------------
for chat in reversed(st.session_state.history):

    st.markdown(
        f"<div class='user-bubble'><b>You:</b> {chat['input']}</div>",
        unsafe_allow_html=True
    )

    eval_text = "<br><small>✅ Evaluations: " + ", ".join(
        f"{k}={v}" for k, v in chat["evaluation"].items()
    ) + "</small>"

    st.markdown(
        f"<div class='bot-bubble'><b>Bot:</b> {chat['output']}{eval_text}</div>",
        unsafe_allow_html=True
    )

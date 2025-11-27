# src/pipeline.py

from typing import Dict, List
from weather import fetch_weather, format_weather_summary
from pdf_rag import query_rag
from llm_utils import summarize_with_llm


def decide_action(user_input: str) -> str:
    """
    Simple routing logic: weather vs RAG.
    LangGraph will call this node to decide the next move.
    """
    lower = user_input.lower()
    weather_triggers = ["weather", "temperature", "rain", "sunny", "forecast", "humidity"]

    return "weather" if any(tok in lower for tok in weather_triggers) else "pdf_rag"


def build_guardrailed_prompt(contexts: List[str], user_question: str) -> str:
    """
    Construct a guardrailed prompt guaranteeing the LLM stays inside the context.
    """

    context_block = (
        "No related documents found in the vector store."
        if not contexts else "\n\n---\n\n".join(contexts)
    )

    return f"""
You are a strict Retrieval-Augmented QA assistant.

RULES:
1. You must answer ONLY using the information provided in the CONTEXT.
2. If the answer is not present in the context, reply ONLY with:  
   "I don’t know based on the provided documents."
3. Do NOT use outside knowledge.
4. Do NOT guess, speculate, or hallucinate.
5. Do NOT add extra details not grounded in the context.

CONTEXT:
{context_block}

QUESTION:
{user_question}

Your answer must strictly follow the rules above.
""".strip()


def extract_city(txt: str) -> str:
    """
    Extracts a city name using a simple heuristic.
    """
    lower = txt.lower()
    if " in " in lower:
        try:
            after = lower.split(" in ", 1)[1]
            return after.split()[0].strip(",.?")
        except Exception:
            pass

    tokens = txt.strip().split()
    return tokens[-1].strip("?,.") if tokens else "London"


# ----------------------------
# Components used by LangGraph
# ----------------------------

def weather_node(state: Dict, openweather_key: str = None) -> Dict:
    """
    LangGraph node: Calls weather API and formats it.
    """
    user_input = state["user_input"]
    city = extract_city(user_input)

    weather_json = fetch_weather(city, api_key=openweather_key)
    summary = format_weather_summary(weather_json)

    return {
        "action": "weather",
        "raw": weather_json,
        "summary": summary
    }


def rag_node(state: Dict) -> Dict:
    """
    LangGraph node: RAG retrieval + guardrailed LLM.
    """
    user_input = state["user_input"]

    # retrieve from vector DB
    contexts = query_rag(user_input)

    # guardrailed prompt
    prompt = build_guardrailed_prompt(contexts, user_input)

    # summarize with LLM
    summary = summarize_with_llm(prompt)

    return {
        "action": "pdf_rag",
        "raw": contexts,
        "summary": summary
    }

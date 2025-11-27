from langgraph.graph import END, StateGraph
from typing import TypedDict
from pipeline import decide_action, weather_node, rag_node

# Correct State definition
class PipelineState(TypedDict):
    user_input: str
    action: str | None
    raw: dict | list | None
    summary: str | None


def build_pipeline_graph(openweather_key: str = None):
    graph = StateGraph(PipelineState)

    # Nodes
    graph.add_node("decider", lambda s: {"action": decide_action(s["user_input"])})
    graph.add_node("weather", lambda s: weather_node(s, openweather_key=openweather_key))
    graph.add_node("pdf_rag", rag_node)

    # Wiring
    graph.set_entry_point("decider")

    graph.add_conditional_edges(
        "decider",
        lambda state: state["action"],
        {
            "weather": "weather",
            "pdf_rag": "pdf_rag"
        }
    )

    graph.add_edge("weather", END)
    graph.add_edge("pdf_rag", END)

    return graph.compile()

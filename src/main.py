# src/main.py

import os
from dotenv import load_dotenv
from graph import build_pipeline_graph

# LangSmith Tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")

load_dotenv()

def run_pipeline():
    user_input = input("Enter your query: ")

    graph = build_pipeline_graph(
        openweather_key=os.getenv("OPENWEATHER_API_KEY")
    )

    result = graph.invoke({"user_input": user_input})

    print("\n=== RESULT ===")
    print("Action:", result.get("action"))
    print("Summary:", result.get("summary"))
    print("Raw:", result.get("raw"))


if __name__ == "__main__":
    run_pipeline()
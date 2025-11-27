# src/langsmith_eval.py
import os

def log_evaluation(prompt: str, response: str, score: float = None, notes: str = None):
    """
    Minimal placeholder for LangSmith evaluation logging.
    Replace this function with the LangSmith SDK calls you have in your environment.
    For example, with langsmith SDK you would create a run/eval and upload the model output and
    associated metrics.

    TODO: replace with your project's LangSmith client code, e.g.:
      from langsmith import Client
      client = Client(api_key=...)
      client.create_evaluation_run(...)
    """
    # Example log saved to local file so you can show a screenshot in LangSmith later
    out = {
        "prompt": prompt,
        "response": response,
        "score": score,
        "notes": notes,
    }
    # simple local dump for demo
    import json, datetime
    fname = f"langsmith_local_eval_{datetime.datetime.now().isoformat()}.json"
    with open(fname, "w") as f:
        json.dump(out, f, indent=2)
    return fname

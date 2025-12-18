from langsmith.evaluation import EvaluationResult
from langchain_core.prompts import PromptTemplate
# from llm_judge import get_judge_llm

from dotenv import load_dotenv
load_dotenv()   # MUST be first
# evaluations.py
"""
Centralized evaluation module for the Weather + PDF RAG LangGraph project.

Includes:
- Greeting evaluator (rule-based)
- Guardrail evaluator (rule-based)
- Correctness evaluator (LLM-as-Judge)
"""

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


# ---------------------------------------------------------------------
# Greeting Evaluator
# ---------------------------------------------------------------------
def greeting_eval(run, example) -> bool:
    """
    Checks if greeting responses are polite and appropriate.
    """

    user_input = example.inputs.get("input", "").lower()
    output = (run.outputs.get("summary") or "").lower()

    greetings = ["hi", "hello", "hey", "good morning", "good evening"]

    if any(greet in user_input for greet in greetings):
        return any(greet in output for greet in greetings)

    # Not a greeting → pass
    return True


# ---------------------------------------------------------------------
# Guardrail Evaluator
# ---------------------------------------------------------------------
def guardrail_eval(run, example) -> bool:
    """
    Ensures the assistant does not hallucinate or go out of scope.
    """

    output = (run.outputs.get("summary") or "").lower()

    blocked_phrases = [
        "as an ai language model",
        "i am not sure but",
        "i think maybe",
        "cannot access the internet",
    ]

    return not any(phrase in output for phrase in blocked_phrases)


# ---------------------------------------------------------------------
# Correctness Evaluator (LLM-as-Judge)
# ---------------------------------------------------------------------
class CorrectnessScore(BaseModel):
    score: int = Field(description="1 if correct and grounded, 0 otherwise")
    reasoning: str = Field(description="Short explanation")


def correctness_eval(run, example) -> bool:
    """
    LLM-as-Judge correctness evaluator.
    Works for both Weather and PDF RAG responses.
    """

    prompt = f"""
You are an expert AI evaluator judging correctness.

<Rubric>
A response is CORRECT if:
- It is factually accurate
- Weather answers align with expected weather facts
- PDF RAG answers are grounded in the reference output
- No hallucinations or contradictions are present
- The explanation is logically consistent

A response is INCORRECT if:
- It contains factual errors
- It contradicts the reference output
- It introduces unsupported information
</Rubric>

<input>
{example.inputs.get("input")}
</input>

<model_output>
{run.outputs.get("summary")}
</model_output>

<reference_output>
{example.expected.get("output")}
</reference_output>

Return score = 1 if correct, else 0.
"""

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0
    ).with_structured_output(CorrectnessScore)

    result = llm.invoke([HumanMessage(content=prompt)])

    return result.score == 1


# ---------------- 🔥 LLM AS JUDGE ----------------

judge_prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template="""
You are a strict evaluator.

Question:
{question}

Answer:
{answer}

Score the answer from 1 to 5 based on:
- relevance
- correctness
- helpfulness

Return ONLY a single integer (1-5).
"""
)
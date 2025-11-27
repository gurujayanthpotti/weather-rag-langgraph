from src.pipeline import decide_action, extract_city, build_guardrailed_prompt

def test_decide_action_weather():
    assert decide_action("What is the weather in Mumbai?") == "weather"

def test_decide_action_pdf():
    assert decide_action("Explain chapter 2 of document") == "pdf_rag"

def test_extract_city():
    assert extract_city("weather in hyderabad") == "hyderabad"
    assert extract_city("Tell me weather Delhi") == "delhi"

def test_guardrails_no_context():
    p = build_guardrailed_prompt([], "test?")
    assert "No related documents found" in p

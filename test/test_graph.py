from src.graph import build_pipeline_graph

def test_graph_weather_path():
    graph = build_pipeline_graph(openweather_key="test")
    result = graph.invoke({"user_input": "weather in pune"})
    assert result["action"] == "weather"

def test_graph_pdf_path():
    graph = build_pipeline_graph()
    result = graph.invoke({"user_input": "Explain the PDF introduction"})
    assert result["action"] == "pdf_rag"

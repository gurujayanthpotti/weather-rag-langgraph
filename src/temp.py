# import os
# from dotenv import load_dotenv
# from langchain.agents import create_agent
# from langchain_openai import AzureChatOpenAI

# # Load .env
# load_dotenv()

# # Tool definition
# def get_weather(city: str) -> str:
#     return f"It's always sunny in {city}!"

# # Azure OpenAI LLM
# llm = AzureChatOpenAI(
#     deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
#     # model_name="gpt-5-mini",
#     # openai_api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
#     api_version="2023-07-01-preview",
#     api_key=os.getenv("AZURE_OPENAI_KEY"),
#     temperature=0.2,
# )

# # Create agent
# agent = create_agent(
#     model=llm,
#     tools=[get_weather],
#     system_prompt="You are a helpful assistant",
# )

# # Run agent (tracing enabled via LANGSMITH_* env variables)
# response = agent.invoke(
#     {"messages": [{"role": "user", "content": "What is the weather in San Francisco?"}]}
# )

# print(response)
from langsmith import Client
import os

client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))
try:
    runs = client.list_runs(limit=1)
    print("✅ LangSmith Auth OK")
except Exception as e:
    print("❌ LangSmith Auth Failed:", e)

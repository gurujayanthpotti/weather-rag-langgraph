import os
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def get_llm(temperature: float = 0.2):
    """
    Returns an Azure OpenAI-compatible Chat LLM.
    """
    # model_name = model_name or os.getenv("LLM_MODEL", "gpt")
    api_key = os.getenv("AZURE_OPENAI_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")  # required for AzureChatOpenAI

    print(api_key)
    print(endpoint)
    print(deployment_name)
    return AzureChatOpenAI(
        deployment_name=deployment_name,
        temperature=temperature,
        openai_api_key=api_key,
        azure_endpoint=endpoint,
        api_version='2024-02-01'
    )


def summarize_with_llm(text: str, llm=None) -> str:
    """
    Summarizes text using LCEL (LangChain Expression Language) instead of deprecated LLMChain.
    """
    if llm is None:
        llm = get_llm()

    prompt = PromptTemplate(
        input_variables=["article"],
        template=(
            "You are a helpful summarizer. Read the content carefully and "
            "provide a concise summary (3-5 sentences):\n\n{article}"
        )
    )

    # LCEL chain: prompt → model → output parser
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"article": text})
    return response

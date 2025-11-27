import os
from typing import List, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import openai
from io import BytesIO
from pypdf import PdfReader
from openai import AzureOpenAI
from uuid import uuid4
from qdrant_utils import get_qdrant_client, create_collection_if_not_exists, upsert_documents, query_similar
from dotenv import load_dotenv

load_dotenv()

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "pdf_docs")
openai_api_key=os.getenv("OPENAI_API_KEY")
qdrant_api_key=os.getenv("QDRANT_API_KEY")


class AzureEmbeddingWrapper:
    def __init__(self, azure_endpoint, api_key, api_version, model_name):
        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version
        )
        self.model_name = model_name

    # For queries (RAG)
    def embed_query(self, query: str):
        response = self.client.embeddings.create(
            model=self.model_name,
            input=query
        )
        return response.data[0].embedding

    # For documents (RAG ingestion)
    def embed_documents(self, docs: list[str]):
        response = self.client.embeddings.create(
            model=self.model_name,
            input=docs
        )
        return [item.embedding for item in response.data]


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from a PDF using PyPDF with a BytesIO buffer."""
    pdf_stream = BytesIO(pdf_bytes)
    reader = PdfReader(pdf_stream)

    text_chunks = []
    for page in reader.pages:
        text_chunks.append(page.extract_text() or "")

    return "\n".join(text_chunks)



def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Split PDF text into chunks using RecursiveCharacterTextSplitter.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    docs = splitter.create_documents([text])
    return [d.page_content for d in docs]

def get_embeddings(text_to_embed):
    try:
        cilent=AzureOpenAI(
            azure_endpoint='https://bpost-azure-bods-dasinco.openai.azure.com/',
            api_key='acebef8cb1f04c1c9b0123a61082732c',
            api_version='2024-02-01'
        )
        response=cilent.embeddings.create(
            model="text-embedding-ada-002",
            input=text_to_embed
        )
        if isinstance(text_to_embed, list):
            return [item.embedding for item in response.data]
        embedding = response.data[0].embedding
        return embedding
    except (openai.RateLimitError) as e:
        print(e)

def build_embeddings_and_upsert(pdf_input, qdrant_url: str = None):
    """
    Accepts either:
    - a file path (str)
    - a Streamlit UploadedFile object
    """

    # --- Detect input type ---
    if isinstance(pdf_input, str):
        # Treat as local file path
        with open(pdf_input, "rb") as f:
            pdf_bytes = f.read()
        source_name = os.path.basename(pdf_input)

    else:
        # Treat as Streamlit UploadedFile
        pdf_bytes = pdf_input.read()
        source_name = getattr(pdf_input, "name", "uploaded.pdf")

    # --- FIX: Pass bytes to extract_text_from_pdf ---
    text = extract_text_from_pdf(pdf_bytes)

    # --- Chunking ---
    chunks = chunk_text(text)

    # --- Embeddings ---
    embeddings = AzureEmbeddingWrapper(
        azure_endpoint='https://bpost-azure-bods-dasinco.openai.azure.com/',
        api_key='acebef8cb1f04c1c9b0123a61082732c',
        api_version="2024-02-01",
        model_name="text-embedding-ada-002"
    )

    vectors = embeddings.embed_documents(chunks)

    # --- Qdrant ---
    qclient = get_qdrant_client()
    dim = len(vectors[0])

    create_collection_if_not_exists(qclient, COLLECTION_NAME, dim)

    ids = [str(uuid4()) for _ in chunks]
    metadatas = [{"text": chunk, "source": source_name} for chunk in chunks]

    upsert_documents(qclient, COLLECTION_NAME, vectors, metadatas, ids)

    return len(vectors)



def query_rag(query: str, top_k: int = 4):
    """
    Query Qdrant vector DB and return top relevant chunks.
    """
    embeddings = AzureEmbeddingWrapper(
    azure_endpoint='https://bpost-azure-bods-dasinco.openai.azure.com/',
    api_key='acebef8cb1f04c1c9b0123a61082732c',
    api_version='2024-02-01',
    model_name="text-embedding-ada-002"
)
    # embeddings = OpenAIEmbeddings()
    q_vec = embeddings.embed_query(query)

    qclient = get_qdrant_client()

    results = query_similar(qclient, COLLECTION_NAME, q_vec, top_k=top_k)
    print(results)
    # Extract context text
    contexts = []
    for r in results:
        print("\nPROCESSING RESULT:", r)

        # Handle both dict and Qdrant ScoredPoint object
        payload = None

        if isinstance(r, dict) and "payload" in r:
            payload = r["payload"]
        elif hasattr(r, "payload"):
            payload = r.payload

        print("PAYLOAD:", payload)

        if isinstance(payload, dict) and "text" in payload:
            txt = payload["text"]
            print("FOUND TEXT:", txt[:200], "...")
            contexts.append(txt)
        else:
            print("⚠ No text field found in payload!")

    print("\nFINAL CONTEXTS:", contexts)
    return contexts

# print(build_embeddings_and_upsert(r'C:\Users\guruj\Documents\langgraph-weather-rag\sample_data\Geography_of_India.pdf'))
# query_rag("Biodiversity in india")
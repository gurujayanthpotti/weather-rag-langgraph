# src/qdrant_utils.py
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    QueryRequest,
    NamedVector
)
from typing import List, Dict
import os
from qdrant_client.models import Filter


# -------------------------------------
# Qdrant Client
# -------------------------------------
def get_qdrant_client(url: str = None, api_key: str = None) -> QdrantClient:
    url = url or os.getenv(
        "QDRANT_URL",
        "https://42a3d4c3-be2f-4463-82c1-294e135a6512.us-east4-0.gcp.cloud.qdrant.io:6333"
    )
    api_key = api_key or os.getenv("QDRANT_API_KEY")

    return QdrantClient(url=url, api_key=api_key)


# -------------------------------------
# Create Collection
# -------------------------------------
def create_collection_if_not_exists(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
    distance: Distance = Distance.COSINE
):
    collections = client.get_collections().collections

    if not any(c.name == collection_name for c in collections):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=distance
            )
        )
        print(f"✅ Created Qdrant collection: {collection_name}")
    else:
        print(f"➡️ Qdrant collection already exists: {collection_name}")


# -------------------------------------
# Upsert Documents
# -------------------------------------
def upsert_documents(
    client: QdrantClient,
    collection_name: str,
    vectors: List[List[float]],
    metadatas: List[Dict],
    ids: List[str]
):
    # Ensure vectors are valid
    cleaned_vectors = [list(map(float, v)) for v in vectors]

    points = [
        PointStruct(
            id=id_,
            vector=cleaned_vectors[i],
            payload=metadatas[i]
        )
        for i, id_ in enumerate(ids)
    ]

    client.upsert(collection_name=collection_name, points=points)
    print(f"📌 Upserted {len(points)} points into {collection_name}")


# -------------------------------------
# Query Similar Vectors
# Uses new Qdrant API → client.query_points()
# -------------------------------------

def query_similar(client, collection_name, query_embedding, top_k=5):
    """
    Correct for qdrant-client 1.16.1
    """
    q_vec = [float(x) for x in query_embedding]
    print(q_vec)
    # Query the collection
    response = client.query_points(
        collection_name=collection_name,
        query=q_vec,
        limit=top_k
    )

    # Normalize results
    results = []
    for p in response.points:
        results.append({
            "id": p.id,
            "score": p.score,
            "payload": p.payload
        })
    return results

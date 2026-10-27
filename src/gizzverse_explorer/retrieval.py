from qdrant_client import QdrantClient
from qdrant_client.models import Fusion, FusionQuery, Prefetch
from sentence_transformers import SentenceTransformer, CrossEncoder
from fastembed import SparseTextEmbedding
from gizzverse_explorer.config import (
    COLLECTION_NAME, PREFETCH_LIMIT, RERANK_LIMIT, RETRIEVAL_LIMIT
)
from gizzverse_explorer.model_loader import (
    embed_query, get_device, load_dense, load_reranker, load_sparse, to_sparse_vector
)

def load_models():
    print(f"Initializing models (device: {get_device()})...")

    return load_dense(), load_sparse(), load_reranker()

def retrieve_chunks(question: str, dense_model: SentenceTransformer, sparse_model: SparseTextEmbedding, reranker: CrossEncoder, client: QdrantClient, limit: int=RERANK_LIMIT) -> list[dict]:
    dense_query = embed_query(dense_model, question)
    sparse_query = to_sparse_vector(next(iter(sparse_model.query_embed(question))))

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(query=dense_query, using="dense", limit=PREFETCH_LIMIT),
            Prefetch(query=sparse_query, using="sparse", limit=PREFETCH_LIMIT)
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=RETRIEVAL_LIMIT,
        with_payload=True
    ).points

    chunks = []
    for result in results:
        if result.payload:
            chunks.append({**result.payload, "fusion_score": float(result.score)})

    if not chunks:
        return []

    scores = reranker.predict([
        [
            question,
            (
                f"Song: {chunk['title']}\n"
                f"Album: {chunk['album']}\n"
                f"Lyrics:\n{chunk['text']}"
            )
        ]
        for chunk in chunks
    ])

    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)

    chunks.sort(key=lambda chunk: chunk["rerank_score"], reverse=True)

    unique_chunks = []
    seen_songs = set()

    for chunk in chunks:
        song_id = chunk.get("track_id") or (
            chunk["title"],
            chunk["album"],
        )

        if song_id in seen_songs:
            continue

        seen_songs.add(song_id)
        unique_chunks.append(chunk)

        if len(unique_chunks) >= limit:
            break

    return unique_chunks
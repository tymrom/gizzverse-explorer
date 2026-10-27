import torch
from fastembed import SparseTextEmbedding
from qdrant_client import models
from sentence_transformers import CrossEncoder, SentenceTransformer
from gizzverse_explorer import config

def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_dense() -> SentenceTransformer:
    model = SentenceTransformer(config.DENSE_MODEL_NAME, device=get_device())

    if config.DENSE_MAX_SEQ_LENGTH:
        model.max_seq_length = config.DENSE_MAX_SEQ_LENGTH

    return model

def load_sparse() -> SparseTextEmbedding:
    return SparseTextEmbedding(model_name=config.SPARSE_MODEL_NAME)

def load_reranker() -> CrossEncoder:
    return CrossEncoder(config.RERANKER_MODEL_NAME, device=get_device(), trust_remote_code=True)

def embed_documents(dense_model: SentenceTransformer, texts: list[str]):
    return dense_model.encode(
        [config.DENSE_DOC_PREFIX + text for text in texts],
        batch_size=32,
        show_progress_bar=True, 
        normalize_embeddings=True
    )

def embed_query(dense_model: SentenceTransformer, question: str) -> list[float]:
    return dense_model.encode(
        config.DENSE_QUERY_PREFIX + question,
        normalize_embeddings=True
    ).tolist()

def to_sparse_vector(embedding) -> models.SparseVector:
    return models.SparseVector(
        indices=embedding.indices.tolist(),
        values=embedding.values.tolist()
    )

def report_truncation(dense_model: SentenceTransformer, texts: list[str]) -> None:
    limit = dense_model.max_seq_length
    prefixed = [config.DENSE_DOC_PREFIX + text for text in texts]
    lengths = [len(ids) for ids in dense_model.tokenizer(prefixed)["input_ids"]]
    over = sum(length > limit for length in lengths)

    print(
        f"Dense model max_seq_length={limit}; "
        f"{over}/{len(texts)} chunks exceed it and will be truncated "
        f"(longest: {max(lengths)} tokens)"
    )

def build_document_text(chunk: dict) -> str:
    metadata = chunk["metadata"]

    return (
        f"Song: {metadata['title']}\n"
        f"Album: {metadata['album']}\n"
        f"Artist: {metadata['artist']}\n"
        f"Lyrics:\n{chunk['text']}"
    )

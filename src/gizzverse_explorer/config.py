from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
DISCOGRAPHY_PATH = RAW_DIR / "discography.json"
LYRICS_DIR = RAW_DIR / "lyrics"
CHUNKS_PATH = DATA_DIR / "processed" / "chunks.json"
QDRANT_PATH = str(DATA_DIR / "qdrant")

COLLECTION_NAME = "kglw_lyrics"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

DENSE_MODEL_NAME = "BAAI/bge-base-en-v1.5"
DENSE_MAX_SEQ_LENGTH = None
DENSE_QUERY_PREFIX = (
    "Represent this sentence for searching relevant passages: "
)
DENSE_DOC_PREFIX = ""
SPARSE_MODEL_NAME = "prithivida/Splade_PP_en_v1"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

PREFETCH_LIMIT = 30
RETRIEVAL_LIMIT = 30
RERANK_LIMIT = 5

LLM_MODEL_NAME = "gemma3:4b"
LLM_TEMPERATURE = 0.1
LLM_NUM_CTX = 8192
LLM_THINK = None
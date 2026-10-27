import streamlit as st
from qdrant_client import QdrantClient

from gizzverse_explorer.config import (
    DENSE_MODEL_NAME,
    LLM_MODEL_NAME,
    QDRANT_PATH,
    RERANKER_MODEL_NAME,
    SPARSE_MODEL_NAME,
)
from gizzverse_explorer.rag import answer_question
from gizzverse_explorer.retrieval import load_models


st.set_page_config(
    page_title="Gizzverse Explorer",
    page_icon="🪐",
    layout="centered",
)

st.markdown(
    """
    <style>
      .block-container { max-width: 860px; padding-top: 2.25rem; }
      .gizz-subtitle { color: #a8aab3; margin-top: -.35rem; }
      .gizz-label { color: #d7bd86; letter-spacing: 0.1em; font-size: 0.72rem; font-weight: 700; display: block; line-height: 1.6; padding: 0.25rem 0; overflow: visible; white-space: normal; }
      .gizz-note { color: #9b9ea8; font-size: .87rem; }
      div[data-testid="stChatMessage"] { border-radius: 14px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_backend():
    """Load large models and open local Qdrant only once per Streamlit server.

    The current project uses a local Qdrant store; treat this as a
    single-user, local prototype rather than a multi-user deployment.
    """
    dense_model, sparse_model, reranker = load_models()
    client = QdrantClient(path=QDRANT_PATH)
    return dense_model, sparse_model, reranker, client


def summarize_tracks(chunks: list[dict]) -> list[dict]:
    """Deduplicate displayed search candidates, without claiming citations."""
    seen = set()
    tracks = []

    for chunk in chunks:
        key = chunk.get("track_id") or (
            chunk.get("title"),
            chunk.get("album"),
        )
        if key in seen:
            continue
        seen.add(key)
        tracks.append(
            {
                "title": chunk.get("title", "Unknown track"),
                "album": chunk.get("album", "Unknown album"),
                "rerank_score": chunk.get("rerank_score"),
                "fusion_score": chunk.get("fusion_score"),
            }
        )

    return tracks


def show_assistant_message(message: dict, show_debug: bool) -> None:
    st.markdown(message["content"])

    tracks = message.get("tracks", [])
    if not tracks:
        return

    with st.expander(f"Retrieved tracks · {len(tracks)}", expanded=False):
        st.caption(
            "These tracks were sent to the answer model. "
            "They are not necessarily all cited in its response."
        )
        for index, track in enumerate(tracks, start=1):
            st.write(f"{index}. {track['title']} — {track['album']}")
            if show_debug:
                scores = []
                for field, label in (
                    ("rerank_score", "Rerank"),
                    ("fusion_score", "RRF"),
                ):
                    value = track.get(field)
                    if isinstance(value, (int, float)):
                        scores.append(f"{label}: {value:.3f}")
                if scores:
                    st.caption("  ·  ".join(scores))


if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Explorer controls")
    show_debug = st.toggle("Show retrieval scores", value=False)
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("CURRENT ENGINE")
    st.write(f"**LLM:** `{LLM_MODEL_NAME}`")
    st.write(f"**Dense:** `{DENSE_MODEL_NAME}`")
    st.write(f"**Sparse:** `{SPARSE_MODEL_NAME}`")
    st.write(f"**Reranker:** `{RERANKER_MODEL_NAME}`")

    st.divider()
    st.caption("NEXT: STORY LAB")
    st.write(
        "Creative album trilogies, story timelines, and deeper thematic "
        "analysis are planned for a future LangGraph workflow."
    )
    st.caption(
        "Conversation history is displayed here, but follow-up questions "
        "are not yet passed to the model as dialogue context."
    )

st.markdown('<div class="gizz-label">EXPLORE THE GIZZVERSE</div>', unsafe_allow_html=True)
st.title("Gizzverse Explorer 🪐")
st.markdown(
    '<div class="gizz-subtitle">Discover lyrics, themes, and connections across '
    'King Gizzard & the Lizard Wizard.</div>',
    unsafe_allow_html=True,
)

example_question = None
if not st.session_state.messages:
    st.write("")
    st.caption("Try a question")
    examples = [
        "Which songs are about snakes?",
        "What environmental themes appear in Planet B?",
        "What connects Gila Monster and Dragon?",
    ]
    for index, question in enumerate(examples):
        if st.button(question, key=f"example_{index}", use_container_width=True):
            example_question = question

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            show_assistant_message(message, show_debug)
        else:
            st.markdown(message["content"])

user_question = st.chat_input("Ask about songs, lyrics, themes, or albums…")
question = user_question or example_question

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Exploring the Gizzverse…"):
                dense_model, sparse_model, reranker, client = load_backend()
                answer, chunks = answer_question(
                    question,
                    dense_model,
                    sparse_model,
                    reranker,
                    client,
                )

            if not chunks:
                answer = "I couldn't find relevant song excerpts for that question."
            elif not answer:
                answer = "I found relevant songs but couldn't generate an answer."

            message = {
                "role": "assistant",
                "content": answer,
                "tracks": summarize_tracks(chunks),
            }
            st.session_state.messages.append(message)
            show_assistant_message(message, show_debug)

        except Exception as exc:
            st.error("The local RAG backend couldn't complete this request.")
            if show_debug:
                st.exception(exc)
            else:
                st.caption(
                    "Check that Ollama is running, Qdrant has been indexed, "
                    "and no other process is holding the local Qdrant directory. "
                    "Enable retrieval scores to see the technical error."
                )

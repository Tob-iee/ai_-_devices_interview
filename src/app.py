import os
import asyncio
import streamlit as st

import torch

# Adjust import to your backend module name
from rag import (
    build_vec_index_from_uploads,
    build_sum_index_from_uploads,
    build_chat_engine,
)
from core.utils import format_response_payload

# # Prevent path errors for Torch
try:
    torch.classes.__path__ = []  # type: ignore[attr-defined]
except Exception:
    pass

# ---- Config lists (edit to your taste) ----
# Shown only when mode is "teach"
GROQ_MODELS = [
    # ensure these exist/are enabled for your Groq account
    "openai/gpt-oss-20b",
    "llama-3.1-8b-instant",
    "llama-3.1-70b-versatile",
]

# Shown only when mode is "summarize" or "explain"
HF_MODELS = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B",
    "HuggingFaceH4/zephyr-7b-beta",
    "Qwen/Qwen2.5-7B-Instruct",
]

EMBED_MODELS = [
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-small-en-v1.5",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
]


DEFAULT_GROQ_MODEL = GROQ_MODELS[0]
DEFAULT_HF_MODEL = HF_MODELS[0]
DEFAULT_EMBED_MODEL = EMBED_MODELS[0]

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

def _init_state():
    for k, v in {
        "chat_engine": None,
        "history": [],
        "last_file_name": None,
        "last_mode": None,
        "last_model": None,
    }.items():
        if k not in st.session_state: st.session_state[k] = v

def _maybe_reset_history(new_file_name: str, mode: str, model_name: str):
    if (
        st.session_state.last_file_name != new_file_name
        or st.session_state.last_mode != mode
        or st.session_state.last_model != model_name
    ):
        st.session_state.history = []
        st.session_state.chat_engine = None
        st.session_state.last_file_name = new_file_name
        st.session_state.last_mode = mode
        st.session_state.last_model = model_name

def _render_history():
    for msg in st.session_state.history:
        st.chat_message(msg["role"]).markdown(msg["content"])

def main():
    st.set_page_config(page_title="AI Document Assistant", layout="wide")
    _init_state()

    st.title("📑 AI Document Assistant")
    st.caption(
        "Upload a PDF, pick a mode — **Summarize**, **Explain**, or **Teach**. "
        "Teach uses **Groq**; other modes use local **HF** models. Qdrant Cloud is used for vector storage if configured."
    )

    st.sidebar.header("Configuration")
    mode = st.sidebar.radio("Interaction Mode", ["summarize", "explain", "teach"], index=0)

    if mode == "teach":
        llm_model = st.sidebar.selectbox("Groq LLM", GROQ_MODELS, index=0)
        if not os.getenv("GROQ_API_KEY"):
            st.sidebar.error("GROQ_API_KEY is missing.")
    else:
        llm_model = st.sidebar.selectbox("HF (local) LLM", HF_MODELS, index=0)

    embed_model = st.sidebar.text_input("Embedding Model", DEFAULT_EMBED_MODEL)

    # Show Qdrant Cloud status
    with st.sidebar.expander("Vector DB"):
        if QDRANT_URL:
            st.success("Qdrant Cloud detected")
            st.code(QDRANT_URL)
        else:
            st.warning("No QDRANT_URL set → will try localhost:6333 → then in-memory")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file is not None:
        _maybe_reset_history(uploaded_file.name, mode, llm_model)

        if st.session_state.chat_engine is None:
            with st.spinner("Preparing your document and engine for querying..."):
                if mode == "summarize":
                    sum_index = build_sum_index_from_uploads(
                        file=uploaded_file,
                        embed_model=embed_model,
                        url=QDRANT_URL,
                        api_key=QDRANT_API_KEY,
                    )
                    vec_index = None
                else:
                    vec_index = build_vec_index_from_uploads(
                        file=uploaded_file,
                        embed_model=embed_model,
                        url=QDRANT_URL,
                        api_key=QDRANT_API_KEY,
                    )
                    sum_index = None

                st.session_state.chat_engine = build_chat_engine(
                    mode=mode,
                    vec_index=vec_index,
                    sum_index=sum_index,
                    llm_model=llm_model,
                )

    st.subheader(f"💬 Chat Mode: {mode.capitalize()}")
    _render_history()

    if st.session_state.chat_engine is None:
        st.info("Upload a PDF to start.")
        return

    prompt = st.chat_input("Ask a question about your document...")
    if not prompt:
        return

    st.session_state.history.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if mode in ["summarize", "explain"]:
                resp = st.session_state.chat_engine.chat(prompt)
                payload = format_response_payload(resp, max_citations=5)
                answer = payload.get("answer", "")
                st.markdown(answer)
                cits = payload.get("citations", [])
                if cits:
                    with st.expander("Citations"):
                        for i, c in enumerate(cits, 1):
                            st.markdown(f"{i}. {c}")
            else:
                async def run_teach(q: str) -> str:
                    agent = st.session_state.chat_engine
                    result = await agent.run(q, ctx=None, stream=False)
                    return str(result)

                loop = asyncio.new_event_loop()
                try:
                    asyncio.set_event_loop(loop)
                    answer = loop.run_until_complete(run_teach(prompt))
                finally:
                    loop.close()

                st.markdown(answer)

    st.session_state.history.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()

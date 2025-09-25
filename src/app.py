import os
import asyncio
import streamlit as st

import torch
from health import run_health_check

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
    # examples – ensure these exist/are enabled for your Groq account
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


def _init_state():
    if "chat_engine" not in st.session_state:
        st.session_state.chat_engine = None
    if "history" not in st.session_state:
        st.session_state.history = []
    if "last_file_name" not in st.session_state:
        st.session_state.last_file_name = None
    if "last_mode" not in st.session_state:
        st.session_state.last_mode = None
    if "last_model" not in st.session_state:
        st.session_state.last_model = None


def _maybe_reset_history(new_file_name: str, mode: str, model_name: str):
    if (
        st.session_state.last_file_name != new_file_name
        or st.session_state.last_mode != mode
        or st.session_state.last_model != model_name
    ):
        st.session_state.history = []
        st.session_state.chat_engine = None  # force rebuild on change
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
        "Upload a PDF, pick a mode — **Summarize**, **Explain**, or **Teach** — then chat with your doc. "
        "Teach mode uses **Groq** models for tool/function calls; Summarize/Explain use **HF** local models."
    )

    # ---------- Sidebar ----------
    st.sidebar.header("Configuration")
    mode = st.sidebar.radio("Interaction Mode", ["summarize", "explain", "teach"], index=0)

    # Constrain model list by mode
    if mode == "teach":
        llm_model = st.sidebar.selectbox("Groq LLM", GROQ_MODELS, index=GROQ_MODELS.index(DEFAULT_GROQ_MODEL))
        # Teach requires Groq API
        if not os.getenv("GROQ_API_KEY"):
            st.sidebar.error("GROQ_API_KEY is missing. Set it in your environment or .env file.")
    else:
        llm_model = st.sidebar.selectbox("HF (local) LLM", HF_MODELS, index=HF_MODELS.index(DEFAULT_HF_MODEL))

    # Embeddings are HF for all modes (vector building)
    embed_model = st.sidebar.text_input("Embedding Model", DEFAULT_EMBED_MODEL)

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    # ---------- Build chat engine (on upload or when config changed) ----------
    if uploaded_file is not None:
        _maybe_reset_history(uploaded_file.name, mode, llm_model)

        if st.session_state.chat_engine is None:
            with st.spinner("Indexing document and preparing engine..."):
                if mode == "summarize":
                    sum_index = build_sum_index_from_uploads(
                        file=uploaded_file,
                        embed_model=embed_model,
                    )
                    vec_index = None
                else:
                    vec_index = build_vec_index_from_uploads(
                        file=uploaded_file,
                        embed_model=embed_model,
                    )
                    sum_index = None

                # build_chat_engine already picks Groq for teach and HF for others in your backend
                st.session_state.chat_engine = build_chat_engine(
                    mode=mode,
                    vec_index=vec_index,
                    sum_index=sum_index,
                    llm_model=llm_model,
                )

    # ---------- Chat UI ----------
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
                # TEACH (agent workflow via Groq)
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

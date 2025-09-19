import os
import sys
import time
import torch
import logging
from typing import Optional

import nest_asyncio
import streamlit as st

from health import run_health_checks
from rag import build_index_from_uploads, run_query, build_llm

# Prevent path errors for Torch
try:
    torch.classes.__path__ = []  # type: ignore[attr-defined]
except Exception:
    pass

def main():

    st.set_page_config(page_title="AI Document Assistant", page_icon=":books:", layout="wide")
    nest_asyncio.apply()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger = logging.getLogger("streamlit-app")

    st.title("AI Document Assistant")
    st.caption("Upload PDF, select your models, and ask questions. Embeddings are built at runtime.")

    LLM_OPTIONS = [
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "HuggingFaceH4/zephyr-7b-beta",
        "openai/gpt-oss-20b"
    ]

    EMBED_OPTIONS = [
        "BAAI/bge-small-en-v1.5",
        "BAAI/bge-base-en-v1.5",
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
    ]

# result = run_health_checks(
#     min_python="3.11",
#     max_python="3.13",
#     qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
#     qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
#     skip_torch_check=False,  # set True if torch not installed in your app container
# )

# if result["overall"] == "ok":
#     st.success("Health: OK")
# else:
#     st.error("Health: FAILED")
#     with st.expander("Details"):
#         for name, info in result["checks"].items():
#             st.write(f"**{name}** — {info['status'].upper()}: {info['detail']}")

    with st.sidebar:
        st.header("Settings")

        llm_model = st.selectbox("Chat LLM", options=LLM_OPTIONS, index=0)
        embed_model = st.selectbox("Embedding Model", options=EMBED_OPTIONS, index=0)
        # response_mode = st.radio("Response mode", options=RESPONSE_MODES, index=0, help="compact = one-shot; refine = iterative")
        top_k = st.slider("Top-K retrieved chunks", min_value=1, max_value=10, value=3, step=1)

        st.markdown("---")
        if st.button("Clear cache"):
            st.cache_resource.clear()
            st.success("Cache cleared. Re-run to rebuild the index.")
    
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"], accept_multiple_files=False)
    question = st.text_input("Your question", placeholder="e.g., What penalties are specified in the document?")

    run_clicked = st.button("Run")

    @st.cache_resource(show_spinner=False)
    def cached_llm(model_name: str):
        return build_llm(model_name)


    @st.cache_resource(show_spinner=False)
    def _cached_build_index(file_bytes: bytes, embed_model: str):
        # Pass raw bytes; rag.build_index_from_uploads will handle temp path etc.
        return build_index_from_uploads(
            file=file_bytes,
            embed_model=embed_model,
        )

    if run_clicked:
        if not uploaded_file:
            st.error("Please upload a PDF.")
            st.stop()
        if not question.strip():
            st.error("Please enter a question.")
            st.stop()

        file_bytes = uploaded_file.getvalue()

        with st.spinner("Building vectors and index…"):
            index = _cached_build_index(file_bytes=file_bytes, embed_model=embed_model)

        with st.spinner("Building and cachiing llm_model…"):
            llm_obj = cached_llm(llm_model)

        with st.spinner("Answering…"):
            t0 = time.time()
            resp = run_query(index=index, llm=llm_obj, question=question, response_mode="compact", top_k=top_k)
            elapsed = time.time() - t0

        st.success(f"Done in {elapsed:.2f}s")
        st.subheader("Answer")
        st.write(resp.response)

        # Show sources if available
        if getattr(resp, "source_nodes", None):
            st.subheader("Sources")
            for i, n in enumerate(resp.source_nodes, start=1):
                with st.expander(f"Source {i} (score {getattr(n, 'score', None)})"):
                    text = n.node.get_text() if hasattr(n, "node") else getattr(n, "text", "")
                    st.write(text[:1000] + ("…" if len(text) > 1000 else ""))
                    if hasattr(n, "node") and hasattr(n.node, "metadata"):
                        st.json(n.node.metadata)


if __name__ == "__main__":
    main()
import os
import io
import sys
import hashlib
import tempfile
import logging
from functools import lru_cache
from typing import List, Tuple, Optional, Union

import torch
import nest_asyncio

from llama_index.core import (
    Settings,
    VectorStoreIndex,
    SimpleDirectoryReader,
)
from llama_parse import LlamaParse
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import MetadataMode
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM


from llama_index.llms.huggingface import HuggingFaceLLM

from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.storage.storage_context import StorageContext

from core.utils import _choose_device, _choose_dtype

nest_asyncio.apply()
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("rag-core")

# data_file = "./data/Nigeria-Tax-Act-2025.pdf"

SYSTEM_PROMPT = """
You are a document Q&A assistant. Your goal is to answer the question the user asks you as
accurately as possible based on the context (document information) and mode instruction provided to you.
""".strip()

TEXT_QA_TEMPLATE = PromptTemplate(
    """You are a document Q&A assistant. Answer the user's question using ONLY the context below.
If the answer is not in the context, say: "I don't know based on the provided documents."
Be concise and do not repeat these instructions.

Context:
{context_str}

Question:
{query_str}
"""
)

REFINE_TEMPLATE = PromptTemplate(
    """We have an existing draft answer:
{existing_answer}

Here is additional context:
{context_msg}

Refine the answer only if this new context adds relevant information. If not, return the original answer.
Keep the final answer concise.
"""
)

@lru_cache(maxsize=3)
def build_llm(model_name: str) -> HuggingFaceLLM:
    """Create a chat-tuned HF model for QA."""
    device = _choose_device()
    dtype = _choose_dtype(device)
        
    device_map = device if device in {"cuda", "mps", "cpu"} else "cpu"

    return HuggingFaceLLM(
        context_window=2048,
        max_new_tokens=256,
        tokenizer_kwargs={"padding_side": "left"},
        system_prompt=SYSTEM_PROMPT,
        query_wrapper_prompt=TEXT_QA_TEMPLATE,
        tokenizer_name=model_name,
        model_name=model_name,
        device_map=device_map,
        model_kwargs={
            "dtype": dtype,
            "low_cpu_mem_usage": True,
        },
        # generate_kwargs={"do_sample": True, "temperature": 0.7, "top_p": 0.9},
    )

def build_embed(model_name: str) -> HuggingFaceEmbedding:
    return HuggingFaceEmbedding(model_name=model_name)


def _connect_qdrant(
    host: str = "localhost",
    port: int = 6333,
    collection_name: str = "ai-document-assistant",
) -> StorageContext:
    """Connect to Qdrant at host/port; fallback to :memory: if unreachable."""
    try:
        client = QdrantClient(host=host, port=port, timeout=10.0, retries=3)
        client.get_locks()  
        logger.info(f"Qdrant reachable at {host}:{port}")
    except Exception:
        logger.warning(f"Qdrant not reachable at {host}:{port} — using in-memory mode.")
        client = QdrantClient(location=":memory:")

    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
    return StorageContext.from_defaults(vector_store=vector_store)

def _ensure_path_from_single_file(
    file: Union[str, bytes, io.BytesIO],
    name_hint: str = "upload.pdf",
) -> str:
    """
    Accept a single file as a filesystem path, raw bytes, or a file-like object (e.g., Streamlit UploadedFile).
    Returns a path on disk for loaders like SimpleDirectoryReader.
    """
    if isinstance(file, str):
        if not os.path.isfile(file):
            raise FileNotFoundError(f"File path not found: {file}")
        return file

    # Convert bytes / file-like to bytes
    if isinstance(file, bytes):
        data = file
    elif isinstance(file, io.BytesIO):
        data = file.getvalue()
    else:
        # Last resort: try .read()
        try:
            data = file.read()  # type: ignore[attr-defined]
        except Exception as e:
            raise TypeError("Unsupported file type; pass a path, bytes, or file-like object.") from e

    tmpdir = tempfile.mkdtemp(prefix="rag_one_")
    out_path = os.path.join(tmpdir, name_hint or "upload.pdf")
    with open(out_path, "wb") as f:
        f.write(data)
    return out_path

def build_index_from_uploads(
    file: Union[str, bytes, io.BytesIO],
    embed_model: str,
    *,
    host: str = "localhost",
    port: int = 6333,
    collection_name: str = "ai-document-assistant",
    chunk_size: int = 512,
    chunk_overlap: int = 100,
) -> VectorStoreIndex:
    """
    Build a VectorStoreIndex for a single uploaded PDF (or path) and persist vectors to Qdrant.

    Args:
      file: path OR bytes/BytesIO for a single PDF (Streamlit's UploadedFile is fine).
      embed_model: HF embedding model id.
      host/port/collection_name/chunk_* have sensible defaults and need not be specified.
    """
    # Embedding model for indexing
    embed = build_embed(embed_model)

    # Vector store
    storage_context = _connect_qdrant(host=host, port=port, collection_name=collection_name)

    # Materialize a path for the single file
    file_path = _ensure_path_from_single_file(file)

    # Load docs with a structured PDF parser first; fallback to simple reader
    try:
        docs = LlamaParse(result_type="text").load_data(file_path)
        logger.info("Loaded documents via LlamaParse")
    except Exception as e:
        logger.warning(f"LlamaParse failed ({e}); falling back to SimpleDirectoryReader")
        docs = SimpleDirectoryReader(input_files=[file_path]).load_data()
    # Strip noisy metadata at the document level
    for d in docs:
        if hasattr(d, "metadata") and isinstance(d.metadata, dict):
            for k in [
                "file_path",
                "page_label",
                "file_name",
                "filename",
                "document_id",
            ]:
                d.metadata.pop(k, None)
    logger.info(f"Loaded {len(docs)} document(s) from: {os.path.basename(file_path)}")

    # Chunk
    splitter = SentenceSplitter(separator=" ", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.get_nodes_from_documents(docs)
    # Strip metadata at the node level as well (ultimate guard)
    for n in nodes:
        if hasattr(n, "metadata") and isinstance(n.metadata, dict):
            n.metadata.clear()

    # Index (vectors persisted to Qdrant); pass embed model explicitly
    index = VectorStoreIndex(nodes, embed_model=embed, storage_context=storage_context)
    logger.info(f"Indexed {len(nodes)} nodes")
    return index


def run_query(
    index: VectorStoreIndex,
    question: str,
    *,
    llm: HuggingFaceLLM,
    response_mode: str = "compact",
    top_k: int = 3,
):
    """
    Query the index with either compact (one-shot) or refine (iterative) synthesis.
    LLM is built at query time from llm_model.
    
    llm_model: HF model id for chat LLM (used at query time, but included for cache keys/UI symmetry).
    """
    
    # Retrieve wider, then rerank down to top_k for better relevance
    retrieve_k = max(top_k * 2, 6)

    reranker = SentenceTransformerRerank(
        model="BAAI/bge-reranker-base",
        top_n=top_k,
    )

    kwargs = dict(
        similarity_top_k=retrieve_k,
        text_qa_template=TEXT_QA_TEMPLATE,
        metadata_mode=MetadataMode.NONE,
    )
    if response_mode == "compact":
        kwargs.update(response_mode="compact")
    elif response_mode == "refine":
        kwargs.update(response_mode="refine", refine_template=REFINE_TEMPLATE)

    qe = index.as_query_engine(llm=llm, node_postprocessors=[reranker], **kwargs)
    logger.info("Query engine created")

    response = qe.query(question)
    logger.info(f"Response: {response.response}")
    return response

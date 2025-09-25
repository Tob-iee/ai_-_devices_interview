import os
import io
import sys
import tempfile
import logging
import argparse
from functools import lru_cache
import traceback
from typing import Union

import dotenv
import asyncio
from transformers import AutoTokenizer

from llama_index.core import (
    Settings,
    VectorStoreIndex,
    SimpleDirectoryReader,
)

from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core.tools import QueryEngineTool 
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser

from llama_index.core.tools import RetrieverTool

from llama_index.llms.groq import Groq
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_parse import LlamaParse

from llama_index.core.chat_engine import ContextChatEngine, CondensePlusContextChatEngine

from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.storage.storage_context import StorageContext

from llama_index.tools.duckduckgo import DuckDuckGoSearchToolSpec
from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow

from llama_index.core.workflow import Context

from core.prompts import (
    SYSTEM_PROMPT,
    TEXT_QA_TEMPLATE,
    CUSTOM_SUMMARIZE_CHAT_HISTORY,
)

from core.utils import _choose_device, _choose_dtype, format_response_payload

dotenv.load_dotenv()
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("rag-core")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLAMA_PARSE_API_KEY = os.getenv("LLAMA_PARSE_API_KEY")

QHOST = os.getenv("QDRANT_HOST", "localhost")
QPORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")

@lru_cache(maxsize=3)
def build_llm(model_name: str) -> HuggingFaceLLM:
    device = _choose_device()
    dtype = _choose_dtype(device)
    device_map = device if device in {"cuda", "mps", "cpu"} else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_length = tokenizer.model_max_length
    print(f"The maximum sequence length is: {max_length}")

    return HuggingFaceLLM(
        context_window=2048,
        # max_new_tokens=256,
        tokenizer_kwargs={
            "padding_side": "left",
            "model_max_length": max_length,  
        },
        system_prompt=SYSTEM_PROMPT,
        query_wrapper_prompt=TEXT_QA_TEMPLATE,
        tokenizer_name=model_name,
        model_name=model_name,
        device_map=device_map,
        model_kwargs={"dtype": dtype, "low_cpu_mem_usage": True},
        generate_kwargs={"do_sample": False, "repetition_penalty": 1.05},
    )

def call_groq_llm(model_name: str, api_key: str) -> Groq:
    return Groq(model=model_name, api_key=api_key)

def build_hf_embeddings(embed_model: str) -> None:
    """Build Huggingface embeddings."""
    return HuggingFaceEmbedding(model_name=embed_model)

def connect_qdrant(
    url: str | None = None,
    api_key: str | None = None,
    host: str | None = None,
    port: int | None = None,
    collection_name: str = "ai-document-assistant",
) -> StorageContext:
    """Connect to Qdrant cloud server or fall back to in-memory mode; return storage context."""

    url = url or os.getenv("QDRANT_URL")
    api_key = api_key or os.getenv("QDRANT_API_KEY")

    try:
        if url:
            client = QdrantClient(url=url, api_key=api_key)
            _ = client.get_collections()
            logger.info(f"Qdrant Cloud reachable at {url}")
        else:
            host = host or os.getenv("QDRANT_HOST", "localhost")
            port = int(port or os.getenv("QDRANT_PORT", "6333"))
            client = QdrantClient(host=host, port=port)
            _ = client.get_collections()
            logger.info(f"Qdrant reachable at {host}:{port}")
    except Exception as e:
        logger.warning(f"Qdrant not reachable ({e}) — using in-memory mode.")
        client = QdrantClient(location=":memory:")

    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
    return StorageContext.from_defaults(vector_store=vector_store)

def _ensure_path_from_single_file(
    file: Union[str, bytes, io.BytesIO], name_hint: str = "upload.pdf"
) -> str:
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
        try:
            data = file.read()  # type: ignore[attr-defined]
        except Exception as e:
            raise TypeError("Unsupported file type; pass a path, bytes, or file-like object.") from e

    tmpdir = tempfile.mkdtemp(prefix="rag_one_")
    out_path = os.path.join(tmpdir, name_hint or "upload.pdf")
    with open(out_path, "wb") as f:
        f.write(data)
    return out_path

def build_vec_index_from_uploads(
    file: Union[str, bytes, io.BytesIO],
    embed_model: str,
    *,
    url: str | None = None,
    api_key: str | None = None,
    host: str = "localhost",
    port: int = 6333,
    collection_name: str = "ai-document-assistant-vectors",
    chunk_size: int = 350,
    chunk_overlap: int = 60,
) -> VectorStoreIndex:
    """
    Build or load a VectorStoreIndex for a single uploaded PDF and persist vectors to Qdrant.
    """
    embed = build_hf_embeddings(embed_model)
    Settings.embed_model = embed

    Settings.chunk_size = chunk_size
    Settings.chunk_overlap = chunk_overlap

    storage_context = connect_qdrant(
        url=url,
        api_key=api_key,
        host=host,
        port=port,
        collection_name=collection_name,
    )
    vector_store = storage_context.vector_store

    file_path = _ensure_path_from_single_file(file)
    file_name = os.path.basename(file_path)

    # Check if file has already been indexed by searching for its metadata
    try:
        # Query Qdrant for points with metadata 'file_name'
        results = vector_store._client.scroll(
            collection_name=collection_name,
            filter={"must": [{"key": "file_name", "match": {"value": file_name}}]},
            limit=1,
        )
        already_indexed = len(results[1]) > 0
    except Exception:
        already_indexed = False

    if already_indexed:
        logger.info(f"File '{file_name}' already indexed. Loading existing index.")
        index = VectorStoreIndex.from_vector_store(vector_store)
        return index


    # Parse and index the document if not already indexed
    try:
        parser = LlamaParse(result_type="text", api_key=LLAMA_PARSE_API_KEY)
        docs = SimpleDirectoryReader(input_files=[file_path],file_extractor={".pdf": parser},).load_data()
        logger.info("Parsed with LlamaParse")
    except Exception as e:
        logger.warning(f"LlamaParse failed ({e}); falling back to SimpleDirectoryReader")
        docs = SimpleDirectoryReader(input_files=[file_path]).load_data()

    # Add file_name as metadata to each document
    for d in docs:
        d.metadata = d.metadata or {}
        d.metadata["file_name"] = file_name

    semantic_splitter = SemanticSplitterNodeParser(
        buffer_size=1,
        breakpoint_percentile_threshold=95,
        embed_model=embed
    )
    nodes = semantic_splitter.get_nodes_from_documents(docs)
    logger.info(f"Indexed {len(nodes)} nodes into collection '{collection_name}'")


    index = VectorStoreIndex.from_documents(nodes, storage_context=storage_context)
    logger.info(f"Upserted {len(nodes)} chunks into '{collection_name}' (doc={file_name})")
    return index

def build_sum_index_from_uploads(
    file: Union[str, bytes, io.BytesIO],
    embed_model: str,
    *,
    url: str | None = None,
    api_key: str | None = None,
    host: str = "localhost",
    port: int = 6333,
    collection_name: str = "ai-document-assistant-summary",
    chunk_size: int = 350,
    chunk_overlap: int = 60,
) -> VectorStoreIndex:
    """
    Build a summary on index for a single uploaded PDF.
    """

    embed = build_hf_embeddings(embed_model)
    Settings.embed_model = embed

    Settings.chunk_size = chunk_size
    Settings.chunk_overlap = chunk_overlap

    storage_context = connect_qdrant(
        url=url,
        api_key=api_key,
        host=host,
        port=port,
        collection_name=collection_name,
    )
    summary_store = storage_context.vector_store

    file_path = _ensure_path_from_single_file(file)
    file_name = os.path.basename(file_path)


    # Check if file has already been indexed by searching for its metadata
    try:
        # Query Qdrant for points with metadata 'file_name'
        results = summary_store._client.scroll(
            collection_name=collection_name,
            filter={"must": [{"key": "file_name", "match": {"value": file_name}}]},
            limit=1,
        )
        already_indexed = len(results[1]) > 0
    except Exception:
        already_indexed = False

    if already_indexed:
        logger.info(f"File '{file_name}' already indexed. Loading existing index.")
        index = VectorStoreIndex.from_vector_store(summary_store)
        return index


    # Parse and index the document if not already indexed
    try:
        parser = LlamaParse(result_type="text", api_key=LLAMA_PARSE_API_KEY)
        docs = SimpleDirectoryReader(input_files=[file_path], file_extractor={".pdf": parser}).load_data()
        logger.info("Parsed with LlamaParse")
    except Exception as e:
        logger.warning(f"LlamaParse failed ({e}); falling back to SimpleDirectoryReader")
        docs = SimpleDirectoryReader(input_files=[file_path]).load_data()

    for d in docs:
        d.metadata = d.metadata or {}
        d.metadata["file_name"] = file_name

    # Chunk
    splitter = SentenceSplitter(separator=" ", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.get_nodes_from_documents(docs)
    logger.info(f"Loaded and split {len(docs)} document(s) into {len(nodes)} nodes from: {file_name}")


    # Summary Index builds its own tree over the docs
    index = VectorStoreIndex.from_documents(nodes, storage_context=storage_context)
    logger.info(f"Upserted {len(nodes)} chunks into '{collection_name}' (doc={file_name})")
    return index

def build_chat_engine(
    *,
    mode: str,
    vec_index: Union[VectorStoreIndex, None],
    sum_index: Union[VectorStoreIndex, None],
    llm_model: str,
    top_k: int = 3,
):
    """
    Build exactly ONE chat engine for the selected mode.
    - summarize -> VectorStoreIndex (compact synthesis, summarization prompt)
    - explain/teach -> VectorStoreIndex (refine synthesis, QA/refine templates)
    """
    
    memory = ChatMemoryBuffer.from_defaults(token_limit=500)

    if mode == "summarize":
        if sum_index is None:
            raise ValueError("Summary Index not provided for summarize mode.")
        
        hf_os_llm = build_llm(llm_model)
        Settings.llm = hf_os_llm
        
        summary_retriever = sum_index.as_retriever() or VectorIndexAutoRetriever(sum_index)

        summary_tool = RetrieverTool.from_defaults(
            retriever=summary_retriever,
            description=(
                "Useful for retrieving context for summarizing of documents."
            )
        )

        retriever = summary_tool.retriever
        summary_context_chat_engine = ContextChatEngine.from_defaults(
        retriever=retriever,
        chat_history= CUSTOM_SUMMARIZE_CHAT_HISTORY,
        system_prompt= SYSTEM_PROMPT,
        memory=memory,
        llm=hf_os_llm
        )

        return summary_context_chat_engine

    # explain => vector index
    elif mode == "explain":
        if vec_index is None:
            raise ValueError("VectorStoreIndex not provided for explain mode.")
        
        hf_os_llm = build_llm(llm_model)
        Settings.llm = hf_os_llm

        explain_retriever = vec_index.as_retriever() or VectorIndexAutoRetriever(vec_index)

        explain_tool = RetrieverTool.from_defaults(
            retriever=explain_retriever,
            description=(
                "Useful for retrieving context for explaining of documents."
            )
        )
        retriever = explain_tool.retriever
        explain_context_chat_engine = CondensePlusContextChatEngine.from_defaults(
        retriever=retriever,
        # chat_history= CUSTOM_EXPLAIN_CHAT_HISTORY,
        system_prompt= SYSTEM_PROMPT,
        memory=memory,
        llm=hf_os_llm
        )

        return explain_context_chat_engine

    # teach => vector index
    elif mode == "teach":
        if vec_index is None:
            raise ValueError("VectorStoreIndex not provided for teach mode.")
        llm = call_groq_llm(model_name=llm_model, api_key=GROQ_API_KEY) #gpt-4o-mini
        Settings.llm = llm

        teach_engine = vec_index.as_query_engine() 
        teach_engine_tool = QueryEngineTool.from_defaults(
            query_engine=teach_engine,
            description=(
                "Useful for retrieving contents from the documents for teaching purposes based on the user's query."
            )
        )
        web_tools = DuckDuckGoSearchToolSpec().to_tool_list()
        retriever_agent = FunctionAgent(
            name="retriever",
            description="Manages data retrieval from the documents and use that as context for teaching purposes.",
            system_prompt="You are a retrieval assistant.",
            tools=[teach_engine_tool, *web_tools],
            llm=llm,
            can_handoff_to=["web_search", "final_answer"],
        )
        agent_workflow = AgentWorkflow(
            agents=[retriever_agent],
            root_agent="retriever",
        )
    return agent_workflow

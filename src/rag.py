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
    SummaryIndex,
    VectorStoreIndex,
    DocumentSummaryIndex,
    SimpleDirectoryReader,
)

from llama_index.core.retrievers import VectorIndexAutoRetriever


from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import MetadataMode
from llama_index.core.tools import QueryEngineTool #RetrieverQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser

from llama_index.core.retrievers import RouterRetriever
from llama_index.core.selectors import PydanticSingleSelector
from llama_index.core.tools import RetrieverTool

from llama_index.llms.groq import Groq
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_parse import LlamaParse

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.chat_engine import ContextChatEngine, CondensePlusContextChatEngine, CondenseQuestionChatEngine

from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.storage.storage_context import StorageContext

from llama_index.tools.duckduckgo import DuckDuckGoSearchToolSpec
from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow, ReActAgent, AgentStream, ToolCallResult

from llama_index.core.workflow import Context

from core.prompts import (
    SYSTEM_PROMPT,
    TEXT_QA_TEMPLATE,
    SUMMARIZE_SYSTEM,
    EXPLAIN_SYSTEM,
    TEACH_SYSTEM,
    CUSTOM_SUMMARIZE_PROMPT,
    CUSTOM_SUMMARIZE_CHAT_HISTORY,
)

from core.utils import _choose_device, _choose_dtype, format_response_payload

dotenv.load_dotenv()
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("rag-core")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

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
    host: str = "localhost",
    port: int = 6333,
    collection_name: str = "ai-document-assistant",
) -> StorageContext:
    """Connect to Qdrant or fall back to :memory: mode; return storage context."""


    try:
        client = QdrantClient(host=host, port=port, timeout=10.0, retries=3)
        # simple ping
        _ = client.get_collections()
        logger.info(f"Qdrant reachable at {host}:{port}")
    except Exception:
        logger.warning(f"Qdrant not reachable at {host}:{port} — using in-memory mode.")
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
        parser = LlamaParse(result_type="markdown")
        docs = SimpleDirectoryReader(
            input_files=[file_path],
            file_extractor={".pdf": parser},
        ).load_data()
        logger.info("Parsed with LlamaParse")
    except Exception as e:
        logger.warning(f"LlamaParse failed ({e}); falling back to SimpleDirectoryReader")
        docs = SimpleDirectoryReader(input_files=[file_path]).load_data()

    logger.info(f"Loaded {len(docs)} document(s) from: {file_name}")

    # Add file_name as metadata to each document
    for doc in docs:
        doc.metadata = doc.metadata or {}
        doc.metadata["file_name"] = file_name

    semantic_splitter = SemanticSplitterNodeParser(
        buffer_size=1,
        breakpoint_percentile_threshold=95,
        embed_model=embed
    )
    nodes = semantic_splitter.get_nodes_from_documents(docs)

    index = VectorStoreIndex.from_documents(
        nodes,
        storage_context=storage_context,
    )
    logger.info(f"Indexed {len(nodes)} nodes into collection '{collection_name}'")
    return index

def build_sum_index_from_uploads(
    file: Union[str, bytes, io.BytesIO],
    embed_model: str,
    *,
    host: str = "localhost",
    port: int = 6333,
    collection_name: str = "ai-document-assistant-summary",
    chunk_size: int = 100,
    chunk_overlap: int = 20,
) -> VectorStoreIndex:
    """
    Build a SummaryIndex for a single uploaded PDF.
    """

    embed = build_hf_embeddings(embed_model)
    Settings.embed_model = embed

    Settings.chunk_size = chunk_size
    Settings.chunk_overlap = chunk_overlap

    storage_context = connect_qdrant(
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
        # index = SummaryIndex(summary_store, show_progress=True, storage_context=storage_context)
        index = VectorStoreIndex.from_vector_store(summary_store)
        return index

    # Parse and index the document if not already indexed
    try:
        parser = LlamaParse(result_type="markdown")
        docs = SimpleDirectoryReader(
            input_files=[file_path], file_extractor={".pdf": parser},
        ).load_data()
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

    # SummaryIndex builds its own tree over the docs
    index = VectorStoreIndex.from_documents(nodes, storage_context=storage_context)
    logger.info(f"SummaryIndexed {len(nodes)} nodes into collection '{collection_name}'")
    return index

def build_chat_engine(
    *,
    mode: str,
    vec_index: Union[VectorStoreIndex, None],
    sum_index: Union[SummaryIndex, None],
    llm_model: str,
    top_k: int = 3,
):
    """
    Build exactly ONE chat engine for the selected mode.
    - summarize -> SummaryIndex (compact synthesis, summarization prompt)
    - explain/teach -> VectorStoreIndex (refine synthesis, QA/refine templates)
    """
    hf_os_llm = build_llm(llm_model)
    llm = call_groq_llm(model_name="openai/gpt-oss-20b", api_key=GROQ_API_KEY) #gpt-4o-mini

    # Settings.llm = llm
    
    memory = ChatMemoryBuffer.from_defaults(token_limit=500)

    if mode == "summarize":
        if sum_index is None:
            raise ValueError("SummaryIndex not provided for summarize mode.")

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
        # prefix_messages=prefix_messages,
        # node_postprocessors=node_postprocessors,
        # context_template=CUSTOM_SUMMARIZE_PROMPT,
        # context_refine_template=context_refine_template,
        llm=hf_os_llm
        )

        return summary_context_chat_engine

    # explain / teach => vector index
    elif mode == "explain":
        if vec_index is None:
            raise ValueError("VectorStoreIndex not provided for explain mode.")

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
        # prefix_messages=prefix_messages,
        # node_postprocessors=node_postprocessors,
        # context_template=CUSTOM_SUMMARIZE_PROMPT,
        # context_refine_template=context_refine_template,
        llm=hf_os_llm
        )

        return explain_context_chat_engine

    elif mode == "teach":
        if vec_index is None:
            raise ValueError("VectorStoreIndex not provided for teach mode.")
        llm = call_groq_llm(model_name="openai/gpt-oss-20b", api_key=GROQ_API_KEY) #gpt-4o-mini
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


async def main():
    parser = argparse.ArgumentParser(description="RAG over a single PDF using LlamaIndex + Qdrant")
    parser.add_argument("--data-file", type=str, required=True, help="Path to a PDF to index")
    parser.add_argument("--llm-model", type=str, default="meta-llama/Llama-3.2-3B-Instruct") # meta-llama/Llama-3.2-3B-Instruct meta-llama/Llama-3.1-8B TinyLlama/TinyLlama-1.1B-Chat-v1.0 Qwen/Qwen1.5-1.8B-Chat meta-llama/Llama-3.1-8B HuggingFaceH4/zephyr-7b-beta  openai/gpt-oss-20b
    parser.add_argument("--embed-model", type=str, default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--qdrant-host", type=str, default="localhost")
    parser.add_argument("--qdrant-port", type=int, default=6333)

    # separate collections (avoid collisions if you later build both)
    parser.add_argument("--vec-collection", type=str, default="ai-doc-assistant-vectors")
    parser.add_argument("--sum-collection", type=str, default="ai-doc-assistant-summary")

    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--mode", type=str, choices=["summarize", "explain", "teach"],
                        default="summarize", help="Select interaction mode.")
    args = parser.parse_args()

    if not os.path.isfile(args.data_file):
        logger.error(f"Data file not found: {args.data_file}")
        sys.exit(1)

    try:
        mode = args.mode.lower()

        # Build only the index you need for the selected mode
        vec_index = None
        sum_index = None

        if mode == "summarize":
            sum_index = build_sum_index_from_uploads(
                file=args.data_file,
                embed_model=args.embed_model,
                host=args.qdrant_host,
                port=args.qdrant_port,
                collection_name=args.sum_collection,
            )

            
        else:  # explain or teach => vector
            vec_index = build_vec_index_from_uploads(
                file=args.data_file,
                embed_model=args.embed_model,
                host=args.qdrant_host,
                port=args.qdrant_port,
                collection_name=args.vec_collection,
            )

        chat_engine_tool = build_chat_engine(
            mode=mode,
            vec_index=vec_index,
            sum_index=sum_index,
            llm_model=args.llm_model,
            top_k=args.top_k,
        )

        print(chat_engine_tool)


        if mode == "summarize" or mode == "explain":

            # Build exactly one chat engine for the chosen mode
            chat_engine= build_chat_engine(
                mode=mode,
                vec_index=vec_index,
                sum_index=sum_index,
                llm_model=args.llm_model,
                top_k=args.top_k,
            )

            print(f"Interactive chat started. Mode = {args.mode}. Type 'exit' or 'quit' to end.")
            while True:
                question = input("\nUser: ")
                if question.lower() in {"exit", "quit"}:
                    print("Exiting chat.")
                    break

                chat_resp = chat_engine.chat(question)
                payload = format_response_payload(chat_resp, max_citations=5)
                print(f"Assistant: {payload['answer']}")

                if payload["citations"]:
                    print("CITATIONS:")
                    for i, c in enumerate(payload["citations"], 1):
                        print(f"[{i}] {c}")

        elif mode == "teach":
            # Build exactly one chat engine for the chosen mode
            agent = build_chat_engine(
                mode=mode,
                vec_index=vec_index,
                sum_index=sum_index,
                llm_model=args.llm_model,
                top_k=args.top_k,
            )

            while True:
                # avoid blocking the event loop with input()
                question = input("\nUser: ")
                # question = await asyncio.to_thread(input, "\nUser: ")
                if question.lower() in {"exit", "quit"}:
                    print("Exiting teaching agent.")
                    break

                # >>> use the async API
                ctx = Context(agent)
                agent_resp = await agent.run(question, ctx=ctx, stream=True)

                print(agent_resp)

    except Exception as e:
        logger.error(f"Error running RAG: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())



# explain the implication of the new tax law for an employee

# what are the key points of the document

# based on the new tax law, give me a strategy on how an employee who earns 800,000 NGN monthly can adjust their financial planning and make the most of their income while complying with the new regulations


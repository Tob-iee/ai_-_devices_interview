import os
import io
import sys
import tempfile
import logging
import argparse
from functools import lru_cache
from typing import Union

import dotenv

from llama_index.core import (
    Settings,
    VectorStoreIndex,
    SimpleDirectoryReader,
)
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import MetadataMode
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser

from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.fastembed import FastEmbedEmbedding

from llama_parse import LlamaParse

from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.storage.storage_context import StorageContext

from core.utils import _choose_device, _choose_dtype, format_response_payload


dotenv.load_dotenv()
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("rag-core")

SYSTEM_PROMPT = (
    "You are a document Q&A assistant. Answer ONLY from the provided context. "
    "If the answer is not in the context, say: \"I don't know based on the provided documents.\" "
    "Be concise and do not repeat instructions."
).strip()

TEXT_QA_TEMPLATE = PromptTemplate(
    "We have provided context information below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given this information, answer the question. Do not reference the context or sources/page numbers in your answer, cite exact phrases from the context, or make up answers:\n"
    "{query_str}\n"
)

REFINE_TEMPLATE = PromptTemplate(
    "We have an existing draft answer:\n"
    "{existing_answer}\n\n"
    "Here is additional context:\n"
    "{context_msg}\n\n"
    "Refine the answer ONLY if this new context adds relevant information. "
    "Otherwise, return the original answer. Keep it concise."
)

@lru_cache(maxsize=3)
def build_llm(model_name: str) -> HuggingFaceLLM:
    device = _choose_device()
    dtype = _choose_dtype(device)
    device_map = device if device in {"cuda", "mps", "cpu"} else "cpu"

    return HuggingFaceLLM(
        context_window=2048,
        max_new_tokens=512,
        tokenizer_kwargs={"padding_side": "left"},
        system_prompt=SYSTEM_PROMPT,
        query_wrapper_prompt=TEXT_QA_TEMPLATE,
        tokenizer_name=model_name,
        model_name=model_name,
        device_map=device_map,
        model_kwargs={"dtype": dtype, "low_cpu_mem_usage": True},
        generate_kwargs={"do_sample": False, "repetition_penalty": 1.05},
    )

def build_fast_embeddings(embed_model: str) -> None:
    """Build FastEmbed embeddings."""
    return FastEmbedEmbedding(model_name=embed_model)

def build_hf_embeddings(embed_model: str) -> None:
    """Build Huggingface embeddings."""
    return HuggingFaceEmbedding(model_name=embed_model)

def connect_qdrant(
    embed_model: str,
    host: str = "localhost",
    port: int = 6333,
    collection_name: str = "ai-document-assistant",
) -> StorageContext:
    """Connect to Qdrant or fall back to :memory: mode; return storage context."""

    # embed = build_hf_embeddings(embed_model)
    # Settings.embed_model = embed
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

# def build_index_from_uploads(
#     file: Union[str, bytes, io.BytesIO],
#     embed_model: str,
#     *,
#     host: str = "localhost",
#     port: int = 6333,
#     collection_name: str = "ai-document-assistant",
#     chunk_size: int = 512,
#     chunk_overlap: int = 100,
# ) -> VectorStoreIndex:
#     """
#     Build a VectorStoreIndex for a single uploaded PDF and persist vectors to Qdrant.
#     """
#     # embeddings once
#     # Settings.chunk_size = chunk_size
#     # Settings.chunk_overlap = chunk_overlap

#     # embed = build_fast_embeddings(embed_model)
#     # embed = build_hf_embeddings(embed_model)
#     # Settings.embed_model = embed

#     # vector store
#     # storage_context = connect_qdrant(host=host, port=port, collection_name=collection_name, embed_model=embed_model)

#     # parse + load
#     # file_path = _ensure_path_from_single_file(file)
#     # try:
#     #     parser = LlamaParse(result_type="markdown")
#     #     docs = SimpleDirectoryReader(
#     #         input_files=[file_path],
#     #         file_extractor={".pdf": parser},
#     #     ).load_data()
#     #     logger.info("Parsed with LlamaParse")
#     # except Exception as e:
#     #     logger.warning(f"LlamaParse failed ({e}); falling back to SimpleDirectoryReader")
#     #     docs = SimpleDirectoryReader(input_files=[file_path]).load_data()

#     # logger.info(f"Loaded {len(docs)} document(s) from: {os.path.basename(file_path)}")

#     # chunk
#     # splitter = SentenceSplitter(separator=" ", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#     # nodes = splitter.get_nodes_from_documents(docs)

#     # semantic_splitter = SemanticSplitterNodeParser(buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed)
#     # nodes = semantic_splitter.get_nodes_from_documents(docs)

#     # index (embeddings taken from Settings)
#     # index = VectorStoreIndex(nodes, 
#     #                         #  embed_model=embed, 
#     #                          storage_context=storage_context
#     #                          )

#     # index = VectorStoreIndex.from_documents(
#     #     nodes,
#     #     storage_context=storage_context,
#     # )
#     # index = VectorStoreIndex.from_vector_store(
#     #     vector_store,
#     # Embedding model should match the original embedding model
#     # embed_model=Settings.embed_model
#     # )
#     # index = DocumentSummaryIndex.from_documents(
#     # documents, embed_model=embed_model, llm=llm)
#    # )
#     # logger.info(f"Indexed {len(nodes)} nodes into collection '{collection_name}'")
#     # return index

def build_vec_index_from_uploads(
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
    Build or load a VectorStoreIndex for a single uploaded PDF and persist vectors to Qdrant.
    """
    embed = build_hf_embeddings(embed_model)
    Settings.embed_model = embed

    storage_context = connect_qdrant(
        host=host,
        port=port,
        collection_name=collection_name,
        embed_model=embed_model,
    )
    vector_store = storage_context.vector_store

    file_path = _ensure_path_from_single_file(file)
    file_name = os.path.basename(file_path)

    # Check if file has already been indexed by searching for its metadata
    # try:
        # Query Qdrant for points with metadata 'file_name'
    #     results = vector_store._client.scroll(
    #         collection_name=collection_name,
    #         filter={"must": [{"key": "file_name", "match": {"value": file_name}}]},
    #         limit=1,
    #     )
    #     already_indexed = len(results[1]) > 0
    # except Exception:
    #     already_indexed = False

    # if already_indexed:
    #     logger.info(f"File '{file_name}' already indexed. Loading existing index.")
    #     index = VectorStoreIndex.from_vector_store(vector_store)
    #     return index

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

def run_query(
    index: VectorStoreIndex,
    question: str,
    *,
    llm_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    response_mode: str = "refine",
    top_k: int = 3,
):
    """
    Query the index with compact (one-shot) or refine (iterative) synthesis.
    """
    llm = build_llm(llm_model)
    Settings.llm = llm  

    # Retrieve wider, re-rank to top_k
    retrieve_k = max(top_k * 3, 9)
    reranker = SentenceTransformerRerank(model="BAAI/bge-reranker-base", top_n=top_k)

    qe_kwargs = dict(
        similarity_top_k=retrieve_k,
        text_qa_template=TEXT_QA_TEMPLATE,
        metadata_mode=MetadataMode.NONE,
    )
    if response_mode == "compact":
        qe_kwargs["response_mode"] = "compact"
    elif response_mode == "refine":
        qe_kwargs.update(response_mode="refine", refine_template=REFINE_TEMPLATE)
    else:
        raise ValueError("response_mode must be 'compact' or 'refine'")

    qe = index.as_query_engine(
        vector_store_query_mode="mmr",
        # llm=llm, 
        node_postprocessors=[reranker], 
        **qe_kwargs
    )

    response = qe.query(question)
    return response

def run_chat(
    index: VectorStoreIndex,
    question: str,
    *,
    llm_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    response_mode: str = "refine",
    top_k: int = 3,
):
    """
    Simulate a chat conversation using the chat engine.
    """
    llm = build_llm(llm_model)
    Settings.llm = llm

    retrieve_k = max(top_k * 3, 9)
    reranker = SentenceTransformerRerank(model="BAAI/bge-reranker-base", top_n=top_k)

    qe_kwargs = dict(
        similarity_top_k=retrieve_k,
        text_qa_template=TEXT_QA_TEMPLATE,
        metadata_mode=MetadataMode.NONE,
    )
    if response_mode == "compact":
        qe_kwargs["response_mode"] = "compact"
    elif response_mode == "refine":
        qe_kwargs.update(response_mode="refine", refine_template=REFINE_TEMPLATE)
    else:
        raise ValueError("response_mode must be 'compact' or 'refine'")

    chat_engine = index.as_chat_engine(
        chat_mode="condense_question",
        verbose=True,
        # vector_store_query_mode="mmr",
        # llm=llm,
        # node_postprocessors=[reranker],
        # **qe_kwargs
    )
    response = chat_engine.chat(question)
    return response

    # For streaming response (if supported by the LLM)
    # streaming_response = chat_engine.stream_chat(question)
    # return streaming_response or streaming_response.response_gen

# def main():
#     parser = argparse.ArgumentParser(description="RAG over a single PDF using LlamaIndex + Qdrant")
#     parser.add_argument("--data-file", type=str, required=True, help="Path to a PDF to index")
#     parser.add_argument("--llm-model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
#     parser.add_argument("--embed-model", type=str, default="BAAI/bge-base-en-v1.5")
#     parser.add_argument("--qdrant-host", type=str, default="localhost")
#     parser.add_argument("--qdrant-port", type=int, default=6333)
#     parser.add_argument("--collection", type=str, default="ai-document-assistant")
#     parser.add_argument("--question", type=str, default="Can you give me a good summary of the document?")
#     parser.add_argument("--chunk-size", type=int, default=512)
#     parser.add_argument("--chunk-overlap", type=int, default=100)
#     parser.add_argument("--top-k", type=int, default=3)
#     parser.add_argument("--response-mode", type=str, choices=["compact", "refine"], default="compact")
#     args = parser.parse_args()

#     if not os.path.isfile(args.data_file):
#         logger.error(f"Data file not found: {args.data_file}")
#         sys.exit(1)

#     try:
#         index = build_index_from_uploads(
#             file=args.data_file,
#             embed_model=args.embed_model,
#             host=args.qdrant_host,
#             port=args.qdrant_port,
#             collection_name=args.collection,
#             chunk_size=args.chunk_size,
#             chunk_overlap=args.chunk_overlap,
#         )

#         # resp = run_query(
#         #     index=index,
#         #     question=args.question,
#         #     llm_model=args.llm_model,
#         #     response_mode=args.response_mode,
#         #     top_k=args.top_k,
#         # )
#         # payload = format_response_payload(resp, max_citations=5)
#         # answer, citations = payload["answer"], payload["citations"]
#         # logger.info(f"Response answer: {answer}")

#         # if citations:
#         #     logger.info("CITATIONS:")
#         #     for i, c in enumerate(citations, 1):
#         #         logger.info(f"[{i}] {c}")

#         # Run chat engine instead of query engine
#         chat_resp = run_chat(
#             index=index,
#             question=args.question,
#             llm_model=args.llm_model,
#             response_mode=args.response_mode,
#             top_k=args.top_k,
#         )
#         payload = format_response_payload(chat_resp, max_citations=5)
#         answer, citations = payload["answer"], payload["citations"]
#         logger.info(f"Response answer: {answer}")

#         if citations:
#             logger.info("CITATIONS:")
#             for i, c in enumerate(citations, 1):
#                 logger.info(f"[{i}] {c}")
#     except Exception as e:
#         logger.error(f"Error running RAG: {e}")
#         sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="RAG over a single PDF using LlamaIndex + Qdrant")
    parser.add_argument("--data-file", type=str, required=True, help="Path to a PDF to index")
    parser.add_argument("--llm-model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--embed-model", type=str, default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--qdrant-host", type=str, default="localhost")
    parser.add_argument("--qdrant-port", type=int, default=6333)
    parser.add_argument("--collection", type=str, default="ai-document-assistant")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--response-mode", type=str, choices=["compact", "refine"], default="compact")
    args = parser.parse_args()

    if not os.path.isfile(args.data_file):
        logger.error(f"Data file not found: {args.data_file}")
        sys.exit(1)

    try:
        index = build_vec_index_from_uploads(
            file=args.data_file,
            embed_model=args.embed_model,
            host=args.qdrant_host,
            port=args.qdrant_port,
            collection_name=args.collection,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )

        print("Interactive chat started. Type 'exit' or 'quit' to end.")
        while True:
            question = input("\nUser: ")
            if question.strip().lower() in {"exit", "quit"}:
                print("Exiting chat.")
                break

            chat_resp = run_chat(
                index=index,
                question=question,
                llm_model=args.llm_model,
                response_mode=args.response_mode,
                top_k=args.top_k,
            )
            payload = format_response_payload(chat_resp, max_citations=5)
            answer, citations = payload["answer"], payload["citations"]
            print(f"Assistant: {answer}")

            if citations:
                print("CITATIONS:")
                for i, c in enumerate(citations, 1):
                    print(f"[{i}] {c}")

    except Exception as e:
        logger.error(f"Error running RAG: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

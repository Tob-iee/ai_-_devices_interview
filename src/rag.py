import os
import sys
import logging
import nest_asyncio

from llama_parse import LlamaParse
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.schema import MetadataMode  
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter 

from llama_index.llms.huggingface import HuggingFaceLLM

from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.storage.storage_context import StorageContext

import torch

nest_asyncio.apply()
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

data_file = "./data/Nigeria-Tax-Act-2025.pdf"
openai_llm = "openai/gpt-oss-20b"

system_prompt = """
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


llm = HuggingFaceLLM(
    context_window=2048,
    max_new_tokens=256,
    tokenizer_kwargs={"padding_side": "left"},
    generate_kwargs={  "do_sample": True,"temperature": 0.7,"top_p": 0.9, },
    system_prompt=system_prompt,
    query_wrapper_prompt=TEXT_QA_TEMPLATE,
    tokenizer_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device_map="auto",  
)

client = QdrantClient(
    host="localhost",
    port=6333,
    # For Qdrant Cloud:
    # url="https://<your-cluster>.cloud.qdrant.io",
    # api_key="YOUR_API_KEY",
)

try:
    client = QdrantClient(host="localhost", port=6333, timeout=10.0, retries=3)
    print("Qdrant reachable at localhost:6333")
    # optional quick probe
    client.get_locks()  # lightweight request; throws if server down
except Exception:
    print("Qdrant not reachable at localhost:6333 - using in-memory mode.")
    client = QdrantClient(location=":memory:")


vector_store = QdrantVectorStore(client=client, collection_name="ai-document-assistant")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Set global defaults so we don't have to pass them everywhere
Settings.llm = llm
Settings.embed_model = embed_model

docs = SimpleDirectoryReader(input_files=[data_file]).load_data()

# docs = LlamaParse(result_type="text").load_data(data_file)

logger.info(f"Loaded {len(docs)} document objects from the file")

splitter = SentenceSplitter(
    separator=" ",
    chunk_size=512,
    chunk_overlap=100,
)

# semantic_splitter = SemanticSplitterNodeParser( 
#     chunk_size=512,
#     chunk_overlap=100, 
#     embedding_model=embedding_model, 
#     )

nodes = splitter.get_nodes_from_documents(docs)

index = VectorStoreIndex(
    nodes,
    storage_context=storage_context,
)

logger.info(f"Indexed {len(nodes)} nodes")


query_engine = index.as_query_engine(
    text_qa_template=TEXT_QA_TEMPLATE,
    refine_template=REFINE_TEMPLATE,
    response_mode="refine",
    similarity_top_k=2,
    llm=llm,
)

logger.info("Query engine created")

response = query_engine.query("What is the main topic of the documents?")

print(response.response)
# logger.info(f"Response: {response.response}")


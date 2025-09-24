from llama_index.core.prompts import PromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole

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

SUMMARIZE_SYSTEM = PromptTemplate(
    "Summarize the key points from the context below in 5-8 bullet points.\n"
    "Avoid quoting large passages; use your own words.\n"
    "---------------------\n{context_str}\n---------------------\n"
    "Given this information, answer the question. Focus on what's most important to answer the user's request. Do not reference the context or sources/page numbers in your answer, cite exact phrases from the context, or make up answers:\n"
    "{query_str}\n"
)

EXPLAIN_SYSTEM = PromptTemplate(
    "You explain concepts clearly and precisely using only the provided context. "
    "Prefer short paragraphs and definitions sourced from the text."
    "---------------------\n{context_str}\n---------------------\n"
    "Given this information, answer the question. Focus on what's most important to answer the user's request. Do not reference the context or sources/page numbers in your answer, cite exact phrases from the context, or make up answers:\n"
    "{query_str}\n"
)

TEACH_SYSTEM = PromptTemplate(
    "You are a helpful tutor. Use ONLY the provided context. "
    "Teach progressively, and if the user’s question is ambiguous, ask a brief clarifying question."
    "---------------------\n{context_str}\n---------------------\n"
    "Given this information, answer the question. Focus on what's most important to answer the user's request. Do not reference the context or sources/page numbers in your answer, cite exact phrases from the context, or make up answers:\n"
    "{query_str}\n"
)

CUSTOM_SUMMARIZE_PROMPT = PromptTemplate(
    """\
Given a conversation (between Human and Assistant) and a follow up message from Human, \
rewrite the message to be a standalone question that captures all relevant context \
from the conversation. \
You are a document Q&A assistant. Answer ONLY from the provided context. \
If the answer is not in the context, say: \"I don't know based on the provided documents.\" \
Be concise and do not repeat instructions.

<Chat History>
{chat_history}

<Follow Up Message>
{question}

<Standalone question>
"""
)

CUSTOM_SUMMARIZE_CHAT_HISTORY = [
    ChatMessage(
        role=MessageRole.USER,
        content="Hello assistant, Summarize the key points from the context which is the documents I uploaded. Avoid quoting large passages; use your own words.",
    ),
    ChatMessage(role=MessageRole.ASSISTANT, content="Given this information, answer the question. Do not reference the context or sources/page numbers in your answer, cite exact phrases from the context, or make up answers"),
]
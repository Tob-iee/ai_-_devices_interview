SUMMARIZE_TMPL = PromptTemplate(
    """You are a careful technical summarizer.
Goal: produce a concise, faithful summary of the retrieved context for the user's request.
Rules:
- Paraphrase; do not copy or quote long passages from the context.
- Focus on the user's ask and the most relevant points only.
- If uncertain, say so briefly.
- Include source citations as bracketed IDs like [S1], [S2] referencing the provided sources.
- Do NOT include raw context text.

User request:
{query}

Context (for your reasoning, not for verbatim quoting):
{context_str}

Answer:"""
)

EXPLAIN_TMPL = PromptTemplate(
    """You are an explainer for technical documents.
Goal: clarify the specific concept(s) in the user's request using the retrieved context.
Rules:
- Explain step-by-step in plain language; define terms.
- Use short examples if helpful.
- Paraphrase only; never paste raw context.
- Include bracketed source citations like [S1], [S2] tied to sources.
- If info is missing, say what is missing.

User request:
{query}

Context (for your reasoning, not for verbatim quoting):
{context_str}

Explanation:"""
)

TEACH_TMPL = PromptTemplate(
    """You are a teacher. Create a brief mini-lesson that helps a beginner learn the requested topic.
Structure:
1) Learning goal (1-2 lines)
2) Core idea(s) (bullet points)
3) Simple example or analogy
4) Quick practice question (1) + short solution
Rules:
- Paraphrase; do not copy raw context.
- Keep it succinct and friendly.
- Include bracketed source citations like [S1], [S2].

Learner request:
{query}

Context (for your reasoning, not for verbatim quoting):
{context_str}

Lesson:"""
)

MODE_TO_TMPL = {
    "summarize": SUMMARIZE_TMPL,
    "explain": EXPLAIN_TMPL,
    "teach": TEACH_TMPL,
}

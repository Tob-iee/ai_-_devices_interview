import os
import torch
import hashlib
from typing import Any, Dict, List, Optional, Tuple


_DEVICE_OVERRIDE = os.getenv("LLM_DEVICE", "").lower().strip()
_DTYPE_OVERRIDE  = os.getenv("LLM_DTYPE", "").lower().strip()

def _choose_device() -> str:
    if _DEVICE_OVERRIDE in {"cuda", "mps", "cpu"}:
        return _DEVICE_OVERRIDE
    if torch.cuda.is_available():
        return "cuda"          # single-GPU; for multi-GPU you can consider "auto"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def _choose_dtype(device: str):
    if _DTYPE_OVERRIDE in {"float16", "fp16"}:
        return torch.float16
    if _DTYPE_OVERRIDE in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if _DTYPE_OVERRIDE in {"float32", "fp32"}:
        return torch.float32

    # sensible defaults per device
    if device == "cuda":
        return torch.float16   # fast & common on NVIDIA
    if device == "mps":
        # MPS handles fp16; bf16 support depends on macOS/PyTorch version.
        return torch.float16
    return torch.float32 


def format_response_payload(
    response,
    *,
    max_citations: int = 5,
    dedupe: bool = True,
    snippet_limit: int = 200,
) -> Dict[str, Any]:
    """
    Compact formatter for LlamaIndex responses.
    Returns: {"answer": str, "citations": [{file, page_label, page, score, snippet}]}
    """
    # Clean answer text only
    answer = (getattr(response, "response", None) or str(response)).strip()

    # Build citations compactly (inline helpers)
    citations: List[Dict[str, Any]] = []
    seen: set[Tuple[Optional[str], Optional[str], Optional[int]]] = set()
    source_nodes = getattr(response, "source_nodes", []) or []

    for nws in source_nodes:
        node = getattr(nws, "node", None)
        if node is None:
            continue

        md: Dict[str, Any] = getattr(node, "metadata", {}) or {}

        # Inline "_safe_meta" behavior
        file_path = md.get("file_path") or md.get("filename") or md.get("source")
        page_label = md.get("page_label") or md.get("page_label_old")
        page_num = md.get("page") or md.get("page_number")

        if dedupe:
            key = (file_path, page_label, page_num)
            if key in seen:
                continue
            seen.add(key)

        # Inline "_short_snippet" behavior
        get_text = getattr(node, "get_text", None)
        raw_snip = get_text() if callable(get_text) else ""
        snip = (raw_snip or "").strip().replace("\n", " ")
        if len(snip) > snippet_limit:
            snip = snip[: snippet_limit - 1] + "…"

        score = getattr(nws, "score", None)
        citations.append({
            "file": file_path,
            "page_label": page_label,
            "page": page_num,
            "score": float(score) if score is not None else None,
            "snippet": snip,
        })

        if len(citations) >= max_citations:
            break

    return {"answer": answer, "citations": citations}

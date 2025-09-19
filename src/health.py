import os
import sys
import json
import time
import tempfile
import argparse
from typing import Dict, Any, List, Optional, Tuple


def _ok(msg: str) -> Dict[str, Any]:
    return {"status": "ok", "detail": msg}


def _fail(msg: str) -> Dict[str, Any]:
    return {"status": "fail", "detail": msg}


def _parse_ver(s: str) -> Tuple[int, int]:
    parts = s.split(".")
    return (int(parts[0]), int(parts[1]))


def check_tmp_write() -> Dict[str, Any]:
    try:
        with tempfile.NamedTemporaryFile(mode="w", delete=True) as f:
            f.write("ping")
        return _ok("Temp directory write OK")
    except Exception as e:
        return _fail(f"Temp directory write failed: {e}")



def check_qdrant(
    url: Optional[str],
    api_key: Optional[str],
    host: Optional[str],
    port: Optional[int],
    timeout: float,
    retries: int,
) -> Dict[str, Any]:
    try:
        from qdrant_client import QdrantClient
        client: Optional[QdrantClient] = None

        if url:
            client = QdrantClient(url=url, api_key=api_key, timeout=timeout, retries=retries)
        elif host and port:
            client = QdrantClient(host=host, port=port, timeout=timeout, retries=retries)
        else:
            return _fail("Qdrant config missing (provide QDRANT_URL or host/port)")

        t0 = time.time()
        client.get_locks()  # lightweight probe
        ms = int((time.time() - t0) * 1000)
        return _ok(f"Qdrant reachable ({'Cloud' if url else f'{host}:{port}'}) in {ms} ms")
    except Exception as e:
        return _fail(f"Qdrant connectivity failed: {e}")


def run_health_checks(
    min_python: str = "3.11",
    max_python: str = "3.13",
    required_modules: Optional[List[str]] = None,
    qdrant_host: Optional[str] = "localhost",
    qdrant_port: int = 6333,
    qdrant_timeout: float = 10.0,
    qdrant_retries: int = 2,
    skip_torch_check: bool = False,
) -> Dict[str, Any]:
    """Run all health checks and return a structured dict. Never exits."""
    if required_modules is None:
        required_modules = [
            "streamlit",
            "llama_index",
            "llama_index.llms.huggingface",
            "llama_index.embeddings.huggingface",
            "llama_index.vector_stores.qdrant",
            "qdrant_client",
            "transformers",
            "accelerate",
        ]

    checks: Dict[str, Dict[str, Any]] = {}


    checks["tmpdir"] = check_tmp_write()

    checks["qdrant"] = check_qdrant(
        host=qdrant_host,
        port=qdrant_port,
        timeout=qdrant_timeout,
        retries=qdrant_retries,
    )

    overall = "ok" if all(v["status"] == "ok" for v in checks.values()) else "fail"
    return {"overall": overall, "checks": checks}

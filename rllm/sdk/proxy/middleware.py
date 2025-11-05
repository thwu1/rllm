"""FastAPI middleware for LiteLLM proxy metadata handling."""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from rllm.sdk.tracing import LLMTracer

from .metadata_slug import extract_metadata_from_path


class MetadataRoutingMiddleware(BaseHTTPMiddleware):
    """Decode metadata slugs, rewrite OpenAI paths, and optionally log via LLMTracer."""

    def __init__(self, app: ASGIApp, *, tracer: LLMTracer | None = None) -> None:
        super().__init__(app)
        self._tracer = tracer

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        src_path = request.url.path
        metadata: dict[str, Any] = {}

        extracted = extract_metadata_from_path(src_path)
        if extracted is not None:
            clean_path, metadata = extracted
            request.scope["path"] = clean_path
            request.scope["raw_path"] = clean_path.encode("utf-8")

        request.state.rllm_metadata = metadata

        body_bytes = await request.body()
        request_payload = self._parse_json(body_bytes)
        request.state.rllm_request_payload = request_payload
        request.state.rllm_request_body = body_bytes

        start = time.perf_counter()
        response = await call_next(request)

        # Buffer the upstream response, log once, then return a regular Response.
        body_chunks: list[bytes] = []
        async for chunk in response.body_iterator:
            if isinstance(chunk, bytes):
                data = chunk
            else:
                data = str(chunk).encode("utf-8")
            body_chunks.append(data)

        latency_ms = (time.perf_counter() - start) * 1000
        if self._tracer is not None:
            await self._log_trace(
                tracer=self._tracer,
                request=request,
                request_payload=request.state.rllm_request_payload,
                request_body=body_bytes,
                response_body=b"".join(body_chunks),
                latency_ms=latency_ms,
            )

        new_response = Response(
            content=b"".join(body_chunks),
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
            background=response.background,
        )
        if hasattr(response, "raw_headers"):
            new_response.raw_headers = response.raw_headers  # type: ignore[attr-defined]

        return new_response

    async def _log_trace(
        self,
        *,
        tracer: LLMTracer,
        request: Request,
        request_payload: dict[str, Any] | None,
        request_body: bytes,
        response_body: bytes,
        latency_ms: float,
    ) -> bool:
        metadata = getattr(request.state, "rllm_metadata", {}) or {}

        payload = request_payload
        if payload is None:
            payload = self._parse_json(request_body)
        if payload is None:
            return False

        text_body = response_body.decode("utf-8") if response_body else ""
        response_payload: dict[str, Any]
        try:
            response_payload = json.loads(text_body) if text_body else {}
        except Exception:
            response_payload = {}
            for line in text_body.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    candidate = json.loads(line)
                except Exception:
                    continue
                if isinstance(candidate, dict) and "usage" in candidate:
                    response_payload = candidate
            if not response_payload and text_body:
                try:
                    response_payload = json.loads(text_body.splitlines()[-1])
                except Exception:
                    response_payload = {}

        model = payload.get("model", "")
        name = f"proxy{request.url.path}"

        usage = response_payload.get("usage") or {}
        tokens = {
            "prompt": int(usage.get("prompt_tokens") or 0),
            "completion": int(usage.get("completion_tokens") or 0),
            "total": int(usage.get("total_tokens") or (usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0)),
        }
        if tokens["total"] == 0 and text_body:
            tokens.update(self._extract_usage_from_text(text_body))

        tracer.log_llm_call(
            name=name,
            model=model or "unknown",
            input=payload,
            output=response_payload,
            metadata=metadata,
            session_id=metadata.get("session_id"),
            latency_ms=latency_ms,
            tokens=tokens,
        )
        return True

    @staticmethod
    def _parse_json(data: bytes) -> dict[str, Any] | None:
        if not data:
            return None
        try:
            return json.loads(data.decode("utf-8"))
        except Exception:
            return None

    @staticmethod
    def _extract_usage_from_text(text: str) -> dict[str, int]:
        prompt = completion = total = 0
        import re

        def _find(key: str) -> int:
            pattern = rf'"{key}"\s*:\s*(\d+)'
            match = re.search(pattern, text)
            return int(match.group(1)) if match else 0

        prompt = _find("prompt_tokens")
        completion = _find("completion_tokens")
        total = _find("total_tokens") or (prompt + completion)
        return {"prompt": prompt, "completion": completion, "total": total}

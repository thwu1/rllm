"""Tests for proxy-aware chat clients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rllm.sdk.chat.proxy_chat_client import ProxyTrackedChatClient
from rllm.sdk.session import SessionContext


@dataclass
class DummyResponse:
    payload: dict[str, Any]

    def model_dump(self) -> dict[str, Any]:
        return self.payload


class DummyChatCompletions:
    def __init__(self, owner: DummyOpenAI) -> None:
        self._owner = owner

    def create(self, **kwargs: Any) -> DummyResponse:
        self._owner.last_request = kwargs
        return DummyResponse({"choices": [], "usage": {}})


class DummyChatNamespace:
    def __init__(self, owner: DummyOpenAI) -> None:
        self.completions = DummyChatCompletions(owner)


class DummyCompletions:
    def __init__(self, owner: DummyOpenAI) -> None:
        self._owner = owner

    def create(self, **kwargs: Any) -> DummyResponse:
        self._owner.last_request = kwargs
        return DummyResponse({"choices": [], "usage": {}})


class DummyOpenAI:
    def __init__(self, *, base_url: str | None = None) -> None:
        self.base_url = base_url
        self.chat = DummyChatNamespace(self)
        self.completions = DummyCompletions(self)
        self.last_request: dict[str, Any] | None = None
        self.children: list[DummyOpenAI] = []

    def with_options(self, *, base_url: str | None = None) -> DummyOpenAI:
        child = DummyOpenAI(base_url=base_url)
        self.children.append(child)
        return child


def test_proxy_client_injects_slug_into_base_url():
    dummy = DummyOpenAI(base_url="http://proxy-host:4000/v1")

    with SessionContext("sess-proxy", job="nightly"):
        client = ProxyTrackedChatClient(client=dummy, base_url="http://proxy-host:4000/v1")
        client.chat.completions.create(messages=[{"role": "user", "content": "hi"}], model="gpt-4o-mini")

    assert dummy.children, "with_options should be invoked to scope base_url"
    scoped = dummy.children[-1]
    assert scoped.base_url is not None
    assert "/meta/" in scoped.base_url
    assert scoped.base_url.startswith("http://proxy-host:4000/meta/")
    assert scoped.base_url.endswith("/v1")

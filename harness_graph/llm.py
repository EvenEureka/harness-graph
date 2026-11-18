"""Chat-model clients: an OpenAI-compatible async client (vLLM or a hosted API) and a scripted mock."""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Callable

_THINK = re.compile(r"<think>.*?</think>", re.S)
_CTX = re.compile(r"maximum context length|context length|too large|ContextWindow", re.I)


class ContextExceeded(Exception):
    """The prompt no longer fits the model's context window."""


@dataclass
class Completion:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    truncated: bool = False        # stopped by the token cap, not by the model
    cached_tokens: int = 0         # part of prompt_tokens that was served from the provider's prompt cache
    reasoning_tokens: int = 0      # part of completion_tokens spent on hidden reasoning
    cost_usd: float = 0.0


class SpendCapReached(Exception):
    """The next call could push the run past its USD cap, so it was not sent."""


@dataclass
class Prices:
    """USD per million tokens.  Reasoning tokens are billed as output."""
    input: float = 0.0
    cached_input: float = 0.0
    output: float = 0.0

    def cost(self, n_in: int, n_cached: int, n_out: int) -> float:
        n_cached = min(n_cached, n_in)
        return ((n_in - n_cached) * self.input + n_cached * self.cached_input + n_out * self.output) / 1e6


class LLM:
    async def chat(self, messages: list[dict], *, max_tokens: int = 4096, seed: int | None = None) -> Completion:
        raise NotImplementedError


class OpenAIChat(LLM):
    """Any OpenAI-compatible endpoint: a local vLLM server or a hosted API.

    Several base URLs (replicas of one model) are used round-robin per ``affinity`` key, so one episode
    stays on one replica and keeps its prefix cache.

    For a paid API pass ``prices`` and ``max_usd``.  Every call first reserves its worst case (the prompt
    at the uncached price plus the full output allowance); if that would pass the cap the call is not
    sent and ``SpendCapReached`` is raised, so the cap cannot be overshot by calls already in flight.
    ``reasoning_effort`` switches to the request shape of reasoning models (``max_completion_tokens``, no
    sampling parameters) and ``token_headroom`` is added to every output allowance so that hidden
    reasoning does not eat the visible answer.
    """

    def __init__(self, model: str, base_url: str, api_key: str = "EMPTY", concurrency: int = 32,
                 temperature: float | None = 0.7, top_p: float | None = 0.8, extra_body: dict | None = None,
                 retries: int = 5, timeout: float = 900.0, reasoning_effort: str | None = None,
                 token_headroom: int = 0, prices: Prices | None = None, max_usd: float | None = None):
        from openai import AsyncOpenAI
        self.model = model
        self.clients = [AsyncOpenAI(base_url=u, api_key=api_key, timeout=timeout, max_retries=0)
                        for u in base_url.split(",")]
        self.sem = asyncio.Semaphore(concurrency)
        self.temperature, self.top_p = temperature, top_p
        self.extra_body = extra_body or {}
        self.retries = retries
        self.reasoning_effort, self.token_headroom = reasoning_effort, token_headroom
        self.prices, self.max_usd = prices or Prices(), max_usd
        self.spent = self.reserved = 0.0
        self.capped = False
        self._secret = api_key if api_key != "EMPTY" else None

    def _reserve(self, messages, max_out: int) -> float:
        est = self.prices.cost(int(sum(len(m["content"]) for m in messages) / 2.5) + 16 * len(messages), 0, max_out)
        if self.max_usd is not None and self.spent + self.reserved + est > self.max_usd:
            self.capped = True
            raise SpendCapReached(f"spent ${self.spent:.2f} + reserved ${self.reserved:.2f} + next ${est:.2f} "
                                  f"> cap ${self.max_usd:.2f}")
        self.reserved += est
        return est

    async def chat(self, messages, *, max_tokens=4096, seed=None, affinity: int = 0) -> Completion:
        client, last = self.clients[affinity % len(self.clients)], None
        max_out = max_tokens + self.token_headroom
        kw: dict = {"model": self.model, "messages": messages, "seed": seed}
        if self.reasoning_effort:
            kw.update(max_completion_tokens=max_out, reasoning_effort=self.reasoning_effort)
        else:
            kw.update(max_tokens=max_out)
        if self.temperature is not None:
            kw["temperature"] = self.temperature
        if self.top_p is not None:
            kw["top_p"] = self.top_p
        if self.extra_body:
            kw["extra_body"] = self.extra_body
        for attempt in range(self.retries):
            async with self.sem:
                hold = self._reserve(messages, max_out)
                try:
                    r = await client.chat.completions.create(**kw)
                except Exception as e:
                    last, status = e, getattr(e, "status_code", None)
                    if status is not None and 400 <= status < 500 and status != 429:
                        msg = str(e).replace(self._secret, "***") if self._secret else str(e)
                        if _CTX.search(msg):
                            raise ContextExceeded(msg[:300]) from None
                        raise RuntimeError(f"model call rejected ({status}): {msg[:500]}") from None
                else:
                    u, c = r.usage, r.choices[0]
                    n_in, n_out = (u.prompt_tokens, u.completion_tokens) if u else (0, 0)
                    cached = int(getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0)
                    reason = int(getattr(getattr(u, "completion_tokens_details", None), "reasoning_tokens", 0) or 0)
                    cost = self.prices.cost(n_in, cached, n_out)
                    self.spent += cost
                    return Completion(_THINK.sub("", c.message.content or "").strip(), n_in, n_out,
                                      truncated=c.finish_reason == "length", cached_tokens=cached,
                                      reasoning_tokens=reason, cost_usd=cost)
                finally:
                    self.reserved -= hold
            await asyncio.sleep(min(60, 2 ** attempt * 2))     # rate limit / network / server: back off
        msg = repr(last).replace(self._secret, "***") if self._secret else repr(last)
        raise RuntimeError(f"model call failed after {self.retries} attempts: {msg[:500]}")


class MockLLM(LLM):
    """Scripted stand-in for tests and dry runs.  ``fn(messages, seed) -> str``."""

    def __init__(self, fn: Callable[[list[dict], int | None], str]):
        self.fn = fn
        self.calls: list[list[dict]] = []

    async def chat(self, messages, *, max_tokens=4096, seed=None, affinity: int = 0) -> Completion:
        self.calls.append([dict(m) for m in messages])
        text = self.fn(messages, seed)
        return Completion(text, sum(len(m["content"]) for m in messages) // 4, len(text) // 4)

"""Hosted-API client: request shape, cost accounting and the hard USD cap (no network)."""
import asyncio
from types import SimpleNamespace as NS

import pytest

from harness_graph.llm import OpenAIChat, Prices, SpendCapReached


class FakeClient:
    def __init__(self):
        self.requests = []
        self.chat = NS(completions=NS(create=self.create))

    async def create(self, **kw):
        self.requests.append(kw)
        usage = NS(prompt_tokens=1000, completion_tokens=200, prompt_tokens_details=NS(cached_tokens=800),
                   completion_tokens_details=NS(reasoning_tokens=50))
        return NS(usage=usage, choices=[NS(message=NS(content="hi"), finish_reason="stop")])


def client(**kw):
    llm = OpenAIChat("m", "http://unused", api_key="sk-test", temperature=None, top_p=None, **kw)
    llm.clients = [FakeClient()]
    return llm


MSG = [{"role": "user", "content": "x" * 2500}]


def test_cost_counts_cached_input_at_its_own_price():
    llm = client(reasoning_effort="low", token_headroom=1000, prices=Prices(1.0, 0.1, 5.0), max_usd=1.0)
    r = asyncio.run(llm.chat(MSG, max_tokens=500))
    assert r.cached_tokens == 800 and r.reasoning_tokens == 50
    assert r.cost_usd == pytest.approx((200 * 1.0 + 800 * 0.1 + 200 * 5.0) / 1e6) and llm.spent == r.cost_usd
    req = llm.clients[0].requests[0]
    assert req["max_completion_tokens"] == 1500 and req["reasoning_effort"] == "low"
    assert "temperature" not in req and "max_tokens" not in req and llm.reserved == pytest.approx(0)


def test_call_that_could_pass_the_cap_is_not_sent():
    llm = client(prices=Prices(1.0, 0.1, 5.0), max_usd=0.004)       # worst case of one call: about $0.0035
    asyncio.run(llm.chat(MSG, max_tokens=500))
    with pytest.raises(SpendCapReached):
        asyncio.run(llm.chat(MSG, max_tokens=500))
    assert len(llm.clients[0].requests) == 1 and llm.capped and llm.spent < 0.004

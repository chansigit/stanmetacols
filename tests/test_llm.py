# tests/test_llm.py
import json

import pandas as pd
import pytest

from stanmetacols.profile import profile_obs
from stanmetacols.schema import RankedCandidates, LLMUnavailable
from stanmetacols.llm import rank_with_llm


def _digest():
    return profile_obs(pd.DataFrame(
        {"sample_id": ["S1"] * 5 + ["S2"] * 5, "tissue": ["lung"] * 10},
        index=[f"c{i}" for i in range(10)]))


class _StubMessages:
    def __init__(self, parsed):
        self._parsed = parsed
        self.kwargs = None

    def parse(self, **kwargs):
        self.kwargs = kwargs

        class _Resp:
            parsed_output = self._parsed
        return _Resp()


class _StubClient:
    def __init__(self, parsed):
        self.messages = _StubMessages(parsed)


def test_maps_and_filters_hallucinations():
    parsed = RankedCandidates(candidates=[
        {"role": "sample", "column": "sample_id", "kind": "single", "score": 0.9, "reason": "looks like a sample id"},
        {"role": "sample", "column": "made_up_column", "kind": "single", "score": 0.8, "reason": "hallucinated"},
    ])
    client = _StubClient(parsed)
    out = rank_with_llm(_digest(), ["sample"], client=client)
    cols = [c.column for c in out["sample"]]
    assert "sample_id" in cols
    assert "made_up_column" not in cols          # filtered: not in digest
    assert all(c.source == "llm" for c in out["sample"])
    # prompt carried the digest
    assert "sample_id" in client.messages.kwargs["messages"][0]["content"]
    assert client.messages.kwargs["model"] == "claude-opus-4-8"


def test_api_error_becomes_llm_unavailable():
    class _Boom:
        class messages:
            @staticmethod
            def parse(**kwargs):
                raise RuntimeError("network down")
    with pytest.raises(LLMUnavailable):
        rank_with_llm(_digest(), ["sample"], client=_Boom())


def test_none_parsed_output_becomes_llm_unavailable():
    with pytest.raises(LLMUnavailable):
        rank_with_llm(_digest(), ["sample"], client=_StubClient(None))


# --- OpenAI-compatible backend (OpenAI, Volcengine ARK, DeepSeek, vLLM, …) ---

class _OAResp:
    def __init__(self, content):
        message = type("Msg", (), {"content": content})()
        choice = type("Choice", (), {"message": message})()
        self.choices = [choice]


class _StubCompletions:
    def __init__(self, content=None, raise_exc=None):
        self._content = content
        self._raise = raise_exc
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self._raise is not None:
            raise self._raise
        return _OAResp(self._content)


class _StubOpenAIClient:
    def __init__(self, content=None, raise_exc=None):
        self.completions = _StubCompletions(content, raise_exc)
        self.chat = type("Chat", (), {"completions": self.completions})()


def test_openai_parses_json_and_filters_hallucinations():
    content = json.dumps({"candidates": [
        {"role": "sample", "column": "sample_id", "kind": "single", "score": 0.9, "reason": "ok"},
        {"role": "sample", "column": "made_up", "kind": "single", "score": 0.8, "reason": "halluc"},
    ]})
    client = _StubOpenAIClient(content)
    out = rank_with_llm(_digest(), ["sample"], provider="openai", client=client)
    assert [c.column for c in out["sample"]] == ["sample_id"]   # made_up dropped
    assert all(c.source == "llm" for c in out["sample"])
    msgs = client.completions.kwargs["messages"]       # system + user, digest carried
    assert msgs[0]["role"] == "system"
    assert "sample_id" in msgs[1]["content"]


def test_openai_strips_markdown_code_fences():
    body = json.dumps({"candidates": [
        {"role": "sample", "column": "sample_id", "kind": "single", "score": 0.7, "reason": "x"}]})
    client = _StubOpenAIClient("```json\n" + body + "\n```")
    out = rank_with_llm(_digest(), ["sample"], provider="openai", client=client)
    assert [c.column for c in out["sample"]] == ["sample_id"]


def test_openai_accepts_bare_json_array():
    content = json.dumps([
        {"role": "sample", "column": "sample_id", "kind": "single", "score": 0.6, "reason": "x"}])
    out = rank_with_llm(_digest(), ["sample"], provider="openai", client=_StubOpenAIClient(content))
    assert [c.column for c in out["sample"]] == ["sample_id"]


def test_openai_non_json_becomes_llm_unavailable():
    with pytest.raises(LLMUnavailable):
        rank_with_llm(_digest(), ["sample"], provider="openai",
                      client=_StubOpenAIClient("I cannot help with that."))


def test_openai_api_error_becomes_llm_unavailable():
    client = _StubOpenAIClient(raise_exc=RuntimeError("network down"))
    with pytest.raises(LLMUnavailable):
        rank_with_llm(_digest(), ["sample"], provider="openai", client=client)


def test_unknown_provider_becomes_llm_unavailable():
    with pytest.raises(LLMUnavailable):
        rank_with_llm(_digest(), ["sample"], provider="cohere")


def test_hint_reaches_user_prompt():
    parsed = RankedCandidates(candidates=[])
    client = _StubClient(parsed)
    rank_with_llm(_digest(), ["sample"], hint="HINTTOKEN", client=client)
    content = client.messages.kwargs["messages"][0]["content"]
    assert "HINTTOKEN" in content

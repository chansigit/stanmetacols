# tests/test_rank.py
import pandas as pd

from stanmetacols.rank import rank_sample_columns
from stanmetacols.schema import LLMUnavailable, RankedCandidates


def _obs():
    return pd.DataFrame(
        {"sample_id": ["S1"] * 5 + ["S2"] * 5, "tissue": ["lung"] * 10},
        index=[f"c{i}" for i in range(10)])


class _StubClient:
    def __init__(self, parsed):
        class _M:
            def parse(_self, **kw):
                class _R:
                    parsed_output = parsed
                return _R()
        self.messages = _M()


class _Boom:
    class messages:
        @staticmethod
        def parse(**kw):
            raise RuntimeError("no network")


def test_no_llm_uses_heuristic():
    res = rank_sample_columns(_obs(), use_llm=False)
    assert res.method == "heuristic"
    assert res.top().column == "sample_id"


def test_llm_path_with_mock_client():
    parsed = RankedCandidates(candidates=[
        {"column": "sample_id", "kind": "single", "score": 0.95, "reason": "ok"}])
    res = rank_sample_columns(_obs(), use_llm=True, client=_StubClient(parsed))
    assert res.method == "llm (anthropic)"
    assert res.top().column == "sample_id"
    assert res.top().source == "llm"


def test_llm_failure_falls_back():
    res = rank_sample_columns(_obs(), use_llm=True, client=_Boom())
    assert res.method.startswith("heuristic (llm unavailable")
    assert res.top().column == "sample_id"


def test_top_k_truncation():
    res = rank_sample_columns(_obs(), use_llm=False, top_k=1)
    assert len(res.candidates) == 1
    res_all = rank_sample_columns(_obs(), use_llm=False, top_k=0)
    assert len(res_all.candidates) >= 1


def test_empty_obs():
    res = rank_sample_columns(pd.DataFrame(), use_llm=False)
    assert res.candidates == []
    assert res.top() is None


def test_input_not_mutated():
    obs = _obs()
    before = obs.copy()
    rank_sample_columns(obs, use_llm=False)
    pd.testing.assert_frame_equal(obs, before)


def test_dataframe_with_obs_column_does_not_crash():
    # A bare DataFrame with a column literally named "obs" must not be mistaken
    # for an AnnData; it should rank normally via the heuristic path.
    df = pd.DataFrame(
        {"obs": ["S1"] * 3 + ["S2"] * 3, "x": list(range(6))},
        index=[f"c{i}" for i in range(6)])
    res = rank_sample_columns(df, use_llm=False)
    assert res.method == "heuristic"


def test_top_k_none_and_negative_return_all():
    df = pd.DataFrame(
        {"sample_id": ["S1"] * 3 + ["S2"] * 3, "tissue": ["lung"] * 6},
        index=[f"c{i}" for i in range(6)])
    n_all = len(rank_sample_columns(df, use_llm=False, top_k=0).candidates)
    assert n_all >= 1
    assert len(rank_sample_columns(df, use_llm=False, top_k=None).candidates) == n_all
    assert len(rank_sample_columns(df, use_llm=False, top_k=-1).candidates) == n_all

import pandas as pd
from stanmetacols.rank import rank_meta_columns
from stanmetacols.schema import RankedCandidates


def _obs():
    n = 40
    return pd.DataFrame({
        "sample": ["S1"] * 20 + ["S2"] * 20,
        "pct_counts_mt": [i / 100 for i in range(n)],
        "total_counts": [1000 + 5 * i for i in range(n)],
    }, index=[f"c{i}" for i in range(n)])


class _StubClient:
    def __init__(self, parsed):
        class _M:
            def parse(_s, **kw):
                class _R: parsed_output = parsed
                return _R()
        self.messages = _M()


class _Boom:
    class messages:
        @staticmethod
        def parse(**kw):
            raise RuntimeError("no network")


def test_no_llm_heuristic_groups_by_role():
    res = rank_meta_columns(_obs(), use_llm=False)
    assert res.method == "heuristic"
    assert res.top("sample").column == "sample"
    assert res.top("pct_mt").column == "pct_counts_mt"
    assert res.top("n_counts").column == "total_counts"


def test_roles_subset():
    res = rank_meta_columns(_obs(), use_llm=False, roles=["pct_mt"])
    assert set(res.roles) == {"pct_mt"}


def test_llm_path_with_mock_client():
    parsed = RankedCandidates(candidates=[
        {"role": "pct_mt", "column": "pct_counts_mt", "kind": "single",
         "score": 0.95, "reason": "ok"}])
    res = rank_meta_columns(_obs(), use_llm=True, adjudicate=False,
                            client=_StubClient(parsed))
    assert res.method == "llm (anthropic)"
    assert res.top("pct_mt").column == "pct_counts_mt"
    assert res.top("pct_mt").source == "llm"


def test_llm_failure_falls_back():
    res = rank_meta_columns(_obs(), use_llm=True, adjudicate=False, client=_Boom())
    assert res.method.startswith("heuristic (llm unavailable")
    assert res.top("sample").column == "sample"


def test_top_k_truncation_per_role():
    res = rank_meta_columns(_obs(), use_llm=False, top_k=1)
    assert all(len(v) <= 1 for v in res.roles.values())


def test_input_not_mutated():
    obs = _obs(); before = obs.copy()
    rank_meta_columns(obs, use_llm=False)
    pd.testing.assert_frame_equal(obs, before)

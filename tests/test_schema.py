# tests/test_schema.py
import pytest
import pandas as pd
import stanmetacols
from stanmetacols.schema import (
    Candidate, ObsDigest, ColumnProfile, CompositeProfile,
    BarcodeProfile, RankedCandidate, RankedCandidates, LLMUnavailable,
    MetaColsResult, Adjudications,
)
from stanmetacols.profile import profile_obs


@pytest.fixture
def digest_fixture():
    return profile_obs(pd.DataFrame({"sample": ["A", "B"]}))


def test_version():
    assert stanmetacols.__version__ == "0.2.0"


def test_candidate_fields():
    digest = ObsDigest(n_obs=1, columns=[], composite_candidates=[], barcode=None)
    c = Candidate(role="sample", column="sample_id", kind="single", score=0.9, reason="r", source="heuristic")
    assert c.role == "sample"
    assert c.column == "sample_id"


def test_labels():
    assert CompositeProfile(columns=["donor", "timepoint"], n_unique=6,
                            cells_per_group={"min": 5, "max": 5, "median": 5.0},
                            balance=1.0).label == "donor + timepoint"
    assert BarcodeProfile(delimiter="_", position="prefix", n_groups=3,
                          cells_per_group={"min": 10, "max": 10, "median": 10.0},
                          balance=1.0, example_groups=["S1", "S2", "S3"]
                          ).label == "<barcode:prefix:_>"


def test_to_prompt_dict_is_jsonable():
    import json
    d = ObsDigest(
        n_obs=2,
        columns=[ColumnProfile(name="a", dtype="categorical", n_unique=2, n_missing=0,
                               example_values=["x", "y"],
                               cells_per_group={"min": 1, "max": 1, "median": 1.0},
                               balance=1.0, unique_per_cell=True, single_value=False,
                               looks_like_barcode=False)],
        composite_candidates=[], barcode=None,
    ).to_prompt_dict()
    assert json.dumps(d, sort_keys=True)
    assert d["columns"][0]["name"] == "a"


def test_pydantic_schema():
    rc = RankedCandidates(candidates=[{"role": "sample", "column": "a", "kind": "single", "score": 0.5, "reason": "x"}])
    assert rc.candidates[0].column == "a"


def test_llm_unavailable_is_exception():
    assert issubclass(LLMUnavailable, Exception)


def test_candidate_has_role():
    c = Candidate(role="sample", column="s", kind="single", score=0.9,
                  reason="r", source="heuristic")
    assert c.role == "sample"


def test_metacolsresult_top(digest_fixture):
    c = Candidate(role="pct_mt", column="pct_counts_mt", kind="single",
                  score=0.9, reason="r", source="llm")
    res = MetaColsResult(roles={"pct_mt": [c], "sample": []},
                         method="heuristic", digest=digest_fixture)
    assert res.top("pct_mt") is c
    assert res.top("sample") is None
    assert res.top("n_genes") is None        # role absent -> None


def test_adjudications_schema():
    a = Adjudications(verdicts=[{"role": "n_counts", "column": "total_counts",
                                 "reason": "canonical total"}])
    assert a.verdicts[0].column == "total_counts"

import pandas as pd


def test_top_level_exports():
    from stanmetacols import (
        rank_meta_columns, profile_obs, Candidate, MetaColsResult,
        ObsDigest, LLMUnavailable, ROLES, ROLE_KEYS, __version__,
    )
    assert __version__ == "0.2.0"
    res = rank_meta_columns(
        pd.DataFrame({"sample_id": ["S1"] * 3 + ["S2"] * 3},
                     index=[f"c{i}" for i in range(6)]),
        use_llm=False)
    assert isinstance(res, MetaColsResult)
    assert res.top("sample").column == "sample_id"

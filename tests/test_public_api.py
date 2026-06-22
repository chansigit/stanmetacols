import pandas as pd


def test_top_level_exports():
    from stanmetacols import (
        rank_sample_columns, profile_obs, Candidate, RankResult,
        ObsDigest, LLMUnavailable, __version__,
    )
    assert __version__ == "0.1.0"
    res = rank_sample_columns(
        pd.DataFrame({"sample_id": ["S1"] * 3 + ["S2"] * 3},
                     index=[f"c{i}" for i in range(6)]),
        use_llm=False)
    assert isinstance(res, RankResult)
    assert res.top().column == "sample_id"

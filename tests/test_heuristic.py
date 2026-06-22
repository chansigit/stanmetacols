# tests/test_heuristic.py
import numpy as np
import pandas as pd

from stanmetacols.profile import profile_obs
from stanmetacols.heuristic import rank_heuristic


def _obs():
    n = 30
    return pd.DataFrame(
        {
            "sample_id": ["S1"] * 10 + ["S2"] * 10 + ["S3"] * 10,
            "donor": ["D1"] * 15 + ["D2"] * 15,
            "timepoint": ["t0", "t1", "t2"] * 10,
            "tissue": ["lung"] * 30,
            "pct_mito": np.linspace(0.0, 10.0, n),
            "cell_id": [f"cell{i}" for i in range(n)],
        },
        index=[f"S{(i // 10) + 1}_AAAC{i:04d}-1" for i in range(n)],
    )


def test_ranking_order_and_filtering():
    res = rank_heuristic(profile_obs(_obs()))
    cols = [c.column for c in res]
    assert all(c.source == "heuristic" for c in res)
    assert all(c.score > 0 for c in res)
    # alias-named, well-grouped columns lead
    assert cols[0] in ("sample_id", "donor")
    assert cols.index("sample_id") < cols.index("timepoint")
    # composite is offered
    assert "donor + timepoint" in cols
    # zero-score entries are dropped entirely
    assert "tissue" not in cols      # single value
    assert "cell_id" not in cols     # unique-per-cell


def test_empty_digest():
    assert rank_heuristic(profile_obs(pd.DataFrame())) == []

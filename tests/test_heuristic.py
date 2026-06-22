import pandas as pd
from stanmetacols.profile import profile_obs
from stanmetacols.heuristic import rank_heuristic
from stanmetacols.roles import ROLE_KEYS


def _digest():
    n = 60
    return profile_obs(pd.DataFrame({
        "sample": ["S1"] * 30 + ["S2"] * 30,
        "pct_counts_mt": [i / 100 for i in range(n)],          # [0,1) floats
        "pct_counts_hb": [i / 1000 for i in range(n)],
        "total_counts": [1000 + 10 * i for i in range(n)],     # large ints
        "n_genes_by_counts": [200 + i for i in range(n)],      # mid ints
        "doublet_score": [i / 100 for i in range(n)],
    }, index=[f"c{i}" for i in range(n)]))


def test_each_role_top_is_correct_column():
    out = rank_heuristic(_digest(), list(ROLE_KEYS))
    assert out["sample"][0].column == "sample"
    assert out["pct_mt"][0].column == "pct_counts_mt"
    assert out["pct_hb"][0].column == "pct_counts_hb"
    assert out["n_counts"][0].column == "total_counts"
    assert out["n_genes"][0].column == "n_genes_by_counts"
    assert out["doublet_score"][0].column == "doublet_score"


def test_numeric_role_requires_name_hit():
    # a bare [0,1] float column with no pct/doublet name must NOT appear for pct_mt
    d = profile_obs(pd.DataFrame({"score_x": [i / 100 for i in range(50)]},
                                 index=[f"c{i}" for i in range(50)]))
    assert rank_heuristic(d, ["pct_mt"])["pct_mt"] == []


def test_value_guard_rejects_unit_column_for_counts():
    # a [0,1] column literally named like counts should not win n_counts on value
    d = profile_obs(pd.DataFrame({"pct_counts": [i / 100 for i in range(50)]},
                                 index=[f"c{i}" for i in range(50)]))
    cands = rank_heuristic(d, ["n_counts"])["n_counts"]
    assert all(c.score < 0.7 for c in cands)    # name may hit, value does not

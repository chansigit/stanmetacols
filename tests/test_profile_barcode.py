import pandas as pd

from stansample.profile import profile_obs


def test_prefix_grouping_from_underscore():
    obs = pd.DataFrame({"x": list(range(30))},
                       index=[f"S{(i // 10) + 1}_AAAC{i:04d}-1" for i in range(30)])
    d = profile_obs(obs)
    assert d.barcode is not None
    assert d.barcode.position == "prefix"
    assert d.barcode.delimiter == "_"
    assert d.barcode.n_groups == 3
    assert d.barcode.label == "<barcode:prefix:_>"


def test_no_grouping_returns_none():
    obs = pd.DataFrame({"x": [1, 2, 3]}, index=["aaa", "bbb", "ccc"])
    assert profile_obs(obs).barcode is None

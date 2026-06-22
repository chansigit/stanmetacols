import json
import anndata
import numpy as np
import pandas as pd
from stanmetacols.__main__ import main


def _write(path, obs, names):
    a = anndata.AnnData(X=np.zeros((len(obs), 2), dtype="float32"),
                        obs=obs.set_index(pd.Index(names)))
    a.write_h5ad(path)


def test_cli_emits_roles_json(tmp_path, capsys):
    p = tmp_path / "x.h5ad"
    n = 20
    obs = pd.DataFrame({"sample": ["S1"] * 10 + ["S2"] * 10,
                        "pct_counts_mt": [i / 100 for i in range(n)]})
    _write(p, obs, [f"c{i}" for i in range(n)])
    code = main([str(p), "--no-llm"])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["method"] == "heuristic"
    assert out["roles"]["pct_mt"][0]["column"] == "pct_counts_mt"


def test_cli_roles_subset(tmp_path, capsys):
    p = tmp_path / "y.h5ad"
    obs = pd.DataFrame({"sample": ["S1"] * 5 + ["S2"] * 5})
    _write(p, obs, [f"c{i}" for i in range(10)])
    code = main([str(p), "--no-llm", "--roles", "sample"])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert set(out["roles"]) == {"sample"}


def test_cli_exit_2_when_nothing(tmp_path, capsys):
    p = tmp_path / "z.h5ad"
    obs = pd.DataFrame({"tissue": ["lung"] * 5})       # nothing matches any role
    _write(p, obs, ["aa", "bb", "cc", "dd", "ee"])
    code = main([str(p), "--no-llm"])
    assert code == 2
    out = json.loads(capsys.readouterr().out)
    assert all(v == [] for v in out["roles"].values())


def test_cli_bad_role_exit_1(tmp_path):
    p = tmp_path / "w.h5ad"
    _write(p, pd.DataFrame({"sample": ["A", "B"]}), ["a", "b"])
    assert main([str(p), "--no-llm", "--roles", "bogus"]) == 1


def test_cli_bad_path_exit_1():
    assert main(["/no/such/file.h5ad", "--no-llm"]) == 1


def test_cli_hint_accepted_offline(tmp_path, capsys):
    p = tmp_path / "h.h5ad"
    obs = pd.DataFrame({"sample": ["S1"] * 5 + ["S2"] * 5})
    _write(p, obs, [f"c{i}" for i in range(10)])
    code = main([str(p), "--no-llm", "--hint", "ignore me offline"])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["roles"]["sample"][0]["column"] == "sample"


def test_cli_default_roles_include_celltype(tmp_path, capsys):
    p = tmp_path / "ct.h5ad"
    n = 30
    obs = pd.DataFrame({"sample": ["S1"] * 15 + ["S2"] * 15,
                        "cell_type": (["Epithelial"] * 10 + ["Immune"] * 10
                                      + ["Stromal"] * 10)})
    _write(p, obs, [f"c{i}" for i in range(n)])
    code = main([str(p), "--no-llm"])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert "cell_type_coarse" in out["roles"] and "cell_type_fine" in out["roles"]
    assert out["roles"]["cell_type_coarse"][0]["column"] == "cell_type"

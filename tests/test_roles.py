import pandas as pd
from stanmetacols.profile import profile_obs
from stanmetacols.roles import (ROLES, ROLE_KEYS, name_signal, value_check,
                                CELLTYPE_ROLE_KEYS, celltype_value_frac,
                                celltype_name_base)


def _profile(values):
    return profile_obs(pd.DataFrame({"x": values})).columns[0]


def test_role_keys():
    assert ROLE_KEYS == ("sample", "pct_mt", "pct_hb", "doublet_score",
                         "n_counts", "n_genes", "cell_type_coarse", "cell_type_fine")


def test_role_keys_include_celltype():
    assert ROLE_KEYS[-2:] == ("cell_type_coarse", "cell_type_fine")
    assert CELLTYPE_ROLE_KEYS == ("cell_type_coarse", "cell_type_fine")
    assert ROLES["cell_type_coarse"].type == "celltype"
    assert ROLES["cell_type_fine"].type == "celltype"


def test_celltype_value_frac_high_for_celltypes():
    prof = _profile(["Epithelial cells", "Immune cells", "Stromal cells",
                     "Endothelial cells"])
    assert celltype_value_frac(prof) >= 0.9


def test_celltype_value_frac_low_for_non_celltypes():
    prof = _profile(["lung", "liver", "kidney", "brain"])
    assert celltype_value_frac(prof) <= 0.25


def test_celltype_name_base():
    assert celltype_name_base("cell_type") == 1.0
    assert celltype_name_base("annotation") == 1.0
    assert celltype_name_base("total_counts") == 0.0


def test_name_exact_and_token_and_substring():
    assert name_signal("pct_counts_mt", ROLES["pct_mt"]) == 1.0      # exact alias
    assert name_signal("MyMitoPercent", ROLES["pct_mt"]) == 0.8      # token rule
    assert name_signal("nonsense", ROLES["pct_mt"]) == 0.0


def test_n_genes_by_counts_resolves_to_genes_not_counts():
    # contains "counts" but also "genes" -> excluded from n_counts, matches n_genes
    assert name_signal("n_genes_by_counts", ROLES["n_counts"]) == 0.0
    assert name_signal("n_genes_by_counts", ROLES["n_genes"]) >= 0.8


def test_value_check_unit_for_pct():
    prof = _profile([0.0, 0.05, 0.1, 0.3])
    assert value_check(prof, ROLES["pct_mt"]) == 1.0
    assert value_check(prof, ROLES["n_counts"]) == 0.0     # not integer/large


def test_value_check_integer_counts():
    prof = _profile([1000, 2000, 8000, 50000])
    assert value_check(prof, ROLES["n_counts"]) == 1.0
    assert value_check(prof, ROLES["pct_mt"]) == 0.0       # not in [0,1]


def test_value_check_genes_band():
    prof = _profile([200, 1500, 4000, 9000])
    assert value_check(prof, ROLES["n_genes"]) == 1.0

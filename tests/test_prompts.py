import pandas as pd
from stanmetacols.profile import profile_obs
from stanmetacols.prompts import SYSTEM_PROMPT, ADJUDICATION_SYSTEM_PROMPT, build_user_prompt, build_adjudication_prompt


def test_prompts_mention_roles_and_json():
    assert "json" in SYSTEM_PROMPT.lower()
    for token in ("sample", "pct_mt", "n_counts", "n_genes", "doublet"):
        assert token in SYSTEM_PROMPT
    assert "canonical" in ADJUDICATION_SYSTEM_PROMPT.lower()


def _d():
    return profile_obs(pd.DataFrame({"sample": ["A", "B"]}))


def test_user_prompt_includes_hint_block():
    p = build_user_prompt(_d(), ["sample"], hint="mito col is mt.frac")
    assert "User guidance" in p
    assert "mito col is mt.frac" in p


def test_user_prompt_omits_block_when_hint_empty():
    p = build_user_prompt(_d(), ["sample"])
    assert "User guidance" not in p


def test_adjudication_prompt_includes_hint():
    p = build_adjudication_prompt(_d(), {}, hint="counts are in total_umis")
    assert "User guidance" in p and "total_umis" in p


def test_prompts_discriminate_organ_and_tissue():
    sp = SYSTEM_PROMPT.lower()
    assert "organ" in sp and "tissue" in sp
    # the discrimination guidance, not just the roles-block listing
    assert "anatomical organ" in sp
    assert "sampled" in sp and "material" in sp

from stanmetacols.prompts import SYSTEM_PROMPT, ADJUDICATION_SYSTEM_PROMPT


def test_prompts_mention_roles_and_json():
    assert "json" in SYSTEM_PROMPT.lower()
    for token in ("sample", "pct_mt", "n_counts", "n_genes", "doublet"):
        assert token in SYSTEM_PROMPT
    assert "canonical" in ADJUDICATION_SYSTEM_PROMPT.lower()

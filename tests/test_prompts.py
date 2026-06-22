import pandas as pd

from stansample.prompts import SYSTEM_PROMPT, ALIAS_HINTS, build_user_prompt
from stansample.profile import profile_obs


def test_alias_hints_present():
    for name in ("sample_id", "donor", "orig.ident", "library", "gsm"):
        assert name in ALIAS_HINTS


def test_system_prompt_mentions_task():
    assert "sample" in SYSTEM_PROMPT.lower()
    assert "json" in SYSTEM_PROMPT.lower()


def test_user_prompt_embeds_digest():
    d = profile_obs(pd.DataFrame({"sample_id": ["S1", "S2"]}, index=["a", "b"]))
    prompt = build_user_prompt(d)
    assert "sample_id" in prompt
    assert "n_obs" in prompt

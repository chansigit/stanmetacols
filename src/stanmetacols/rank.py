"""Public orchestrator: build digest, rank per role (LLM stage 1 + heuristic
fallback). Numeric adjudication (stage 2) is wired in Task 6."""

from __future__ import annotations

from .schema import MetaColsResult, LLMUnavailable
from .profile import profile_obs
from .roles import ROLE_KEYS
from .llm import rank_with_llm
from .heuristic import rank_heuristic


def _extract(data):
    if hasattr(data, "obs_names") and hasattr(data, "obs"):
        return data.obs, list(data.obs_names)
    return data, list(data.index)


def rank_meta_columns(data, *, roles=None, use_llm: bool = True,
                      adjudicate: bool = True, provider: str = "anthropic",
                      model: str = "claude-opus-4-8", client=None,
                      base_url: str | None = None, api_key: str | None = None,
                      top_k: int | None = 5) -> MetaColsResult:
    role_keys = list(roles) if roles else list(ROLE_KEYS)
    obs, obs_names = _extract(data)
    digest = profile_obs(obs, obs_names)

    if use_llm:
        try:
            ranked = rank_with_llm(digest, role_keys, provider=provider,
                                   model=model, client=client,
                                   base_url=base_url, api_key=api_key)
            method = f"llm ({provider})"
        except LLMUnavailable as exc:
            ranked = rank_heuristic(digest, role_keys)
            method = f"heuristic (llm unavailable: {exc})"
    else:
        ranked = rank_heuristic(digest, role_keys)
        method = "heuristic"

    for k in ranked:
        ranked[k] = sorted(ranked[k], key=lambda c: c.score, reverse=True)
        if top_k and top_k > 0:
            ranked[k] = ranked[k][:top_k]
    return MetaColsResult(roles=ranked, method=method, digest=digest)

"""Public orchestrator: build digest, rank via LLM (with heuristic fallback)."""

from __future__ import annotations

from .schema import RankResult, LLMUnavailable
from .profile import profile_obs
from .llm import rank_with_llm
from .heuristic import rank_heuristic


def _extract(data):
    """Return (obs_dataframe, obs_names) for an AnnData or a bare DataFrame.

    AnnData exposes `.obs_names`; a bare DataFrame does not (and a DataFrame may
    even have a column literally named "obs"), so detect AnnData via obs_names —
    not via getattr(data, "obs").
    """
    if hasattr(data, "obs_names") and hasattr(data, "obs"):
        return data.obs, list(data.obs_names)
    return data, list(data.index)


def rank_sample_columns(data, *, use_llm: bool = True, model: str = "claude-opus-4-8",
                        client=None, top_k: int | None = 5) -> RankResult:
    obs, obs_names = _extract(data)
    digest = profile_obs(obs, obs_names)

    if use_llm:
        try:
            candidates = rank_with_llm(digest, model=model, client=client)
            method = "llm"
        except LLMUnavailable as exc:
            candidates = rank_heuristic(digest)
            method = f"heuristic (llm unavailable: {exc})"
    else:
        candidates = rank_heuristic(digest)
        method = "heuristic"

    candidates = sorted(candidates, key=lambda c: c.score, reverse=True)
    if top_k and top_k > 0:
        candidates = candidates[:top_k]
    return RankResult(candidates=candidates, method=method, digest=digest)

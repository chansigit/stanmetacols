"""Public orchestrator: build digest, rank via LLM (with heuristic fallback)."""

from .schema import RankResult, LLMUnavailable
from .profile import profile_obs
from .llm import rank_with_llm
from .heuristic import rank_heuristic


def _extract(data):
    """Return (obs_dataframe, obs_names) for an AnnData or a bare DataFrame."""
    obs = getattr(data, "obs", None)
    if obs is not None:
        obs_names = list(getattr(data, "obs_names", obs.index))
        return obs, obs_names
    return data, list(data.index)


def rank_sample_columns(data, *, use_llm: bool = True, model: str = "claude-opus-4-8",
                        client=None, top_k: int = 5) -> RankResult:
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
    if top_k:
        candidates = candidates[:top_k]
    return RankResult(candidates=candidates, method=method, digest=digest)

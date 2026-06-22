"""Deterministic fallback ranker over an ObsDigest. No network, no API key."""

from .schema import ObsDigest, Candidate
from .prompts import ALIAS_HINTS


def _norm(name: str) -> str:
    return name.lower().replace("_", "").replace(".", "").replace(" ", "")


_NORM_ALIASES = {_norm(a) for a in ALIAS_HINTS}


def _name_signal(name: str) -> float:
    n = _norm(name)
    if n in _NORM_ALIASES:
        return 1.0
    if any(alias in n for alias in _NORM_ALIASES):
        return 0.6
    return 0.0


def _cardinality_signal(n_unique: int, n_obs: int) -> float:
    if n_obs == 0 or n_unique < 2:
        return 0.0
    upper = max(50, int(n_obs * 0.2))
    return 1.0 if n_unique <= upper else 0.3


def _score(name, n_unique, balance, dtype, n_missing, n_obs, looks_like_barcode):
    name_sig = _name_signal(name)
    card = _cardinality_signal(n_unique, n_obs)
    penalty = 0.0
    if dtype == "float":
        penalty += 0.5
    if looks_like_barcode:
        penalty += 0.5
    if n_obs and (n_missing / n_obs) > 0.5:
        penalty += 0.3
    raw = 0.5 * name_sig + 0.25 * card + 0.25 * balance - penalty
    return max(0.0, min(1.0, raw)), name_sig, card


def rank_heuristic(digest: ObsDigest) -> list[Candidate]:
    out = []
    n_obs = digest.n_obs
    for c in digest.columns:
        if c.single_value or c.unique_per_cell:
            continue
        score, name_sig, card = _score(
            c.name, c.n_unique, c.balance, c.dtype, c.n_missing, n_obs, c.looks_like_barcode)
        if score <= 0:
            continue
        out.append(Candidate(
            column=c.name, kind="single", score=score, source="heuristic",
            reason=(f"name match={name_sig:.1f}, n_unique={c.n_unique}, "
                    f"balance={c.balance:.2f}")))
    for comp in digest.composite_candidates:
        # name signal averaged over members, modest discount vs. a single column
        name_sig = sum(_name_signal(col) for col in comp.columns) / len(comp.columns)
        card = _cardinality_signal(comp.n_unique, n_obs)
        raw = 0.85 * (0.5 * name_sig + 0.25 * card + 0.25 * comp.balance)
        score = max(0.0, min(1.0, raw))
        if score <= 0:
            continue
        out.append(Candidate(
            column=comp.label, kind="composite", score=score, source="heuristic",
            reason=(f"composite of {comp.columns}, n_unique={comp.n_unique}, "
                    f"balance={comp.balance:.2f}")))
    if digest.barcode is not None:
        bc = digest.barcode
        score = max(0.0, min(1.0, 0.45 * bc.balance + 0.1))
        if score > 0:
            out.append(Candidate(
                column=bc.label, kind="barcode", score=score, source="heuristic",
                reason=(f"barcode {bc.position} on '{bc.delimiter}', "
                        f"{bc.n_groups} groups, balance={bc.balance:.2f}")))
    out.sort(key=lambda c: c.score, reverse=True)
    return out

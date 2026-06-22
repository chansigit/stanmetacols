"""Prompts for the holistic ranking call and the numeric adjudication call."""

from __future__ import annotations

import json

from .schema import ObsDigest
from .roles import ROLES

_ROLE_DESCRIPTIONS = {
    "sample": "the sample each cell came from (grouping unit for per-sample QC / pseudobulk)",
    "pct_mt": "per-cell mitochondrial-gene fraction (a float in [0,1])",
    "pct_hb": "per-cell hemoglobin-gene fraction (a float in [0,1])",
    "doublet_score": "per-cell doublet detection score (a float in [0,1])",
    "n_counts": "total counts / UMIs per cell (a non-negative integer, large)",
    "n_genes": "number of genes detected per cell (a non-negative integer)",
}


def _roles_block(roles) -> str:
    lines = []
    for k in roles:
        aliases = ", ".join(ROLES[k].aliases[:6])
        lines.append(f"- {k}: {_ROLE_DESCRIPTIONS[k]}. Common names: {aliases}.")
    return "\n".join(lines)


SYSTEM_PROMPT = (
    "You are given a digest of an AnnData .obs table from a single-cell dataset. "
    "For EACH requested role, rank which .obs column best fills it. The roles:\n"
    + _roles_block(ROLES.keys()) + "\n\n"
    "All pct_* values and doublet_score are fractions in [0,1], NOT percents in "
    "[0,100]. Use the provided value stats (v_min/v_max/v_median, frac_unit, "
    "is_integer_valued) as evidence, not just the column name: counts/genes are "
    "non-negative integers (counts >> genes), the fractions live in [0,1]. "
    "Distinguish look-alikes: total_counts is the per-cell total, while "
    "total_counts_mt / total_counts_hb are subset counts and are NOT n_counts; "
    "n_genes_by_counts is n_genes, not n_counts.\n\n"
    "Return JSON only, matching the schema: a list of candidates, each with "
    "`role` (one of the requested roles), `column` (the .obs column name; for the "
    "sample role a composite may use its exact 'a + b' label or the barcode "
    "label), `kind` ('single'|'composite'|'barcode'), `score` in 0..1, and a "
    "one-sentence `reason`. Only include plausible candidates; omit a role "
    "entirely if no column fits."
)

ADJUDICATION_SYSTEM_PROMPT = (
    "You are disambiguating look-alike numeric columns in a single-cell .obs "
    "table. For each role below you are given several candidate columns with "
    "their value statistics. Pick the SINGLE canonical column for each role. "
    "Remember: pct_* and doublet_score are fractions in [0,1]; n_counts is the "
    "per-cell TOTAL counts/UMIs (largest), not a per-subset count like "
    "total_counts_mt; n_genes is genes detected per cell. Return JSON only: a "
    "list of verdicts, each with `role`, `column` (must be one of that role's "
    "given candidates), and a one-sentence `reason`."
)


def build_user_prompt(digest: ObsDigest, roles) -> str:
    return (
        "Requested roles: " + ", ".join(roles) + "\n\n"
        "Here is the .obs digest (JSON):\n\n"
        + json.dumps(digest.to_prompt_dict(), sort_keys=True, indent=2)
        + "\n\nRank the columns that fill each requested role."
    )


def build_adjudication_prompt(digest: ObsDigest, contention) -> str:
    # contention: dict[role_key -> list[Candidate]]
    by_col = {c.name: c for c in digest.columns}
    blocks = []
    for role, cands in contention.items():
        lines = [f"Role {role} — candidates:"]
        for cand in cands:
            p = by_col.get(cand.column)
            stats = ("" if p is None else
                     f" [v_min={p.v_min:.3g}, v_max={p.v_max:.3g}, "
                     f"v_median={p.v_median:.3g}, frac_unit={p.frac_unit:.2f}, "
                     f"is_integer_valued={p.is_integer_valued}]")
            lines.append(f"  - {cand.column}{stats}")
        blocks.append("\n".join(lines))
    return ("Pick the canonical column for each role.\n\n" + "\n\n".join(blocks)
            + "\n\nReturn one verdict per role.")

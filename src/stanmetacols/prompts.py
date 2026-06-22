"""Prompt text and the alias hint list for both the LLM and heuristic paths."""

import json

from .schema import ObsDigest

ALIAS_HINTS = [
    "sample", "sample_id", "sampleid", "donor", "donor_id", "patient",
    "patient_id", "subject", "individual", "specimen", "orig.ident",
    "library", "library_id", "gsm", "geo_accession", "srr", "batch",
    "channel", "well", "lane", "replicate",
]

SYSTEM_PROMPT = (
    "You are given a digest of an AnnData .obs table from a single-cell dataset. "
    "Rank which entries identify the SAMPLE each cell came from — the natural "
    "grouping unit used for per-sample QC, batch grouping, or pseudobulk. Do not "
    "distinguish biological (donor/patient) from technical (library/channel); "
    "surface whatever best partitions cells into samples.\n\n"
    "Prefer entries whose name and values look like sample/donor/library/GEO "
    "identifiers, with moderate cardinality (more than one group, far fewer than "
    "one-per-cell) and reasonable group balance. Common sample-like names include: "
    + ", ".join(ALIAS_HINTS) + ". "
    "Penalize per-cell-unique columns (cell barcodes/indices), single-value "
    "columns, and continuous measurements. Also consider the provided "
    "composite-key candidates (combinations of columns) and the barcode-derived "
    "grouping.\n\n"
    "Return JSON only, matching the provided schema: a list of candidates, each "
    "with the entry's name in `column` (for a composite use the exact 'a + b' "
    "label given; for the barcode grouping use its exact label), a `kind` of "
    "'single', 'composite', or 'barcode', a `score` in 0..1, and a one-sentence "
    "`reason`. Only include plausible candidates."
)


def build_user_prompt(digest: ObsDigest) -> str:
    return (
        "Here is the .obs digest (JSON):\n\n"
        + json.dumps(digest.to_prompt_dict(), sort_keys=True, indent=2)
        + "\n\nRank the candidates that identify the sample each cell came from."
    )

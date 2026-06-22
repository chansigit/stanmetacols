"""CLI: rank the sample column(s) of an .h5ad file."""

import argparse
import json
import sys
from dataclasses import asdict

from .rank import rank_sample_columns


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="stansample",
        description="Rank which .obs column identifies the sample each cell came from.")
    parser.add_argument("path", help="path to an .h5ad file")
    parser.add_argument("--no-llm", action="store_true",
                        help="force the offline heuristic ranker (no API call)")
    parser.add_argument("--top", type=int, default=5,
                        help="show top K candidates (default 5; 0 = all)")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument("--model", default="claude-opus-4-8",
                        help="LLM model id (default claude-opus-4-8)")
    args = parser.parse_args(argv)

    try:
        import anndata
        adata = anndata.read_h5ad(args.path, backed="r")
    except Exception as exc:
        print(f"error: cannot read {args.path!r}: {exc}", file=sys.stderr)
        return 1

    result = rank_sample_columns(
        adata, use_llm=not args.no_llm, model=args.model, top_k=args.top)

    if args.json:
        print(json.dumps(
            {"method": result.method,
             "candidates": [asdict(c) for c in result.candidates]}, indent=2))
    else:
        print(f"method: {result.method}")
        if not result.candidates:
            print("  (no plausible sample column found)")
        for c in result.candidates:
            print(f"  {c.score:.2f}  [{c.kind}/{c.source}]  {c.column}  — {c.reason}")

    return 0 if result.candidates else 2


if __name__ == "__main__":
    raise SystemExit(main())

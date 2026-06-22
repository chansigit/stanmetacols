"""CLI: identify metadata-role columns of an .h5ad file. JSON on stdout."""

import argparse
import json
import os
import sys
from dataclasses import asdict

from .rank import rank_meta_columns
from .roles import ROLE_KEYS


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="stanmetacols",
        description="Identify which .obs columns fill standard metadata roles. "
                    "Emits a JSON object on stdout.")
    parser.add_argument("path", help="path to an .h5ad file")
    parser.add_argument("--roles", default=None,
                        help="comma-separated subset of: " + ",".join(ROLE_KEYS))
    parser.add_argument("--no-llm", action="store_true",
                        help="force the offline heuristic ranker (no API call)")
    parser.add_argument("--top", type=int, default=5,
                        help="keep top K candidates per role (default 5; 0 = all)")
    parser.add_argument("--provider", choices=["anthropic", "openai"],
                        default="anthropic")
    parser.add_argument("--model", default="claude-opus-4-8")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--hint", default="",
                        help="optional free-text guidance for the LLM to locate "
                             "columns (LLM path only; ignored with --no-llm)")
    args = parser.parse_args(argv)

    roles = None
    if args.roles:
        roles = [r.strip() for r in args.roles.split(",") if r.strip()]
        bad = [r for r in roles if r not in ROLE_KEYS]
        if bad:
            print(f"error: unknown role(s): {', '.join(bad)}; "
                  f"valid: {', '.join(ROLE_KEYS)}", file=sys.stderr)
            return 1

    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None

    try:
        import anndata
        adata = anndata.read_h5ad(args.path, backed="r")
    except Exception as exc:
        print(f"error: cannot read {args.path!r}: {exc}", file=sys.stderr)
        return 1

    result = rank_meta_columns(
        adata, roles=roles, use_llm=not args.no_llm, hint=args.hint,
        provider=args.provider, model=args.model, base_url=args.base_url,
        api_key=api_key, top_k=args.top)

    print(json.dumps(
        {"method": result.method,
         "roles": {k: [asdict(c) for c in v] for k, v in result.roles.items()}},
        indent=2))

    any_found = any(v for v in result.roles.values())
    return 0 if any_found else 2


if __name__ == "__main__":
    raise SystemExit(main())

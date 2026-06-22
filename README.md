# stansample

Rank which AnnData `.obs` column — or composite key, or barcode-derived
grouping — identifies the **sample** each cell came from (the natural grouping
unit for per-sample QC, batch grouping, or pseudobulk). It **ranks**; it does
not decide.

Primary path: a single structured LLM call (`claude-opus-4-8`) over a compact,
deterministic digest of `.obs`. With no API key or no network, it falls back to
a deterministic heuristic ranker over the same digest — so it always returns an
answer offline.

## Install

```bash
pip install -e .            # core + heuristic only
pip install -e ".[llm]"     # add the LLM path (anthropic)
pip install -e ".[anndata]" # add .h5ad reading / AnnData inputs
```

The LLM path reads `ANTHROPIC_API_KEY` from the environment.

## CLI

```bash
stansample sample.h5ad                 # LLM if key present, else heuristic
stansample sample.h5ad --no-llm        # force offline heuristic
stansample sample.h5ad --top 0 --json  # all candidates, machine-readable
python -m stansample sample.h5ad       # equivalent module form
```

Exit codes: `0` at least one candidate, `2` none found, `1` on IO error.

## Library

```python
from stansample import rank_sample_columns

res = rank_sample_columns(adata)          # or pass a pandas .obs DataFrame
for c in res.candidates:                  # sorted by score, top 5 by default
    print(c.score, c.kind, c.column, "—", c.reason)
print(res.method)                         # "llm" or "heuristic (...)"
best = res.top()                          # highest-scored; you decide whether to use it
```

`rank_sample_columns(data, *, use_llm=True, model="claude-opus-4-8", client=None, top_k=5)`.
Never mutates the input; writes no files.

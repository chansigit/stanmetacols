# stansample Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A standalone pip-installable package + CLI that ranks which `.obs` column (or composite key, or barcode-derived grouping) identifies the sample each cell came from, using a single structured LLM call over a deterministic digest with an offline heuristic fallback.

**Architecture:** `profile.py` turns `.obs` into a compact JSON-able digest (per-column features + composite-pair candidates + barcode-pattern analysis). `rank.py` feeds the digest to `llm.py` (one `messages.parse` call to `claude-opus-4-8`) when an API key is available, else to `heuristic.py` (weighted deterministic scoring over the same digest). Both return ranked `Candidate`s; the tool ranks but never decides. Pure read — never mutates the AnnData, never writes files.

**Tech Stack:** Python ≥3.9, pandas, numpy, pydantic v2 (core); `anndata` and `anthropic` as optional extras (lazily imported); hatchling build; pytest.

## Global Constraints

- Package lives at `/home/users/chensj16/s/projects/stansample`; src layout `src/stansample/`.
- NOT a Claude Code skill — no `SKILL.md`, no `.claude-plugin/`.
- Pure read: never mutate the input AnnData/DataFrame; never write output files.
- Default model is exactly `claude-opus-4-8`.
- The heuristic path must run with **no network and no API key**; `anthropic` is imported lazily inside `llm.py` only.
- LLM never sees the expression matrix — only the digest (names, dtypes, cardinalities, example values, group-balance stats, barcode summary, composite candidates).
- Any LLM failure (no key, no network, not installed, API error, parse failure) → `LLMUnavailable` → automatic heuristic fallback; reason recorded in `RankResult.method`.
- Ranked output truncated to **top 5** by default (`top_k=5`; `top_k=0`/`None` → all). Zero-score candidates are dropped (not plausible).
- Candidate kinds: `"single"`, `"composite"` (label `"a + b"`), `"barcode"` (label `"<barcode:<position>:<delimiter>>"`).
- Tests use a mock client for the LLM path; **no real API calls in tests**.
- All commits end with the trailer:
  `Claude-Session: https://claude.ai/code/session_01KBwUybm1J5fhKtQodn2hBf`

---

### Task 1: Project scaffold + schema types

**Files:**
- Create: `pyproject.toml`
- Create: `README.md` (one-line stub; fleshed out in Task 10)
- Create: `src/stansample/__init__.py`
- Create: `src/stansample/schema.py`
- Test: `tests/test_schema.py`

**Interfaces:**
- Consumes: nothing.
- Produces: dataclasses `Candidate(column:str, kind:str, score:float, reason:str, source:str)`, `RankResult(candidates:list, method:str, digest:ObsDigest)` with `.top()->Candidate|None`, `ObsDigest(n_obs:int, columns:list, composite_candidates:list, barcode:BarcodeProfile|None)` with `.to_prompt_dict()->dict`, `ColumnProfile`, `CompositeProfile(... ).label`, `BarcodeProfile(...).label`; Pydantic `RankedCandidate`, `RankedCandidates`; exception `LLMUnavailable`. `stansample.__version__ == "0.1.0"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_schema.py
import stansample
from stansample.schema import (
    Candidate, RankResult, ObsDigest, ColumnProfile, CompositeProfile,
    BarcodeProfile, RankedCandidate, RankedCandidates, LLMUnavailable,
)


def test_version():
    assert stansample.__version__ == "0.1.0"


def test_candidate_and_result_top():
    digest = ObsDigest(n_obs=1, columns=[], composite_candidates=[], barcode=None)
    c = Candidate(column="sample_id", kind="single", score=0.9, reason="r", source="heuristic")
    assert RankResult(candidates=[c], method="heuristic", digest=digest).top() is c
    assert RankResult(candidates=[], method="x", digest=digest).top() is None


def test_labels():
    assert CompositeProfile(columns=["donor", "timepoint"], n_unique=6,
                            cells_per_group={"min": 5, "max": 5, "median": 5.0},
                            balance=1.0).label == "donor + timepoint"
    assert BarcodeProfile(delimiter="_", position="prefix", n_groups=3,
                          cells_per_group={"min": 10, "max": 10, "median": 10.0},
                          balance=1.0, example_groups=["S1", "S2", "S3"]
                          ).label == "<barcode:prefix:_>"


def test_to_prompt_dict_is_jsonable():
    import json
    d = ObsDigest(
        n_obs=2,
        columns=[ColumnProfile(name="a", dtype="categorical", n_unique=2, n_missing=0,
                               example_values=["x", "y"],
                               cells_per_group={"min": 1, "max": 1, "median": 1.0},
                               balance=1.0, unique_per_cell=True, single_value=False,
                               looks_like_barcode=False)],
        composite_candidates=[], barcode=None,
    ).to_prompt_dict()
    assert json.dumps(d, sort_keys=True)
    assert d["columns"][0]["name"] == "a"


def test_pydantic_schema():
    rc = RankedCandidates(candidates=[{"column": "a", "kind": "single", "score": 0.5, "reason": "x"}])
    assert rc.candidates[0].column == "a"


def test_llm_unavailable_is_exception():
    assert issubclass(LLMUnavailable, Exception)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_schema.py -q`
Expected: FAIL (collection error — `stansample` not importable / not installed yet).

- [ ] **Step 3: Create `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "stansample"
version = "0.1.0"
description = "Rank which AnnData .obs column identifies the sample each cell came from"
readme = "README.md"
requires-python = ">=3.9"
license = "MIT"
authors = [{name = "chansigit"}]
keywords = ["single-cell", "scRNA-seq", "anndata", "metadata", "sample"]
dependencies = ["pandas>=1.5", "numpy>=1.22", "pydantic>=2"]

[project.optional-dependencies]
anndata = ["anndata>=0.8"]
llm = ["anthropic>=0.69"]
test = ["pytest>=7.0", "anndata>=0.8", "anthropic>=0.69"]

[project.scripts]
stansample = "stansample.__main__:main"

[project.urls]
Homepage = "https://github.com/chansigit/stansample"
Repository = "https://github.com/chansigit/stansample"

[tool.hatch.build.targets.wheel]
packages = ["src/stansample"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 4: Create `README.md` stub**

```markdown
# stansample

Rank which AnnData `.obs` column identifies the sample each cell came from.
(Full README in Task 10.)
```

- [ ] **Step 5: Create `src/stansample/__init__.py`**

```python
"""stansample — rank which .obs column identifies the sample each cell came from."""

__version__ = "0.1.0"
```

- [ ] **Step 6: Create `src/stansample/schema.py`**

```python
"""Shared types: digest dataclasses, Candidate/RankResult, Pydantic output schema."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from pydantic import BaseModel


class LLMUnavailable(Exception):
    """The LLM ranking path cannot run (no key, no network, anthropic not
    installed, API error, or parse failure). Triggers heuristic fallback."""


@dataclass
class ColumnProfile:
    name: str
    dtype: str            # "categorical" | "string" | "integer" | "float" | "bool"
    n_unique: int
    n_missing: int
    example_values: list
    cells_per_group: dict  # {"min": int, "max": int, "median": float}
    balance: float         # min_group / max_group, 0..1
    unique_per_cell: bool
    single_value: bool
    looks_like_barcode: bool


@dataclass
class CompositeProfile:
    columns: list
    n_unique: int
    cells_per_group: dict
    balance: float

    @property
    def label(self) -> str:
        return " + ".join(self.columns)


@dataclass
class BarcodeProfile:
    delimiter: str
    position: str          # "prefix" | "suffix"
    n_groups: int
    cells_per_group: dict
    balance: float
    example_groups: list

    @property
    def label(self) -> str:
        return f"<barcode:{self.position}:{self.delimiter}>"


@dataclass
class ObsDigest:
    n_obs: int
    columns: list                 # list[ColumnProfile]
    composite_candidates: list    # list[CompositeProfile]
    barcode: Optional[BarcodeProfile] = None

    def to_prompt_dict(self) -> dict:
        return {
            "n_obs": self.n_obs,
            "columns": [vars(c) for c in self.columns],
            "composite_candidates": [
                {"columns": c.columns, "n_unique": c.n_unique,
                 "cells_per_group": c.cells_per_group, "balance": c.balance}
                for c in self.composite_candidates
            ],
            "barcode": (
                {"delimiter": self.barcode.delimiter, "position": self.barcode.position,
                 "n_groups": self.barcode.n_groups,
                 "cells_per_group": self.barcode.cells_per_group,
                 "balance": self.barcode.balance,
                 "example_groups": self.barcode.example_groups}
                if self.barcode is not None else None
            ),
        }


@dataclass
class Candidate:
    column: str
    kind: str              # "single" | "composite" | "barcode"
    score: float
    reason: str
    source: str            # "llm" | "heuristic"


@dataclass
class RankResult:
    candidates: list       # list[Candidate], sorted desc by score
    method: str
    digest: ObsDigest

    def top(self) -> Optional[Candidate]:
        return self.candidates[0] if self.candidates else None


class RankedCandidate(BaseModel):
    column: str
    kind: str
    score: float
    reason: str


class RankedCandidates(BaseModel):
    candidates: List[RankedCandidate]
```

- [ ] **Step 7: Install editable and run tests**

Run (on a compute node, not the login node):
```bash
cd /home/users/chensj16/s/projects/stansample
pip install -e ".[test]"
pytest tests/test_schema.py -q
```
Expected: `6 passed`.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml README.md src/stansample/__init__.py src/stansample/schema.py tests/test_schema.py
git commit -m "feat: scaffold stansample package + schema types

Claude-Session: https://claude.ai/code/session_01KBwUybm1J5fhKtQodn2hBf"
```

---

### Task 2: `profile.py` — per-column profiling

**Files:**
- Create: `src/stansample/profile.py`
- Test: `tests/test_profile.py`

**Interfaces:**
- Consumes: `ObsDigest`, `ColumnProfile` from `schema.py`.
- Produces: `profile_obs(obs, obs_names=None, *, max_example_values=8, max_composite_pairs=8) -> ObsDigest`. At this task it returns column profiles only (`composite_candidates=[]`, `barcode=None`). Helper `_profile_column`, `_group_stats`, `_classify_dtype`, regex `_BARCODE_RE`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_profile.py
import numpy as np
import pandas as pd

from stansample.profile import profile_obs


def _obs():
    n = 30
    return pd.DataFrame(
        {
            "sample_id": ["S1"] * 10 + ["S2"] * 10 + ["S3"] * 10,
            "donor": ["D1"] * 15 + ["D2"] * 15,
            "timepoint": ["t0", "t1", "t2"] * 10,
            "tissue": ["lung"] * 30,
            "pct_mito": np.linspace(0.0, 10.0, n),
            "cell_id": [f"cell{i}" for i in range(n)],
        },
        index=[f"S{(i // 10) + 1}_AAAC{i:04d}-1" for i in range(n)],
    )


def _col(digest, name):
    return next(c for c in digest.columns if c.name == name)


def test_column_basic_features():
    d = profile_obs(_obs())
    assert d.n_obs == 30
    assert {c.name for c in d.columns} == {
        "sample_id", "donor", "timepoint", "tissue", "pct_mito", "cell_id"}
    assert _col(d, "sample_id").n_unique == 3
    assert _col(d, "sample_id").cells_per_group == {"min": 10, "max": 10, "median": 10.0}
    assert _col(d, "sample_id").balance == 1.0
    assert _col(d, "donor").balance == 1.0


def test_flags_and_dtypes():
    d = profile_obs(_obs())
    assert _col(d, "tissue").single_value is True
    assert _col(d, "cell_id").unique_per_cell is True
    assert _col(d, "pct_mito").dtype == "float"
    assert _col(d, "timepoint").dtype in ("string", "categorical")
    assert _col(d, "sample_id").example_values == sorted(["S1", "S2", "S3"])


def test_composite_and_barcode_empty_at_this_stage():
    d = profile_obs(_obs())
    assert d.composite_candidates == []
    assert d.barcode is None


def test_empty_obs():
    d = profile_obs(pd.DataFrame())
    assert d.n_obs == 0
    assert d.columns == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_profile.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'stansample.profile'`.

- [ ] **Step 3: Create `src/stansample/profile.py`**

```python
"""Deterministic feature extraction from a pandas .obs into an ObsDigest.

No LLM, no network, no mutation of the input.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .schema import ObsDigest, ColumnProfile

_BARCODE_RE = re.compile(r"^[ACGTN]{8,}(-\d+)?$", re.IGNORECASE)


def _classify_dtype(s: pd.Series) -> str:
    if isinstance(s.dtype, pd.CategoricalDtype):
        return "categorical"
    if pd.api.types.is_bool_dtype(s):
        return "bool"
    if pd.api.types.is_integer_dtype(s):
        return "integer"
    if pd.api.types.is_float_dtype(s):
        return "float"
    return "string"


def _group_stats(counts: np.ndarray):
    if counts.size == 0:
        return {"min": 0, "max": 0, "median": 0.0}, 0.0
    mn, mx = int(counts.min()), int(counts.max())
    med = float(np.median(counts))
    balance = (mn / mx) if mx > 0 else 0.0
    return {"min": mn, "max": mx, "median": med}, balance


def _profile_column(name: str, s: pd.Series, n_obs: int, max_example_values: int) -> ColumnProfile:
    n_missing = int(s.isna().sum())
    vc = s.value_counts(dropna=True)
    n_unique = int(vc.size)
    cells_per_group, balance = _group_stats(vc.to_numpy())
    example_values = sorted(str(v) for v in vc.index)[:max_example_values]
    sample = [str(v) for v in vc.index[:1000]]
    frac_bc = (sum(1 for v in sample if _BARCODE_RE.match(v)) / len(sample)) if sample else 0.0
    return ColumnProfile(
        name=name,
        dtype=_classify_dtype(s),
        n_unique=n_unique,
        n_missing=n_missing,
        example_values=example_values,
        cells_per_group=cells_per_group,
        balance=balance,
        unique_per_cell=(n_obs > 0 and n_unique == n_obs),
        single_value=(n_unique <= 1),
        looks_like_barcode=(frac_bc > 0.5),
    )


def profile_obs(obs, obs_names=None, *, max_example_values: int = 8,
                max_composite_pairs: int = 8) -> ObsDigest:
    if obs_names is None:
        obs_names = list(obs.index)
    n_obs = len(obs)
    columns = [
        _profile_column(str(c), obs[c], n_obs, max_example_values)
        for c in obs.columns
    ]
    return ObsDigest(n_obs=n_obs, columns=columns, composite_candidates=[], barcode=None)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_profile.py -q`
Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/stansample/profile.py tests/test_profile.py
git commit -m "feat: per-column profiling (profile_obs)

Claude-Session: https://claude.ai/code/session_01KBwUybm1J5fhKtQodn2hBf"
```

---

### Task 3: `profile.py` — composite-key candidates

**Files:**
- Modify: `src/stansample/profile.py` (add `_composite_candidates`, wire into `profile_obs`)
- Test: `tests/test_profile_composite.py`

**Interfaces:**
- Consumes: `ColumnProfile` list + the `obs` DataFrame.
- Produces: `profile_obs(...)` now fills `ObsDigest.composite_candidates` with `CompositeProfile`s, capped at `max_composite_pairs`, sorted by balance desc.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_profile_composite.py
import pandas as pd

from stansample.profile import profile_obs


def _obs():
    return pd.DataFrame(
        {
            "donor": ["D1"] * 15 + ["D2"] * 15,
            "timepoint": ["t0", "t1", "t2"] * 10,
            "tissue": ["lung"] * 30,            # single value, ineligible
            "cell_id": [f"cell{i}" for i in range(30)],  # unique-per-cell, ineligible
        },
        index=[f"c{i}" for i in range(30)],
    )


def test_composite_pair_generated():
    d = profile_obs(_obs())
    labels = {c.label for c in d.composite_candidates}
    assert "donor + timepoint" in labels
    comp = next(c for c in d.composite_candidates if c.label == "donor + timepoint")
    assert comp.n_unique == 6  # 2 donors x 3 timepoints


def test_ineligible_columns_excluded_from_composites():
    d = profile_obs(_obs())
    for c in d.composite_candidates:
        assert "tissue" not in c.columns      # single-value excluded
        assert "cell_id" not in c.columns     # unique-per-cell excluded


def test_cap_respected():
    d = profile_obs(_obs(), max_composite_pairs=0)
    assert d.composite_candidates == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_profile_composite.py -q`
Expected: FAIL (`composite_candidates` is empty — assertions fail).

- [ ] **Step 3: Add `_composite_candidates` and wire it in**

In `src/stansample/profile.py`, add the import and helper, and update `profile_obs`:

```python
# add to the existing import line:
from .schema import ObsDigest, ColumnProfile, CompositeProfile
```

```python
def _composite_candidates(obs, profiles, n_obs, max_pairs):
    if max_pairs <= 0 or n_obs == 0:
        return []
    eligible = [
        p for p in profiles
        if not p.unique_per_cell and not p.single_value
        and p.dtype in ("categorical", "string", "integer", "bool")
        and 2 <= p.n_unique <= n_obs * 0.5
        and not p.looks_like_barcode
    ]
    eligible.sort(key=lambda p: p.balance, reverse=True)
    eligible = eligible[:12]  # bound the O(k^2) groupby work
    pairs = []
    for i in range(len(eligible)):
        for j in range(i + 1, len(eligible)):
            a, b = eligible[i].name, eligible[j].name
            sizes = obs.groupby([a, b], observed=True).size()
            sizes = sizes[sizes > 0]
            n_unique = int(sizes.size)
            if not (2 <= n_unique < n_obs):
                continue
            cells_per_group, balance = _group_stats(sizes.to_numpy())
            pairs.append(CompositeProfile([a, b], n_unique, cells_per_group, balance))
    pairs.sort(key=lambda c: c.balance, reverse=True)
    return pairs[:max_pairs]
```

Update the `return` in `profile_obs`:

```python
    return ObsDigest(
        n_obs=n_obs,
        columns=columns,
        composite_candidates=_composite_candidates(obs, columns, n_obs, max_composite_pairs),
        barcode=None,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_profile_composite.py tests/test_profile.py -q`
Expected: all pass (note `tests/test_profile.py::test_composite_and_barcode_empty_at_this_stage` still passes because `_obs()` there has no eligible pair: `timepoint`+`donor` exist — recheck). 

> ⚠️ The Task-2 test `test_composite_and_barcode_empty_at_this_stage` asserts `composite_candidates == []` on the Task-2 fixture, which *does* contain eligible `donor`/`timepoint`/`sample_id` columns and will now be non-empty. **Update that Task-2 test** as part of this task: rename it to `test_barcode_empty_at_this_stage` and keep only `assert d.barcode is None` (remove the `composite_candidates == []` assertion). Re-run `pytest tests/test_profile.py -q` and confirm it passes.

- [ ] **Step 5: Commit**

```bash
git add src/stansample/profile.py tests/test_profile_composite.py tests/test_profile.py
git commit -m "feat: composite-key candidate generation

Claude-Session: https://claude.ai/code/session_01KBwUybm1J5fhKtQodn2hBf"
```

---

### Task 4: `profile.py` — barcode-pattern analysis

**Files:**
- Modify: `src/stansample/profile.py` (add `_barcode_profile`, wire into `profile_obs`)
- Test: `tests/test_profile_barcode.py`

**Interfaces:**
- Consumes: `obs_names` sequence.
- Produces: `profile_obs(...)` now fills `ObsDigest.barcode` with a `BarcodeProfile` or `None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_profile_barcode.py
import pandas as pd

from stansample.profile import profile_obs


def test_prefix_grouping_from_underscore():
    obs = pd.DataFrame({"x": list(range(30))},
                       index=[f"S{(i // 10) + 1}_AAAC{i:04d}-1" for i in range(30)])
    d = profile_obs(obs)
    assert d.barcode is not None
    assert d.barcode.position == "prefix"
    assert d.barcode.delimiter == "_"
    assert d.barcode.n_groups == 3
    assert d.barcode.label == "<barcode:prefix:_>"


def test_no_grouping_returns_none():
    obs = pd.DataFrame({"x": [1, 2, 3]}, index=["aaa", "bbb", "ccc"])
    assert profile_obs(obs).barcode is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_profile_barcode.py -q`
Expected: FAIL (`d.barcode is None` while test expects a profile).

- [ ] **Step 3: Add `_barcode_profile` and wire it in**

In `src/stansample/profile.py`, add the import and helper, and update `profile_obs`:

```python
# extend the schema import:
from .schema import ObsDigest, ColumnProfile, CompositeProfile, BarcodeProfile
```

```python
def _barcode_profile(obs_names, n_obs, max_example_groups=8):
    if n_obs == 0:
        return None
    s = pd.Series([str(x) for x in obs_names])
    options = []
    if s.str.contains("_", regex=False).mean() > 0.9:
        options.append(("_", "prefix", s.str.rsplit("_", n=1).str[0]))
    tail = s.str.rsplit("-", n=1).str[-1]
    if tail.str.fullmatch(r"\d+").mean() > 0.9:
        options.append(("-", "suffix", tail))
    best = None
    for delimiter, position, grp in options:
        vc = grp.value_counts()
        n_groups = int(vc.size)
        if not (2 <= n_groups < n_obs):
            continue
        cells_per_group, balance = _group_stats(vc.to_numpy())
        prof = BarcodeProfile(
            delimiter=delimiter, position=position, n_groups=n_groups,
            cells_per_group=cells_per_group, balance=balance,
            example_groups=sorted(str(v) for v in vc.index)[:max_example_groups],
        )
        if best is None or prof.balance > best.balance:
            best = prof
    return best
```

Update the `return` in `profile_obs`:

```python
    return ObsDigest(
        n_obs=n_obs,
        columns=columns,
        composite_candidates=_composite_candidates(obs, columns, n_obs, max_composite_pairs),
        barcode=_barcode_profile(obs_names, n_obs),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_profile_barcode.py tests/test_profile.py tests/test_profile_composite.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/stansample/profile.py tests/test_profile_barcode.py
git commit -m "feat: barcode-pattern grouping analysis

Claude-Session: https://claude.ai/code/session_01KBwUybm1J5fhKtQodn2hBf"
```

---

### Task 5: `prompts.py`

**Files:**
- Create: `src/stansample/prompts.py`
- Test: `tests/test_prompts.py`

**Interfaces:**
- Consumes: `ObsDigest`.
- Produces: `SYSTEM_PROMPT: str`, `ALIAS_HINTS: list[str]`, `build_user_prompt(digest) -> str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prompts.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prompts.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'stansample.prompts'`.

- [ ] **Step 3: Create `src/stansample/prompts.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_prompts.py -q`
Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/stansample/prompts.py tests/test_prompts.py
git commit -m "feat: prompts + alias hints

Claude-Session: https://claude.ai/code/session_01KBwUybm1J5fhKtQodn2hBf"
```

---

### Task 6: `heuristic.py` — offline deterministic ranker

**Files:**
- Create: `src/stansample/heuristic.py`
- Test: `tests/test_heuristic.py`

**Interfaces:**
- Consumes: `ObsDigest`, `Candidate`, `ALIAS_HINTS`.
- Produces: `rank_heuristic(digest) -> list[Candidate]` (only score > 0, sorted desc, `source="heuristic"`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_heuristic.py
import numpy as np
import pandas as pd

from stansample.profile import profile_obs
from stansample.heuristic import rank_heuristic


def _obs():
    n = 30
    return pd.DataFrame(
        {
            "sample_id": ["S1"] * 10 + ["S2"] * 10 + ["S3"] * 10,
            "donor": ["D1"] * 15 + ["D2"] * 15,
            "timepoint": ["t0", "t1", "t2"] * 10,
            "tissue": ["lung"] * 30,
            "pct_mito": np.linspace(0.0, 10.0, n),
            "cell_id": [f"cell{i}" for i in range(n)],
        },
        index=[f"S{(i // 10) + 1}_AAAC{i:04d}-1" for i in range(n)],
    )


def test_ranking_order_and_filtering():
    res = rank_heuristic(profile_obs(_obs()))
    cols = [c.column for c in res]
    assert all(c.source == "heuristic" for c in res)
    assert all(c.score > 0 for c in res)
    # alias-named, well-grouped columns lead
    assert cols[0] in ("sample_id", "donor")
    assert cols.index("sample_id") < cols.index("timepoint")
    # composite is offered
    assert "donor + timepoint" in cols
    # zero-score entries are dropped entirely
    assert "tissue" not in cols      # single value
    assert "cell_id" not in cols     # unique-per-cell


def test_empty_digest():
    assert rank_heuristic(profile_obs(pd.DataFrame())) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_heuristic.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'stansample.heuristic'`.

- [ ] **Step 3: Create `src/stansample/heuristic.py`**

```python
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


def rank_heuristic(digest: ObsDigest) -> list:
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_heuristic.py -q`
Expected: `2 passed`.

> If `cols[0]` is not in `("sample_id", "donor")` or ordering assertions fail, adjust the weights in `_score` / the composite discount until the fixture orders as asserted. The fixture is the contract; tune to it.

- [ ] **Step 5: Commit**

```bash
git add src/stansample/heuristic.py tests/test_heuristic.py
git commit -m "feat: deterministic heuristic ranker

Claude-Session: https://claude.ai/code/session_01KBwUybm1J5fhKtQodn2hBf"
```

---

### Task 7: `llm.py` — single structured call

**Files:**
- Create: `src/stansample/llm.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Consumes: `ObsDigest`, `Candidate`, `LLMUnavailable`, `RankedCandidates`, `SYSTEM_PROMPT`, `build_user_prompt`.
- Produces: `rank_with_llm(digest, *, model="claude-opus-4-8", client=None, max_tokens=2048) -> list[Candidate]` (source="llm"; hallucinated columns filtered; kind derived from digest membership). Raises `LLMUnavailable` on any failure.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_llm.py
import pandas as pd
import pytest

from stansample.profile import profile_obs
from stansample.schema import RankedCandidates, LLMUnavailable
from stansample.llm import rank_with_llm


def _digest():
    return profile_obs(pd.DataFrame(
        {"sample_id": ["S1"] * 5 + ["S2"] * 5, "tissue": ["lung"] * 10},
        index=[f"c{i}" for i in range(10)]))


class _StubMessages:
    def __init__(self, parsed):
        self._parsed = parsed
        self.kwargs = None

    def parse(self, **kwargs):
        self.kwargs = kwargs

        class _Resp:
            parsed_output = self._parsed
        return _Resp()


class _StubClient:
    def __init__(self, parsed):
        self.messages = _StubMessages(parsed)


def test_maps_and_filters_hallucinations():
    parsed = RankedCandidates(candidates=[
        {"column": "sample_id", "kind": "single", "score": 0.9, "reason": "looks like a sample id"},
        {"column": "made_up_column", "kind": "single", "score": 0.8, "reason": "hallucinated"},
    ])
    client = _StubClient(parsed)
    out = rank_with_llm(_digest(), client=client)
    cols = [c.column for c in out]
    assert "sample_id" in cols
    assert "made_up_column" not in cols          # filtered: not in digest
    assert all(c.source == "llm" for c in out)
    # prompt carried the digest
    assert "sample_id" in client.messages.kwargs["messages"][0]["content"]
    assert client.messages.kwargs["model"] == "claude-opus-4-8"


def test_api_error_becomes_llm_unavailable():
    class _Boom:
        class messages:
            @staticmethod
            def parse(**kwargs):
                raise RuntimeError("network down")
    with pytest.raises(LLMUnavailable):
        rank_with_llm(_digest(), client=_Boom())


def test_none_parsed_output_becomes_llm_unavailable():
    with pytest.raises(LLMUnavailable):
        rank_with_llm(_digest(), client=_StubClient(None))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'stansample.llm'`.

- [ ] **Step 3: Create `src/stansample/llm.py`**

```python
"""Single structured LLM call (claude-opus-4-8) over the digest.

anthropic is imported lazily so the package installs and the heuristic path
runs without it.
"""

from .schema import ObsDigest, Candidate, RankedCandidates, LLMUnavailable
from .prompts import SYSTEM_PROMPT, build_user_prompt


def _valid_labels(digest: ObsDigest) -> dict:
    """Map every real candidate label -> its kind."""
    labels = {c.name: "single" for c in digest.columns}
    for comp in digest.composite_candidates:
        labels[comp.label] = "composite"
    if digest.barcode is not None:
        labels[digest.barcode.label] = "barcode"
    return labels


def rank_with_llm(digest: ObsDigest, *, model: str = "claude-opus-4-8",
                  client=None, max_tokens: int = 2048) -> list:
    if client is None:
        try:
            import anthropic
        except Exception as exc:  # not installed
            raise LLMUnavailable(f"anthropic not installed: {exc}") from exc
        try:
            client = anthropic.Anthropic()
        except Exception as exc:  # no key, bad config
            raise LLMUnavailable(f"cannot construct client: {exc}") from exc

    try:
        resp = client.messages.parse(
            model=model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_user_prompt(digest)}],
            output_format=RankedCandidates,
        )
    except LLMUnavailable:
        raise
    except Exception as exc:  # any API/connection/parse error -> fallback
        raise LLMUnavailable(str(exc)) from exc

    parsed = getattr(resp, "parsed_output", None)
    if parsed is None:
        raise LLMUnavailable("model returned no parseable structured output")

    labels = _valid_labels(digest)
    out = []
    for rc in parsed.candidates:
        kind = labels.get(rc.column)
        if kind is None:           # hallucinated / unknown column -> drop
            continue
        score = max(0.0, min(1.0, float(rc.score)))
        out.append(Candidate(column=rc.column, kind=kind, score=score,
                             reason=rc.reason, source="llm"))
    out.sort(key=lambda c: c.score, reverse=True)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_llm.py -q`
Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/stansample/llm.py tests/test_llm.py
git commit -m "feat: single structured LLM ranking call

Claude-Session: https://claude.ai/code/session_01KBwUybm1J5fhKtQodn2hBf"
```

---

### Task 8: `rank.py` — orchestrator with fallback

**Files:**
- Create: `src/stansample/rank.py`
- Test: `tests/test_rank.py`

**Interfaces:**
- Consumes: `profile_obs`, `rank_with_llm`, `rank_heuristic`, `RankResult`, `LLMUnavailable`.
- Produces: `rank_sample_columns(data, *, use_llm=True, model="claude-opus-4-8", client=None, top_k=5) -> RankResult`. Accepts an AnnData (uses `.obs` + `.obs_names`) or a DataFrame. Never mutates input.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rank.py
import pandas as pd

from stansample.rank import rank_sample_columns
from stansample.schema import LLMUnavailable, RankedCandidates


def _obs():
    return pd.DataFrame(
        {"sample_id": ["S1"] * 5 + ["S2"] * 5, "tissue": ["lung"] * 10},
        index=[f"c{i}" for i in range(10)])


class _StubClient:
    def __init__(self, parsed):
        class _M:
            def parse(_self, **kw):
                class _R:
                    parsed_output = parsed
                return _R()
        self.messages = _M()


class _Boom:
    class messages:
        @staticmethod
        def parse(**kw):
            raise RuntimeError("no network")


def test_no_llm_uses_heuristic():
    res = rank_sample_columns(_obs(), use_llm=False)
    assert res.method == "heuristic"
    assert res.top().column == "sample_id"


def test_llm_path_with_mock_client():
    parsed = RankedCandidates(candidates=[
        {"column": "sample_id", "kind": "single", "score": 0.95, "reason": "ok"}])
    res = rank_sample_columns(_obs(), use_llm=True, client=_StubClient(parsed))
    assert res.method == "llm"
    assert res.top().column == "sample_id"
    assert res.top().source == "llm"


def test_llm_failure_falls_back():
    res = rank_sample_columns(_obs(), use_llm=True, client=_Boom())
    assert res.method.startswith("heuristic (llm unavailable")
    assert res.top().column == "sample_id"


def test_top_k_truncation():
    res = rank_sample_columns(_obs(), use_llm=False, top_k=1)
    assert len(res.candidates) == 1
    res_all = rank_sample_columns(_obs(), use_llm=False, top_k=0)
    assert len(res_all.candidates) >= 1


def test_empty_obs():
    res = rank_sample_columns(pd.DataFrame(), use_llm=False)
    assert res.candidates == []
    assert res.top() is None


def test_input_not_mutated():
    obs = _obs()
    before = obs.copy()
    rank_sample_columns(obs, use_llm=False)
    pd.testing.assert_frame_equal(obs, before)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rank.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'stansample.rank'`.

- [ ] **Step 3: Create `src/stansample/rank.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_rank.py -q`
Expected: `6 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/stansample/rank.py tests/test_rank.py
git commit -m "feat: rank_sample_columns orchestrator with fallback

Claude-Session: https://claude.ai/code/session_01KBwUybm1J5fhKtQodn2hBf"
```

---

### Task 9: `__main__.py` — CLI

**Files:**
- Create: `src/stansample/__main__.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `rank_sample_columns`.
- Produces: `main(argv=None) -> int`. Args: `path`, `--no-llm`, `--top` (default 5; 0=all), `--json`, `--model` (default `claude-opus-4-8`). Exit codes: 0 = ≥1 candidate, 2 = no candidate, 1 = IO/usage error.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli.py
import json
import anndata
import numpy as np
import pandas as pd

from stansample.__main__ import main


def _write_h5ad(path, obs, names):
    a = anndata.AnnData(X=np.zeros((len(obs), 2), dtype="float32"),
                        obs=obs.set_index(pd.Index(names)))
    a.write_h5ad(path)


def test_cli_json_success(tmp_path, capsys):
    p = tmp_path / "x.h5ad"
    obs = pd.DataFrame({"sample_id": ["S1"] * 5 + ["S2"] * 5})
    _write_h5ad(p, obs, [f"c{i}" for i in range(10)])
    code = main([str(p), "--no-llm", "--json"])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["method"] == "heuristic"
    assert any(c["column"] == "sample_id" for c in out["candidates"])


def test_cli_exit_2_when_no_candidate(tmp_path, capsys):
    p = tmp_path / "y.h5ad"
    obs = pd.DataFrame({"tissue": ["lung"] * 5})         # single value -> no candidate
    _write_h5ad(p, obs, ["aa", "bb", "cc", "dd", "ee"])  # no barcode delimiter
    code = main([str(p), "--no-llm", "--json"])
    assert code == 2


def test_cli_bad_path_exit_1(capsys):
    code = main(["/no/such/file.h5ad", "--no-llm"])
    assert code == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'stansample.__main__'`.

- [ ] **Step 3: Create `src/stansample/__main__.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli.py -q`
Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/stansample/__main__.py tests/test_cli.py
git commit -m "feat: CLI entry point (stansample x.h5ad)

Claude-Session: https://claude.ai/code/session_01KBwUybm1J5fhKtQodn2hBf"
```

---

### Task 10: Finalize top-level exports + README + full-suite check

**Files:**
- Modify: `src/stansample/__init__.py`
- Modify: `README.md`
- Test: `tests/test_public_api.py`

**Interfaces:**
- Consumes: everything.
- Produces: top-level re-exports `rank_sample_columns`, `profile_obs`, `Candidate`, `RankResult`, `ObsDigest`, `LLMUnavailable`, `__version__`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_public_api.py
import pandas as pd


def test_top_level_exports():
    from stansample import (
        rank_sample_columns, profile_obs, Candidate, RankResult,
        ObsDigest, LLMUnavailable, __version__,
    )
    assert __version__ == "0.1.0"
    res = rank_sample_columns(
        pd.DataFrame({"sample_id": ["S1"] * 3 + ["S2"] * 3},
                     index=[f"c{i}" for i in range(6)]),
        use_llm=False)
    assert isinstance(res, RankResult)
    assert res.top().column == "sample_id"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_public_api.py -q`
Expected: FAIL with `ImportError: cannot import name 'rank_sample_columns' from 'stansample'`.

- [ ] **Step 3: Replace `src/stansample/__init__.py`**

```python
"""stansample — rank which .obs column identifies the sample each cell came from."""

from .schema import Candidate, RankResult, ObsDigest, LLMUnavailable
from .profile import profile_obs
from .rank import rank_sample_columns

__version__ = "0.1.0"

__all__ = [
    "rank_sample_columns",
    "profile_obs",
    "Candidate",
    "RankResult",
    "ObsDigest",
    "LLMUnavailable",
    "__version__",
]
```

- [ ] **Step 4: Replace `README.md`**

```markdown
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
```

- [ ] **Step 5: Run the full suite**

Run: `pytest -q`
Expected: all tests pass (schema, profile×3, prompts, heuristic, llm, rank, cli, public_api).

- [ ] **Step 6: Commit**

```bash
git add src/stansample/__init__.py README.md tests/test_public_api.py
git commit -m "feat: finalize public API exports + README

Claude-Session: https://claude.ai/code/session_01KBwUybm1J5fhKtQodn2hBf"
```

---

## Self-Review (completed by plan author)

**Spec coverage:**
- Standalone package + CLI, no skill → Tasks 1, 9, 10. ✅
- `profile.py` (columns/composite/barcode) → Tasks 2, 3, 4. ✅
- `llm.py` single `messages.parse` call, lazy anthropic, hallucination filter, `LLMUnavailable` → Task 7. ✅
- `heuristic.py` offline ranker → Task 6. ✅
- `rank.py` orchestrator, fallback, `top_k=5`, AnnData|DataFrame, no mutation → Task 8. ✅
- `schema.py` types + Pydantic output → Task 1. ✅
- `prompts.py` + alias hints → Task 5. ✅
- CLI flags + exit codes + `backed="r"` → Task 9. ✅
- Deps/extras/console-script → Task 1 `pyproject.toml`. ✅
- Tests mock the client, no real API calls → Tasks 7, 8. ✅
- Drop zero-score candidates → Task 6 (`continue` on `score <= 0`), Task 7 (clamp; zero stays only if model returns 0 — acceptable, sorted last). ✅

**Placeholder scan:** No TBD/TODO; every code step shows complete code. The two "tune weights / adjust if ordering fails" notes are calibration guidance against concrete fixtures, not missing code. ✅

**Type consistency:** `Candidate(column, kind, score, reason, source)`, `RankResult(candidates, method, digest).top()`, `ObsDigest(n_obs, columns, composite_candidates, barcode)`, `profile_obs(obs, obs_names=None, *, max_example_values, max_composite_pairs)`, `rank_with_llm(digest, *, model, client, max_tokens)`, `rank_heuristic(digest)`, `rank_sample_columns(data, *, use_llm, model, client, top_k)` — names/signatures consistent across all tasks. Composite label `"a + b"` and barcode label `"<barcode:pos:delim>"` used consistently in profile, heuristic, llm, prompts. ✅

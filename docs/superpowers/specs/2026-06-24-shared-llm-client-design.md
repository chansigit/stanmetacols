# Shared LLM client design spec

**Date:** 2026-06-24
**Scope:** Extract stanmetacols' LLM-calling logic into a self-contained, zero-
dependency module (`llm_client.py`) that other `stan*` projects can vendor, and
migrate `standissect` onto it. `stangene` is **out of scope** (it has no LLM code
today).

## 1. Goal

Today each project that calls an LLM re-implements the same "send (system, user)
→ get text → extract JSON → validate → object" loop in its own style. Extract the
**common denominator** into one self-contained file that:

- depends on **nothing beyond the Python standard library** (no `anthropic`,
  `openai`, or even `pydantic` as a hard import);
- exposes a small, schema-agnostic API any project can `import` or `cp`;
- is adopted by **stanmetacols** (replacing its duplicated provider plumbing) and
  **standissect** (replacing its hand-rolled urllib client + JSON parser),
  preserving each project's existing behavior.

## 2. Background — how the three projects call LLMs today

| Project | LLM mechanism | Structured output | Config | Errors |
|---|---|---|---|---|
| **stanmetacols** | `anthropic` SDK native `messages.parse` + `openai` SDK `chat.completions` | Pydantic schemas | `--provider`, `--model`, `--base-url`, `--api-key`; env `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `OPENAI_BASE_URL` | `LLMUnavailable` → heuristic fallback |
| **standissect** | stdlib `urllib.request` → Volcengine ARK OpenAI-compatible `/chat/completions` (no SDK) | hand-parsed JSON (`re` de-fence + `json.loads` + enum check → dataclass); **no Pydantic** | `--ark-endpoint`, `--ark-model`, `--ark-api-key-env` (default `ARK_API_KEY`) | `RuntimeError` → `RuleDiagnosisEngine` fallback; `.complete(system,user)->str` injection seam |
| **stangene** | **none** — zero LLM calls in source (LLM-ness is an external Claude Code skill) | — | — | — |

The two real consumers diverge: standissect is deliberately **stdlib-only**,
stanmetacols is **SDK + Pydantic + native anthropic parse**. The shared layer must
bridge both without forcing either to abandon its stance. The common seam already
exists in standissect: the `.complete(system, user) -> str` chat-client protocol.

## 3. Design decisions (user-approved)

1. **stdlib-first transport.** The universal client is a `urllib`-based
   OpenAI-compatible client — zero third-party deps. It covers ARK, OpenAI,
   DeepSeek, vLLM, Ollama, and Claude via Anthropic's OpenAI-compatible endpoint.
2. **`call_structured` takes a `parse: Callable[[dict], T]`, not a forced Pydantic
   schema.** stanmetacols passes `Schema.model_validate`; standissect passes its
   own `dict -> DiagnosisResult` function. The shared module therefore imports
   **no** third-party library — not even pydantic.
3. **Anthropic native `messages.parse` stays stanmetacols-local**, not in the
   shared file, so the shared file remains lean and dependency-free. stanmetacols
   keeps its strongest-structured-output guarantee for its default provider.
4. **Vendored single file**, copied identically into each repo (not a pip
   package). Trade-off: two copies to keep in sync; mitigated by a header comment
   naming the canonical source.
5. **Scope:** build + wire in stanmetacols; migrate standissect. `stangene`
   excluded (speculative — no LLM need today).

## 4. The shared module — `llm_client.py`

Self-contained. Imports only `json`, `urllib.request`, `urllib.error`, `re`,
`typing`. A header comment marks it as a vendored shared module and names the
canonical source (`stanmetacols/src/stanmetacols/llm_client.py`).

```python
class LLMUnavailable(Exception):
    """The structured LLM call could not be completed (no key, no network,
    HTTP error, empty/garbled reply, or schema-validation failure)."""


def extract_json(text: str | None) -> str:
    """Return the outermost JSON object/array in text, tolerating Markdown
    code fences and surrounding prose. Raise LLMUnavailable if none found."""


class OpenAICompatClient:
    """Minimal stdlib client for any OpenAI-compatible /chat/completions
    endpoint. No third-party SDK."""
    def __init__(self, base_url: str, api_key: str, model: str, *,
                 timeout: float = 60.0, temperature: float | None = None): ...
    def complete(self, system: str, user: str, *,
                 max_tokens: int | None = None) -> str:
        # POST {base_url}/chat/completions, Authorization: Bearer {api_key},
        # body {model, messages:[{system},{user}]}, plus temperature and
        # max_tokens ONLY when not None. Returns choices[0].message.content.
        # Any HTTP/URL/shape error -> LLMUnavailable.


def call_structured(client, system: str, user: str,
                    parse: "Callable[[dict|list], T]", *,
                    list_key: str | None = None,
                    max_tokens: int | None = None) -> "T":
    """text = client.complete(system, user, max_tokens=max_tokens);
    data = json.loads(extract_json(text));
    if isinstance(data, list) and list_key: data = {list_key: data};
    return parse(data). Any json/parse error -> LLMUnavailable."""
```

- `client` is duck-typed: anything with `.complete(system, user, *, max_tokens)`
  works (e.g. standissect's `CallableChatClient`, or a test stub).
- `base_url` is the API root **without** `/chat/completions`; the client appends
  it. (standissect's full `--ark-endpoint` is normalized by stripping a trailing
  `/chat/completions`.)
- Unit: `T` is whatever `parse` returns — a Pydantic model, a dataclass, a dict.

## 5. stanmetacols changes

### 5.1 `llm_client.py` (new)
The module above. Plus stanmetacols defines `__all__` and the package
`__init__.py` re-exports `OpenAICompatClient`, `call_structured`, `extract_json`,
`LLMUnavailable` so other code can `from stanmetacols import call_structured`.

### 5.2 `schema.py`
`LLMUnavailable` definition **moves** to `llm_client.py`. `schema.py` does
`from .llm_client import LLMUnavailable` and keeps it in its namespace, so every
existing `from .schema import LLMUnavailable` (rank.py, llm.py, tests, `__init__`)
keeps working. One-way dependency `schema → llm_client`; no cycle (llm_client
imports no stanmetacols module).

### 5.3 `llm.py` (rewritten thin)
- Add a local `_anthropic_parse(system, user, schema, *, model, client, max_tokens)`:
  lazy `import anthropic`, `client.messages.parse(..., output_format=schema)`,
  return `parsed_output`; errors → `LLMUnavailable`. (Keeps native structured
  output for the default provider; ~15 lines.)
- `rank_with_llm`: build prompt + pick `RankedCandidates`; if
  `provider == "anthropic"` → `_anthropic_parse(...)`; if `provider == "openai"` →
  `call_structured(OpenAICompatClient(base_url, api_key, model), SYSTEM_PROMPT,
  build_user_prompt(...), RankedCandidates.model_validate, list_key="candidates",
  max_tokens=2048)`; else `LLMUnavailable("unknown provider")`. Keep the existing
  post-processing (role filter + hallucination guard via digest labels →
  `Candidate`).
- `adjudicate_numeric`: same dispatch with `Adjudications` /
  `build_adjudication_prompt` / `list_key="verdicts"` / `max_tokens=1024`, then the
  existing verdict mapping.
- Delete the four duplicated `_call_anthropic` / `_call_openai` /
  `_call_*_adjudication` functions and the private `_extract_json`/`_parse_ranked`
  (now `llm_client.extract_json`).
- **Behavior preserved on the wire**: `temperature` and `max_tokens` are omitted
  from the request body when `None`. stanmetacols' openai path sets no temperature
  today → keep `temperature=None` (provider default, unchanged) and pass the same
  explicit `max_tokens` (2048/1024). So the actual request is byte-equivalent to
  today's openai call.
- **`client=` injection seam — deliberate, documented change on the openai path.**
  Public signatures of `rank_with_llm` / `adjudicate_numeric` are identical. For
  `provider="anthropic"`, `client=` is still an anthropic client used by
  `_anthropic_parse` (unchanged). For `provider="openai"`, the path no longer
  touches the OpenAI SDK at all — so `client=` now means "any object exposing
  `.complete(system, user, *, max_tokens) -> str`" (an `OpenAICompatClient`, a
  stub, or a `CallableChatClient`); when `client is None` an `OpenAICompatClient`
  is built from `base_url`/`api_key`/`model`. A pre-built **OpenAI SDK** client is
  no longer accepted on the openai path. Update the README's `client=` note
  accordingly. (Only tests inject here today.)

### 5.4 No changes to `rank.py` / `__main__.py`
They call `rank_with_llm` / `adjudicate_numeric` unchanged.

## 6. standissect changes (separate repo, own branch)

standissect is a separate git repo (`/scratch/users/chensj16/projects/standissect`,
branch `main`, currently with unrelated uncommitted edits). Work on a **new
branch**; do **not** sweep up the pre-existing uncommitted `README.md`/`__init__.py`
changes.

- **Vendor** the identical `llm_client.py` into standissect (same file).
- `ArkChatClient` → thin wrapper over `OpenAICompatClient` (or replaced by it).
  `--ark-endpoint` (full URL) → `base_url` by stripping a trailing
  `/chat/completions`; `--ark-model` → `model`; `--ark-api-key-env` → read env for
  `api_key`. The `.complete(system, user) -> str` seam and `CallableChatClient`
  duck-type are preserved (OpenAICompatClient satisfies the protocol).
- `parse_llm_result`: delegate transport + JSON extraction to
  `call_structured(client, system, user, _parse_diagnosis, max_tokens=...)`, where
  `_parse_diagnosis(data: dict) -> DiagnosisResult` keeps the existing
  `ALLOWED_CAUSES` enum validation and dataclass construction. **No Pydantic
  introduced**; standissect stays stdlib-only.
- Rule-engine fallback unchanged: `LLMDiagnosisEngine.diagnose`'s
  `except Exception` already catches `LLMUnavailable` (a plain `Exception`
  subclass), so `fallback_to_rule` behavior is preserved.

## 7. Testing

**stanmetacols**
- `test_llm_client.py` (new): drive `call_structured` and `extract_json` directly
  with a **stub client** (`.complete` returning canned text) and a **toy Pydantic
  schema** AND a **toy plain-dataclass parse function** — proving the module works
  for arbitrary `parse` with zero stanmetacols coupling. Cover: object reply, bare
  array + `list_key`, fenced JSON, empty/garbled reply → `LLMUnavailable`, HTTP
  error path of `OpenAICompatClient` via a stubbed opener → `LLMUnavailable`.
- `test_llm.py` (existing 13): must still pass. The openai-path tests currently
  inject a stub `client` exposing `.chat.completions.create`; after the rewrite the
  openai path goes through `OpenAICompatClient.complete`. Align those stubs to the
  new seam (inject a `.complete`-exposing stub, or an `OpenAICompatClient` whose
  `urlopen` is patched). Anthropic-path tests keep using a stub with
  `.messages.parse`. Behavior assertions unchanged.

**standissect**
- Its existing diagnosis tests (if any) must still pass. Add/adjust a test that
  injects a `.complete`-stub client through the existing seam and asserts the
  parsed `DiagnosisResult` is unchanged vs. the pre-migration path. Confirm the
  rule-fallback still triggers when the client raises.

## 8. Out of scope (YAGNI)

- No new providers, streaming, retries, async, or token accounting.
- No Pydantic dependency added to standissect.
- No pip-packaging of the shared module (vendored file only).
- No `stangene` LLM integration.
- stanmetacols' anthropic native-parse path is preserved, not unified into the
  OpenAI-compatible client.

## 9. Risks / notes

- **Two copies to sync.** Header comment names the canonical source; any change to
  `llm_client.py` must be copied to both repos. Acceptable per the chosen
  vendored-file model.
- **standissect uncommitted edits.** Branch from `main`; stage only migration
  files; leave the pre-existing `README.md`/`__init__.py` edits untouched.
- **Endpoint normalization.** standissect passes a full chat-completions URL today;
  the shared client wants a base URL. Normalize by stripping the known suffix; if a
  user passes a non-standard path, document that `base_url` should omit
  `/chat/completions`.

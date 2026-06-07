# ModelHub Integration Plan

## Goal

Use Byte internal ModelHub as the LLM provider for NHTS pandemic event-feature generation while preserving reproducibility, leakage controls, and schema validation.

## Current Constraint

- OpenRouter is treated as unavailable.
- OpenAI API key is not configured.
- Byte ModelHub can be used if a key and endpoint are provided.
- Current available model: `gpt5.5` only.

## Security Rule

Do not paste the ModelHub key into chat or commit it to the repository.

Preferred local setup:

```powershell
$env:MODELHUB_API_KEY = "<set locally>"
$env:MODELHUB_BASE_URL = "<modelhub endpoint>"
$env:MODELHUB_MODEL = "gpt5.5"
$env:MODELHUB_API_STYLE = "openai_chat"
```

Alternative local setup:

- Copy `.env.example` to `.env`.
- Fill local values in `.env`.
- Keep `.env` uncommitted.

## Required ModelHub Details

Before writing the request runner, confirm:

- Base URL.
- API path.
- Auth header and scheme.
- Whether the API is OpenAI-compatible.
- Whether it uses Responses API, Chat Completions, or a custom schema.
- Whether `response_format` or strict JSON schema is supported.
- Rate limits and timeout expectations.
- Internal data policy for sending anonymized cohort profiles.

## Experimental Design Change

Because only `gpt5.5` is available, we will not rely on external model-size comparison as the main ablation.

Replacement ablations:

- Full schema vs numeric-only schema.
- With short rationale vs without rationale.
- Cohort-level features vs sampled household-level features.
- Hand-coded pandemic features.
- Random event features.
- Local 4090 open-source LLM control if feasible.

## Implementation Order

1. Keep `validate_llm_event_features.py` as the hard output gate.
2. Add ModelHub provider configuration with environment-variable loading.
3. Add dry-run mode that reads prompts and reports token/record counts.
4. Add a 20-50 cohort pilot selector.
5. Run one request as a connectivity test only after endpoint details are confirmed.
6. Run the pilot with `max_concurrency=3`.
7. Audit raw outputs, invalid JSONL, and normalized CSV before full generation.

## Reproducibility Outputs

For every ModelHub run, save:

- provider: `byte_modelhub`
- model: `gpt5.5`
- API style and endpoint path, excluding secrets
- prompt file hash
- request parameters
- raw response JSONL
- normalized feature CSV
- invalid/error JSONL
- token/cost metadata if provided by ModelHub

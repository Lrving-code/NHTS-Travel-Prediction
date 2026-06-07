# CursorCC Integration Plan

## Goal

Use the local Cursor/CursorCC stack to generate LLM event features when direct OpenAI, OpenRouter, and Byte ModelHub routes are unavailable.

## Local Findings

Test date: 2026-06-07.

Installed commands:

- `cursorcc`
- `cursor`
- `claude`
- `ccr`

Observed architecture:

- `cursorcc` starts local `cursor-api`.
- `cursorcc` starts or uses `claude-code-router`.
- The local Cursor API endpoint is available at `http://127.0.0.1:3008/v1/chat/completions`.

Smoke tests:

- `cursorcc -p --model gpt-5.5` returned text successfully.
- `cursorcc -p --json-schema ...` returned empty stdout, so it is not reliable for structured-output enforcement.
- Direct local API calls worked with:
  - `gpt-5.5-high`
  - `gpt-5.5-medium`
  - `gpt-5.5-low`
- Direct local API call did not work with:
  - `gpt-5.4-mini`
- Direct local API accepted `response_format`, but the model did not obey JSON output in the smoke test, so strict provider-enforced schema support is not established.

## Recommendation

Use direct local Cursor API calls rather than the `cursorcc` CLI for dataset generation.

Rationale:

- Direct API returns machine-readable response envelopes.
- Easier to implement concurrency.
- Easier to log raw responses.
- Easier to resume by `cohort_id`.
- Avoids relying on CLI stdout behavior.

## Model Choice

Pilot:

- `gpt-5.5-low`
- `max_concurrency=1`
- 20-50 cohorts

Full run if pilot is stable:

- `gpt-5.5-high` if usage limits allow
- otherwise `gpt-5.5-medium`
- `max_concurrency=2-3` after rate-limit testing

## Output Enforcement

Do not rely on `response_format` or `--json-schema` for this route.

Use:

1. strict prompt instruction to output only JSON;
2. local JSON parser;
3. `validate_llm_event_features.py`;
4. one repair retry for invalid records;
5. invalid JSONL logging after retry failure.

## Risks

- Cursor subscription or local session limits may throttle or block batch generation.
- Provider-enforced structured output is not confirmed.
- The route may be harder to describe than a standard API provider.
- The local router may change model IDs or behavior after updates.

## Mitigations

- Keep raw input/output JSONL.
- Save local model ID exactly as used.
- Save router route metadata excluding secrets.
- Use conservative concurrency.
- Run a manually audited pilot before full generation.
- Keep non-LLM baselines and local 4090 ablations for robustness.

## Implementation Status

Updated on 2026-06-07:

- Implemented `src/generate_llm_event_features.py`.
- Implemented helper module `src/llm_event_generation.py`.
- Dry-run works on the 1,327-cohort prompt file.
- Python HTTP calls now bypass local proxy settings for the local Cursor API route.
- Single-cohort connectivity test passed with `gpt-5.5-low`.
- Five-cohort mini pilot passed with `gpt-5.5-low`.
- Validator result for five-cohort pilot: 5 valid, 0 invalid.
- Twenty-cohort pilot passed with `gpt-5.5-low`.
- Validator result for twenty-cohort pilot: 20 valid, 0 invalid.
- Low-token concurrency probe passed up to `max_concurrency=30` with `gpt-5.5-low`.
- High-concurrency probe result: 150 short requests, 150 successful, 0 errors.
- Full 1,327-cohort generation completed with `gpt-5.5-low` and `max_concurrency=15`.
- Full generation result: 1,327 successful, 0 generation errors.
- Full validation result: 1,327 valid, 0 invalid.

Generated pilot outputs:

- `outputs/llm_event_features/cursor_api_connectivity_gpt55_low/`
- `outputs/llm_event_features/cursor_api_pilot_gpt55_low_5/`
- `outputs/llm_event_features/cursor_api_pilot_gpt55_low_20/`
- `outputs/llm_event_features/cursor_api_concurrency_gpt55_low_30req/`
- `outputs/llm_event_features/cursor_api_full_gpt55_low_c15/`

Concurrency probe summary:

| Concurrency | Requests | Success | Errors | P95 latency | Throughput |
|---:|---:|---:|---:|---:|---:|
| 10 | 30 | 30 | 0 | 11.243s | 0.8238 req/s |
| 15 | 30 | 30 | 0 | 12.077s | 1.3066 req/s |
| 20 | 30 | 30 | 0 | 11.213s | 1.4008 req/s |
| 25 | 30 | 30 | 0 | 11.027s | 1.8636 req/s |
| 30 | 30 | 30 | 0 | 11.802s | 2.3749 req/s |

Interpretation:

- The local Cursor API route can tolerate 30 concurrent short requests in this environment.
- This does not automatically prove that 30 concurrent full NHTS prompts are safe, because full prompts are much longer and outputs require schema-valid JSON.
- Recommended next real-prompt concurrency test: 50 cohort prompts at `max_concurrency=10`, then 50 at `max_concurrency=15` if the first test has 0 errors.
- Full generation has now been completed at `max_concurrency=15`, so the next step is not more generation; it is joining validated cohort features back to household rows and running residual-adaptation experiments.

Recommended next command for a manually audited pilot:

```powershell
$env:CURSOR_API_AUTH_TOKEN = "<set locally>"
python src\generate_llm_event_features.py `
  --limit 20 `
  --selection even `
  --model gpt-5.5-low `
  --max-concurrency 1 `
  --max-attempts 2 `
  --output-dir outputs\llm_event_features\cursor_api_pilot_gpt55_low_20 `
  --overwrite `
  --request-sleep-seconds 1
python src\validate_llm_event_features.py `
  --input-jsonl outputs\llm_event_features\cursor_api_pilot_gpt55_low_20\llm_event_features.jsonl `
  --output-dir outputs\llm_event_features\cursor_api_pilot_gpt55_low_20\validated
```

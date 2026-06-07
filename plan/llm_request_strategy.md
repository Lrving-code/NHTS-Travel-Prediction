# LLM Request and Model Strategy

Decision date: 2026-06-07

## Goal

Define how to call LLMs for event-feature generation in a reproducible, auditable, and cost-controlled way before implementing API calls.

## Source Check

Official docs checked on 2026-06-07:

- OpenAI Models: https://developers.openai.com/api/docs/models
- OpenAI Latest Model Guide: https://developers.openai.com/api/docs/guides/latest-model
- OpenAI Batch API: https://developers.openai.com/api/docs/guides/batch
- OpenAI Structured Outputs: https://developers.openai.com/api/docs/guides/structured-outputs
- OpenRouter Models API: https://openrouter.ai/docs/api/api-reference/models/get-models

Current OpenAI docs recommend `gpt-5.5` as the starting point for complex reasoning, with `gpt-5.4-mini` and `gpt-5.4-nano` as lower-cost/lower-latency variants. Structured Outputs are preferred over JSON mode when schema adherence matters, and Batch API supports JSONL-based offline processing.

## Current Decision

Do not immediately run all 1,327 cohort prompts through an LLM.

Use a staged workflow:

1. **Pilot audit:** 20-50 representative cohorts, synchronous or low-concurrency calls.
2. **Prompt/schema revision:** inspect JSON validity, feature stability, leakage behavior, and rationale quality.
3. **Small controlled batch:** 200-300 cohorts, moderate concurrency.
4. **Full run:** all 1,327 cohorts through a fixed model snapshot/config.
5. **Ablation run:** compare at least one cheaper model or local/open model if feasible.

Update on 2026-06-07:

- Treat OpenRouter as unavailable by user instruction, even if a stale local environment variable is present.
- Use Byte internal ModelHub as the preferred external LLM provider if credentials and endpoint details are available.
- Current ModelHub constraint: only `gpt5.5` is available.
- Do not ask the user to paste secrets into chat. Use `MODELHUB_API_KEY` in the shell or a local ignored `.env` file.

Update on 2026-06-07 after ModelHub network constraint:

- Byte ModelHub appears to require Byte intranet access, so it is not the current runnable route on this machine.
- The local `cursorcc` stack is installed and can reach Cursor's local API.
- Use the local Cursor API route for pilot experiments if no direct OpenAI/OpenRouter/ModelHub API route is available.
- Direct Cursor API endpoint tested: `http://127.0.0.1:3008/v1/chat/completions`.
- Direct Cursor API model IDs tested as working: `gpt-5.5-high`, `gpt-5.5-medium`, `gpt-5.5-low`.
- Direct Cursor API model ID tested as not working: `gpt-5.4-mini`.
- `response_format`/JSON schema parameters were accepted but not enforced in the test, so local validation and retry remain mandatory.

## Why Not Full Parallel Immediately

The LLM output becomes part of the scientific method. If outputs are noisy, invalid, or leak target information, the downstream residual adapter can become hard to interpret.

We need:

- prompt audit logs
- exact model ID
- exact request parameters
- schema validation
- retry logs
- cost/token accounting
- raw outputs preserved

## Concurrency Strategy

Concurrency is allowed, but it is not the first scientific step.

The first implementation should support concurrent requests, but the default run mode must be conservative and auditable:

- pilot default: `max_concurrency=3`
- small-batch default: `max_concurrency=5`
- hard cap before manual review: `max_concurrency=10`
- full production run: prefer Batch API over live concurrent calls when the provider supports it

### Preferred for final large run

Use a batch API when available for the chosen provider.

Rationale:

- better reproducibility
- lower cost if provider discounts batch processing
- fewer rate-limit problems
- clear input/output JSONL artifacts

### Preferred for pilot and debugging

Use asynchronous concurrent requests with a conservative worker limit:

- start with `max_concurrency=3`
- increase to `5-10` only after stable responses
- exponential backoff on 429/5xx
- per-request timeout
- idempotent resume by `cohort_id`
- no silent fallback to a different model

## Model Selection

### Recommended primary model

Use a strong reasoning model as the **teacher** for the first serious run.

Current OpenAI docs list `gpt-5.5` as the flagship model for complex reasoning and coding, while `gpt-5.4-mini` and `gpt-5.4-nano` are cheaper/lower-latency variants. For this project, the LLM task is semantic event reasoning rather than text classification, so the first high-quality run should use the stronger model.

Primary recommendation:

- `gpt-5.5`
- temperature: `0`
- structured JSON schema output
- fixed prompt version
- fixed schema version

### Cost-efficient comparison model

Use a smaller model to test whether the semantic features are robust and whether we can scale cheaply.

Candidate:

- `gpt-5.4-mini`
- temperature: `0`
- same schema
- same prompts

### Historical OpenRouter route

OpenRouter was considered earlier, but it is no longer the active route by user instruction. If it is reconsidered later, query the OpenRouter Models API before running to verify the exact model IDs and whether `structured_outputs` is supported.

Likely naming pattern:

- `openai/gpt-5.5`
- `openai/gpt-5.4-mini`

But the script should not hardcode these without a model-availability check.

Verified on 2026-06-07 through the OpenRouter Models API:

- `openai/gpt-5.5` exists and supports `structured_outputs`.
- `openai/gpt-5.4-mini` exists and supports `structured_outputs`.
- `openai/gpt-5.4-nano` exists and supports `structured_outputs`.

Historical local credential state recorded on 2026-06-07 before the route change:

- `OPENAI_API_KEY`: not configured.
- `OPENROUTER_API_KEY`: configured.
- OpenRouter key usage: `0`.
- OpenRouter key-level credit limit: `null`.
- OpenRouter key reports `is_free_tier=true`, so account credits should be confirmed before paid-model runs.

Interpretation:

- Codex conversation usage is sufficient for coding, planning, debugging, and running local scripts.
- It should not be treated as the budget for generating the research dataset.
- LLM feature generation should use provider API credits through the active external provider, currently Byte ModelHub if approved and configured.

### Byte ModelHub route

This is now the preferred route because OpenRouter should be treated as unavailable.

Required local environment variables:

- `MODELHUB_API_KEY`
- `MODELHUB_BASE_URL`
- `MODELHUB_MODEL=gpt5.5`
- `MODELHUB_API_STYLE`

Integration assumptions to verify before implementation:

- whether ModelHub is OpenAI-compatible;
- whether the endpoint is Responses-style, Chat Completions-style, or custom HTTP;
- whether it supports strict schema output directly;
- exact auth header format;
- request/minute and token/minute limits;
- whether internal data policy permits sending cohort-level survey profiles.

If only `gpt5.5` is available, the experiment remains valid. The model-comparison ablation changes from "strong vs cheap external model" to:

- `gpt5.5` full schema vs numeric-only schema;
- `gpt5.5` with rationale vs without rationale;
- cohort-level generation vs household/sample-level generation;
- local 4090 open-source model as a negative/control ablation if feasible;
- non-LLM baselines: target-mean calibration, covariate reweighting, and hand-coded pandemic features.

### CursorCC / Local Cursor API route

This is the current runnable fallback when ModelHub is not reachable outside Byte intranet.

Two access layers were tested:

1. `cursorcc -p --model gpt-5.5 ...`
2. Direct local HTTP endpoint: `http://127.0.0.1:3008/v1/chat/completions`

Findings:

- Plain `cursorcc -p --model gpt-5.5` works for text output.
- `cursorcc -p --json-schema ...` returned empty stdout in the smoke test, so it should not be used as the schema-enforcement layer.
- Direct local HTTP calls work with `gpt-5.5-high`, `gpt-5.5-medium`, and `gpt-5.5-low`.
- Direct local HTTP calls preserve raw JSON responses and are easier to use for concurrency, resume, retries, and logging.
- Direct local HTTP calls accepted `response_format`, but the returned content did not obey JSON in the smoke test, so provider-enforced structured output is not established.

Recommended Cursor route:

- Pilot model: `gpt-5.5-low`
- Final generation model: `gpt-5.5-high` if usage limits allow, otherwise `gpt-5.5-medium`
- Output enforcement: prompt-only JSON instruction plus `validate_llm_event_features.py`
- Retry rule: one repair prompt for invalid JSON/schema, then write to invalid JSONL
- Concurrency: start at `max_concurrency=1-2`; increase only after no rate-limit or blank-output failures

Important limitation:

This route depends on the user's Cursor/CursorCC subscription/session and local router. It is pragmatic for data generation, but less clean than a normal provider API key. The paper should report the model family and provider route carefully and keep raw outputs for audit.

## Token and Cost Scale

Current prompt file:

- File: `outputs/llm_event_features/household_cohort_prompts.jsonl`
- Cohort prompts: `1,327`
- Approximate total input tokens: `1.86M`
- Approximate average input tokens per prompt: `1.4K`

Rough full-run cost estimates, assuming `0.40M-0.66M` output tokens:

| Model | Input cost estimate | Output cost estimate | Total estimate |
|---|---:|---:|---:|
| `openai/gpt-5.5` | ~$9.30 | ~$12-$20 | ~$21-$30 |
| `openai/gpt-5.4-mini` | ~$1.40 | ~$1.80-$3.00 | ~$3-$5 |

These are planning estimates, not billing guarantees. The request runner must compute actual token usage and cost from provider responses.

### Local model route

RTX 4090 can be used for local LLM ablations, but this should not be the primary source of event priors unless quality is verified.

Potential role:

- negative/control comparison
- cost-free ablation
- reproducibility appendix

Risk:

- weaker event reasoning
- weaker schema adherence
- more prompt sensitivity

## Structured Output Requirement

Use strict JSON schema output whenever supported.

The output must include only:

- `trip_suppression_risk`
- `remote_work_substitution_likelihood`
- `transit_avoidance_likelihood`
- `online_delivery_substitution_likelihood`
- `post_pandemic_recovery_sensitivity`
- `primary_event_mechanism`
- `short_explanation`
- `confidence`

All numeric scores must be in `[0, 1]`.

## Reproducibility Rules

For every run, save:

- provider
- model ID
- model snapshot if available
- request parameters
- prompt schema version
- input JSONL hash
- raw response JSONL
- normalized feature CSV
- invalid/error JSONL
- token usage/cost log

## Leakage Rules

Prompts must not include:

- `CNTTDHH`
- sample weights
- household identifiers
- 2022 aggregate target statistics
- post-hoc 2022 NHTS outcome summaries

LLM output must not include:

- direct trip-count predictions
- inferred target labels
- references to hidden labels or aggregate target distribution

## Experimental Use

LLM features are not final predictions.

They are event-response priors used by a supervised residual adapter:

```text
y_hat_final = f_tabular(X) + g_residual(X, z_llm)
```

## Immediate Next Implementation

Before API calls:

1. Finish and test `validate_llm_event_features.py`.
2. Add a provider-agnostic request runner interface.
3. Add a dry-run mode that estimates token volume and cost.
4. Add a pilot selector for 20-50 representative cohorts.
5. Only then run the first LLM pilot.

Implementation gate:

- Do not call an external LLM until the validator passes on a controlled sample.
- Do not run all 1,327 cohorts until the 20-50 cohort pilot has been manually audited.
- Do not silently swap models after failures; failures must be logged with the original model ID and provider.

# Manual Reasonableness Audit

Audit date: 2026-06-07

## Scope

- Output directory: `outputs/llm_event_features/cursor_api_pilot_gpt55_low_20/`
- Model: `gpt-5.5-low`
- Provider route: local Cursor API
- Records reviewed: 20
- Validator result: 20 valid, 0 invalid

## Verdict

The 20-cohort pilot is reasonable enough to proceed to a larger pilot or full cohort generation.

## Checks

| Check | Result |
|---|---:|
| Required schema fields present | Pass |
| Numeric scores in `[0, 1]` | Pass |
| Generation errors | 0 |
| Validation errors | 0 |
| Rule-based consistency flags | 0 / 20 |
| Forbidden leakage terms found | 0 |

## Consistency Observations

- Cohorts with zero workers receive low remote-work substitution scores.
- Multi-worker cohorts receive moderate to high remote-work substitution scores.
- Zero-vehicle and rail-accessible urban cohorts receive high transit-avoidance scores.
- Vehicle-rich and no-rail cohorts receive low transit-avoidance scores.
- Zero-vehicle cohorts receive higher trip-suppression scores.
- Vehicle-rich no-rail cohorts receive lower trip-suppression scores.
- Explanations are mechanism-oriented and do not directly predict household trip counts.

## Potential Weaknesses

- Some `primary_event_mechanism` strings are too verbose and read more like full explanations than compact categories.
- Confidence scores are compressed into a narrow range, approximately `0.62-0.74`.
- The local Cursor API route does not provide verified provider-enforced structured output, so local validation must remain mandatory.

## Recommendation

Proceed with the next generation step.

Preferred next step:

- Run a 50-cohort pilot with the same `gpt-5.5-low` route and `max_concurrency=1`.

Acceptable alternative:

- Run the full 1,327 cohorts with `gpt-5.5-low` first, then optionally rerun with `gpt-5.5-high` if the low-model features help downstream residual adaptation.

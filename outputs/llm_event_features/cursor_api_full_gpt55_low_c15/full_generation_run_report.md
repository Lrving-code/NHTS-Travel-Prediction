# Full LLM Event Feature Generation Report

Run date: 2026-06-07

## Configuration

- Provider route: local Cursor API
- Model: `gpt-5.5-low`
- Max concurrency: 15
- Max attempts per cohort: 2
- Prompt file: `outputs/llm_event_features/household_cohort_prompts.jsonl`

## Generation Result

- Selected cohorts: 1,327
- Successful cohorts: 1,327
- Generation errors: 0
- Raw response records: 1,327
- Repair retries used: 0

## Validation Result

- Valid records: 1,327
- Invalid records: 0
- Normalized feature CSV: `validated/llm_event_features_normalized.csv`

## Token and Latency Summary

- Prompt tokens total: 2,356,173
- Completion tokens total: 299,131
- Total tokens: 2,655,304
- Mean prompt tokens per request: 1,778.2
- Mean completion tokens per request: 225.8
- Mean latency: 11.989 seconds
- Median latency: 11.568 seconds
- Max latency: 128.619 seconds

## Feature Distribution

| Feature | Mean | Min | Median | Max |
|---|---:|---:|---:|---:|
| `trip_suppression_risk` | 0.4559 | 0.2400 | 0.4200 | 0.7800 |
| `remote_work_substitution_likelihood` | 0.3806 | 0.0300 | 0.4600 | 0.7800 |
| `transit_avoidance_likelihood` | 0.3147 | 0.0400 | 0.1800 | 0.8300 |
| `online_delivery_substitution_likelihood` | 0.5058 | 0.2800 | 0.5000 | 0.7400 |
| `post_pandemic_recovery_sensitivity` | 0.4711 | 0.2200 | 0.4600 | 0.7700 |
| `confidence` | 0.6586 | 0.4200 | 0.6700 | 0.7800 |

## Next Step

Join the validated cohort-level features to 2022 household rows and run residual-adaptation experiments.

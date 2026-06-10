# Local/Open-Source LLM Prior Replication Report

- Environment status: `READY`
- Model id: `Qwen/Qwen2.5-1.5B-Instruct`
- Input prompt file: `outputs\llm_event_features\household_cohort_prompts.jsonl`
- Selected cohorts: `32`
- Successful local priors: `32`
- Invalid/error records: `0`
- Output directory: `outputs\local_llm_prior_replication`

## Local vs GPT-Reference Prior Agreement

| Feature | n | MAE | Pearson | Spearman |
|---|---:|---:|---:|---:|
| trip_suppression_risk | 32 | 0.2538 | -0.2319 | -0.1053 |
| remote_work_substitution_likelihood | 32 | 0.3507 | -0.0916 | -0.1201 |
| transit_avoidance_likelihood | 32 | 0.2948 | -0.1524 | -0.1278 |
| online_delivery_substitution_likelihood | 32 | 0.2239 | -0.2465 | -0.2437 |
| post_pandemic_recovery_sensitivity | 32 | 0.2321 | 0.1026 | 0.1356 |
| confidence | 32 | 0.2185 | 0.3353 | 0.1127 |

## Interpretation

The local open-source model produced valid structured priors for `32` cohorts with mean feature MAE `0.2623` versus the GPT-reference priors. Treat this as a reproducibility and sensitivity control rather than a drop-in replacement for the main prior source unless a larger cohort run shows stable agreement.

## Paper Use

This artifact supports a completed small-sample open-source GPU LLM control.
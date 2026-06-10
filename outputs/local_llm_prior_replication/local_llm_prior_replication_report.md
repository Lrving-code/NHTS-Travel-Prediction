# Local/Open-Source LLM Prior Replication Report

- Environment status: `READY`
- Model id: `Qwen/Qwen2.5-1.5B-Instruct`
- Input prompt file: `outputs\llm_event_features\household_cohort_prompts.jsonl`
- Selected cohorts: `4`
- Successful local priors: `4`
- Invalid/error records: `0`
- Output directory: `outputs\local_llm_prior_replication`

## Local vs GPT-Reference Prior Agreement

| Feature | n | MAE | Pearson | Spearman |
|---|---:|---:|---:|---:|
| trip_suppression_risk | 4 | 0.3250 | -0.7241 | -0.8333 |
| remote_work_substitution_likelihood | 4 | 0.3600 | -0.7451 | -0.8944 |
| transit_avoidance_likelihood | 4 | 0.3850 | -0.5222 | -0.5443 |
| online_delivery_substitution_likelihood | 4 | 0.2325 | -0.7842 | -0.7379 |
| post_pandemic_recovery_sensitivity | 4 | 0.2625 | 0.9183 | 0.8944 |
| confidence | 4 | 0.2675 | -0.5774 | -0.5774 |

## Interpretation

The local open-source model produced valid structured priors for `4` cohorts with mean feature MAE `0.3054` versus the GPT-reference priors. Treat this as a reproducibility and sensitivity control rather than a drop-in replacement for the main prior source unless a larger cohort run shows stable agreement.

## Paper Use

This artifact supports a completed small-sample open-source GPU LLM control.
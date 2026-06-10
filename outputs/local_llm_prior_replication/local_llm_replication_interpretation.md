# Local Open-Source LLM Replication Interpretation

This note interprets the 32-cohort CUDA run recorded in
`local_llm_prior_replication_report.md`.

## Run Status

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Hardware: `NVIDIA GeForce RTX 4090`
- Selected cohorts: `32`
- Successful structured priors: `32`
- Invalid/error records: `0`
- Approximate generation latency: about 4.3-5.2 seconds per cohort in the logged run.

## Agreement With GPT-Reference Priors

| Feature | MAE | Pearson | Spearman |
|---|---:|---:|---:|
| trip_suppression_risk | 0.2538 | -0.2319 | -0.1053 |
| remote_work_substitution_likelihood | 0.3507 | -0.0916 | -0.1201 |
| transit_avoidance_likelihood | 0.2948 | -0.1524 | -0.1278 |
| online_delivery_substitution_likelihood | 0.2239 | -0.2465 | -0.2437 |
| post_pandemic_recovery_sensitivity | 0.2321 | 0.1026 | 0.1356 |
| confidence | 0.2185 | 0.3353 | 0.1127 |

Mean feature MAE is `0.2623`.

## Failure Mode

The local model successfully follows the JSON schema, but its event scores are less discriminative than the GPT-reference priors. Several local output vectors repeat across many cohorts; for example, the pattern
`0.9 / 0.8 / 0.7 / 0.6 / 0.5 / 0.95` appears eight times, and
`0.5 / 0.7 / 0.3 / 0.2 / 0.8 / 0.9` appears seven times in the 32-cohort sample.

This suggests that the 1.5B local model is good enough for a reproducibility smoke test and structured-output control, but it is not yet strong enough to replace the main GPT-reference prior source. The weak or negative rank correlations are especially important: the main adapter depends on event-prior ranking and calibration, not only schema validity.

## Paper-Safe Wording

Use:

> We add a CUDA-enabled local Qwen control on 32 cohorts. It validates that the prior-generation protocol can run on an open-source local model, but the resulting priors show limited agreement with the GPT-reference priors. We therefore treat it as a reproducibility and sensitivity control rather than the main prior source.

Avoid:

> The open-source local model reproduces the proprietary LLM priors and can replace the main event-prior generator.

## Next Step

For a submission version, the useful extension is not simply more samples with the same small model. A stronger open-source instruction model, stricter prompting, or calibration of local priors against the GPT-reference schema should be evaluated before claiming open-source prior equivalence.


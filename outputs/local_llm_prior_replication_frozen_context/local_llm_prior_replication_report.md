# Local/Open-Source LLM Prior Replication Report

- Environment status: `READY`
- Model id: `Qwen/Qwen2.5-1.5B-Instruct`
- Input prompt file: `outputs\event_context_corpus\frozen_context_single_prompts.jsonl`
- Selected cohorts: `8`
- Successful local priors: `0`
- Invalid/error records: `1`
- Output directory: `outputs\local_llm_prior_replication_frozen_context`
- Error log: `outputs\local_llm_prior_replication_frozen_context\local_llm_event_feature_errors.jsonl`

## Runtime Errors

| Cohort | Error |
|---|---|
| runtime_model_load | We couldn't connect to 'https://huggingface.co' to load this file, couldn't find it in the cached files and it looks like Qwen/Qwen2.5-1.5B-Instruct is not the path to a directory containing a file named config.json. Checkout your internet  |

No local-vs-reference comparison was generated. This is expected for `--audit-only` runs or when the local model environment is not ready.

## Paper Use

This artifact should be cited as an implemented replication protocol and environment audit, not as completed generation evidence.
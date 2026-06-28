# Local/Open-Source LLM Prior Replication Environment Audit

- Status: `READY`
- GPU from `nvidia-smi`: `NVIDIA GeForce RTX 4090`
- Torch installed: `True`
- Torch version: `2.12.0+cu126`
- Torch CUDA available: `True`
- Torch CUDA version: `12.6`
- Transformers installed: `True`
- Transformers version: `5.10.2`
- Accelerate installed: `True`
- BitsAndBytes installed: `False`
- Default model id: `Qwen/Qwen2.5-1.5B-Instruct`
- Recommendation: Run local LLM generation with a small cohort subset, then scale if validation passes.

## Interpretation

This audit is a reproducibility guardrail for the open-source LLM control. A paper claim that local LLM priors were generated on GPU should be made only when this status is `READY` and a non-empty `local_llm_event_features_normalized.csv` is present.
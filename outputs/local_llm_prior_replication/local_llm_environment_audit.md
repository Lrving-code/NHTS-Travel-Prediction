# Local/Open-Source LLM Prior Replication Environment Audit

- Status: `BLOCKED_TORCH_CPU`
- GPU from `nvidia-smi`: `NVIDIA GeForce RTX 4090`
- Torch installed: `True`
- Torch version: `2.8.0`
- Torch CUDA available: `False`
- Torch CUDA version: `None`
- Transformers installed: `True`
- Transformers version: `4.44.2`
- Accelerate installed: `False`
- BitsAndBytes installed: `False`
- Default model id: `Qwen/Qwen2.5-1.5B-Instruct`
- Recommendation: Install a CUDA-enabled PyTorch build in the project environment before claiming GPU local-LLM replication, for example the official PyTorch CUDA wheel matching the installed driver.

## Interpretation

This audit is a reproducibility guardrail for the open-source LLM control. A paper claim that local LLM priors were generated on GPU should be made only when this status is `READY` and a non-empty `local_llm_event_features_normalized.csv` is present.
# Reference PPT Integration Audit

Source reviewed:

- `refs/基于大模型蒸馏的高效社会行为预测方法(1).pptx`

## What the Reference Deck Contributes

The reference deck frames an efficient LLM-based social behavior prediction pipeline:

1. direct LLM prediction is expressive but too slow for large-scale simulation;
2. LLM knowledge can be distilled into explicit IF-THEN / scoring rules;
3. a lightweight rule layer handles common cases;
4. low-confidence cases may trigger an LLM fallback;
5. the main comparison is data fitting, direct LLM prediction, and a hybrid distilled-rule system.

This is valuable for presentation structure and for explaining why an LLM should not be called once per household.

## How It Is Integrated into Our Project

The final deck now treats the reference idea as a method-spectrum layer rather than a competing project:

```text
data-only fitting
-> pure LLM prior
-> zero-shot LLM rule tree
-> LLM rule + small historical calibration
-> historical model + LLM event adapter
```

The integration is reflected in:

- new main slide: `方法谱系 / Method Spectrum`;
- revised main slide: `LLM 蒸馏与部署 / Distillation & Deployment`;
- updated speaker notes explaining the role of LLM rule distillation and low-confidence fallback.

## What Is Not Directly Adopted

The reference deck's specific efficiency ratio, such as "75% rule / 25% LLM", is not copied into the main results because it is not a measured number in our NHTS experiment.

Our measured efficiency claim remains:

- 7,893 households are aggregated into 1,327 cohorts;
- batch size 15 reduces the LLM generation process to 89 prompts;
- the reported 2022 prediction stage uses a deterministic adapter with no online per-household LLM call.

Low-confidence LLM fallback is kept as a deployable extension, not as part of the main 2022 evaluation.

## Unified Oral Framing

If asked whether the two approaches conflict, use this answer:

> They are the same design family at different operating points. The rule-distillation route is strongest for sparse-data or cold-start settings. Our current NHTS 2022 task has historical survey labels but no target-year labels, so the strongest design is to anchor routine household mobility in historical NHTS data and use the LLM only for event priors. The rule-distillation baseline is included as a bridge and ablation, not as a rejected idea.

## Evidence Boundary

The main numerical results should still use the measured project metrics:

- ordinary XGBoost weighted MAE: `4.3377`;
- pure LLM pressure weighted MAE: `2.7175`;
- zero-shot LLM rule tree weighted MAE: `2.7508`;
- LLM rule + 500 history weighted MAE: `2.7723`;
- primary hybrid gated adapter weighted MAE: `2.5023`;
- primary hybrid gated adapter weighted bias: `-0.0230`.

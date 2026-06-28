# Zero-Shot Rule-Tree Baseline Specification

## Purpose

This ablation answers whether a zero-shot LLM-style decision tree can directly replace the hybrid routine-model-plus-event-prior design.

## Label Policy

- No 2022 `CNTTDHH` labels are used when defining the qualitative rule tree.
- No 2022 `CNTTDHH` labels are used when distilling the pseudo-label decision tree.
- 2022 labels are used only for final evaluation metrics.

## Qualitative Rule Tree

1. Estimate routine household demand from household size, workers, vehicles, adults, and drivers.
2. Apply an event factor from LLM event priors: trip suppression, remote-work substitution, transit avoidance, delivery substitution, and recovery sensitivity.
3. Apply small context adjustments for urban rail exposure, weekend travel day, and seasonal activity.
4. Clip the resulting household trip-count prediction to `[0, 18]`.

## Distilled Pseudo-Label Decision Tree

The following sklearn tree is trained on the zero-shot rule predictions, not on true labels:

```text
|--- HHSIZE <= 1.500
|   |--- trip_suppression_risk <= 0.545
|   |   |--- HHVEHCNT <= 1.500
|   |   |   |--- WRKCOUNT <= 0.500
|   |   |   |   |--- value: [1.843]
|   |   |   |--- WRKCOUNT >  0.500
|   |   |   |   |--- value: [2.451]
|   |   |--- HHVEHCNT >  1.500
|   |   |   |--- remote_work_substitution_likelihood <= 0.200
|   |   |   |   |--- value: [2.466]
|   |   |   |--- remote_work_substitution_likelihood >  0.200
|   |   |   |   |--- value: [3.327]
|   |--- trip_suppression_risk >  0.545
|   |   |--- HHVEHCNT <= 0.500
|   |   |   |--- HHFAMINC <= 3.500
|   |   |   |   |--- value: [0.669]
|   |   |   |--- HHFAMINC >  3.500
|   |   |   |   |--- value: [0.818]
|   |   |--- HHVEHCNT >  0.500
|   |   |   |--- WRKCOUNT <= 0.500
|   |   |   |   |--- value: [1.547]
|   |   |   |--- WRKCOUNT >  0.500
|   |   |   |   |--- value: [2.066]
|--- HHSIZE >  1.500
|   |--- HHSIZE <= 3.500
|   |   |--- trip_suppression_risk <= 0.505
|   |   |   |--- trip_suppression_risk <= 0.355
|   |   |   |   |--- value: [5.772]
|   |   |   |--- trip_suppression_risk >  0.355
|   |   |   |   |--- value: [4.813]
|   |   |--- trip_suppression_risk >  0.505
|   |   |   |--- HHVEHCNT <= 0.500
|   |   |   |   |--- value: [2.231]
|   |   |   |--- HHVEHCNT >  0.500
|   |   |   |   |--- value: [3.566]
|   |--- HHSIZE >  3.500
|   |   |--- trip_suppression_risk <= 0.500
|   |   |   |--- remote_work_substitution_likelihood <= 0.615
|   |   |   |   |--- value: [7.531]
|   |   |   |--- remote_work_substitution_likelihood >  0.615
|   |   |   |   |--- value: [6.049]
|   |   |--- trip_suppression_risk >  0.500
|   |   |   |--- value: [4.812]

```

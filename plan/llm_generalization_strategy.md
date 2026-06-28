# LLM Generalization Strategy

## Core Position

The LLM should not be used as a direct trip-count regressor. Its stronger role is **event-generalizable semantic adaptation**:

```text
traditional supervised baseline learns routine mobility
LLM generalizes event mechanisms to unlabeled household cohorts
fixed adapter turns event priors into prediction corrections
```

## Why This Uses LLM Strength

Traditional supervised models depend on patterns already represented in the training table. They are strong for routine household heterogeneity, such as household size, vehicle ownership, workers, income, region, urban/rural status, and rail availability.

The LLM adds value when the target year contains mechanisms not fully encoded by those covariates:

- remote-work substitution
- transit crowding avoidance
- online shopping and delivery substitution
- uneven post-pandemic recovery
- differential constraints for low-income, zero-vehicle, urban, rural, worker-heavy, and transit-access households

This is a generalization problem: infer plausible event-response pressure for unseen cohorts without target-year labels.

## Better LLM Usage Pattern

### 1. Batch Cohort Prompting

Current one-cohort prompting is easy to validate but slow. Batched prompting keeps the same cohort-level leakage controls while reducing requests.

Current prepared batch files:

- Batch size 8: `1327` cohorts -> `166` requests, `87.49%` request reduction.
- Batch size 15: `1327` cohorts -> `89` requests, `93.29%` request reduction.

Use batch size 15 when the endpoint has enough context and stable JSON generation. Use batch size 8 if validation failures become frequent.

### 2. Prospective Event Context

Use a frozen event-context file rather than relying on free-form model memory:

- `plan/prospective_event_context_2022.md`

This makes the deployment story cleaner: the LLM receives household cohort profiles plus allowed event mechanisms, but not 2022 NHTS labels, weights, IDs, or aggregate target outcomes.

### 3. Scenario-Generalizable Priors

The same pipeline can be reused for other shocks by swapping the event context:

- pandemic recovery
- public transit disruption
- fuel-price shock
- extreme weather or heatwave
- remote-work policy shift

The output schema can stay stable: trip suppression, remote-work substitution, transit avoidance, delivery substitution, recovery sensitivity, confidence, and explanation.

### 4. Multi-Output Behavior Adapter

The same event priors should feed multiple behavior outputs:

- trip generation: `trip_suppression_risk`
- transit share: `transit_avoidance_likelihood`
- purpose structure: `remote_work_substitution_likelihood` and `online_delivery_substitution_likelihood`
- derived mode-specific trip volumes: predicted total trips × predicted mode shares

This is stronger than using the LLM only for `CNTTDHH`.

### 5. Self-Consistency and Validation

For paper-level work, run each batch with 2-3 low-temperature samples and aggregate validated outputs by median. Keep:

- schema validation
- numeric range checks
- leakage scan
- explanation audit
- random-prior permutation controls

## Recommended Presentation Claim

Do not say:

> The LLM predicts household travel behavior.

Say:

> The LLM generalizes event mechanisms to unlabeled household cohorts, producing structured event priors that help a traditional supervised model adapt to a post-pandemic distribution shift.

## Implementation Artifacts

- `src/build_batched_llm_event_prompts.py`
- `outputs/llm_event_features/household_cohort_batch_prompt_b15_summary.md`
- `outputs/llm_event_features/household_cohort_batch_prompts_b15.jsonl`
- `plan/prospective_event_context_2022.md`

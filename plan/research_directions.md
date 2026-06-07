# NHTS Research Directions

## Current Main Direction

The current main direction is:

**Event-Aware LLM-Guided Temporal Adaptation for Post-Pandemic Household Travel Demand Prediction**

The project started from a household travel-demand transfer task, but the first CUDA XGBoost baselines show a strong and systematic positive bias when historical models are transferred to 2022. This suggests that the 2022 NHTS wave should not be treated as an ordinary future survey wave. It is better framed as a post-pandemic event-shift setting, where routine historical mobility patterns conflict with COVID-era behavioral constraints.

The research should therefore focus on whether LLM-derived event semantics can help correct historical tabular models under this rare societal shock.

Saved design documents:

- Paper narrative: `plan/paper_logic_chain.md`
- 2026 literature notes: `plan/llm_mobility_2026_literature_notes.md`
- LLM experiment plan: `plan/llm_event_adaptation_plan.md`

Current baseline evidence:

- Best historical baseline: pooled historical XGBoost with `survey_year`
- Test wave: 2022
- Weighted MAE: 4.4062
- Weighted RMSE: 5.4076
- Weighted bias: +3.7202

Immediate next step:

- Build cohort-level household profiles for LLM event feature generation, without exposing target labels or 2022 aggregate outcome statistics.

## Data Readiness

The project now has local public-use CSV data for NHTS 2001, 2009, 2017, and 2022.

Readable table coverage:

- Household: 2001, 2009, 2017, 2022
- Person: 2001, 2009, 2017, 2022
- Vehicle: 2001, 2009, 2017, 2022
- Trip: 2001, 2009, 2017, 2022
- Long-distance trip: 2001, 2022 only

Cross-year common variables:

- Household: 28 variables
- Person: 51 variables
- Vehicle: 33 variables
- Trip: 60 variables
- Long-distance trip: 48 variables across 2001 and 2022 only

## Strong Research Candidates

### 1. Temporal Domain Adaptation for Household Travel Demand

Prediction target:

- `CNTTDHH`: household travel-day trip count

Core question:

- How well can models trained on older NHTS waves predict 2022 household travel behavior under temporal distribution shift?

Why feasible:

- `CNTTDHH` and many household features are common across all four years.
- The current README baseline already aligns with this task.

Methods:

- XGBoost 2017 -> 2022 direct transfer
- Multi-source training with year features: 2001 + 2009 + 2017 -> 2022
- Sample reweighting using household covariates
- Calibration or residual correction on a small 2022 validation slice

### 2. Trip Mode and Purpose Shift Modeling

Prediction targets:

- `TRPTRANS`: trip transport mode
- `WHYTRP1S`, `WHYTO`, `WHYFROM`: trip purpose fields

Core question:

- Which travel modes and purposes are hardest to transfer across survey waves, especially into 2022?

Why feasible:

- Trip table has 60 common variables across all four years.
- The task supports classification, calibration, and error decomposition by income, urban/rural status, and region.

Methods:

- Multiclass XGBoost or LightGBM
- Weighted classification using `WTTRDFIN`
- Group-wise transfer error analysis

### 3. Person-Level Mobility Participation and Transit Use

Prediction targets:

- `CNTTDTR`: person travel-day trip count
- `PTUSED`, `USEPUBTR`, `WRKTRANS`: public transit and commute mode indicators

Core question:

- How did individual-level travel participation and transit-related behavior shift across 2001, 2009, 2017, and 2022?

Why feasible:

- Person table has 51 common variables across all four years.
- Strong equity framing is possible using age, sex, income, worker status, urban/rural status, and region.

Methods:

- Count prediction and binary/multiclass classification
- Distribution shift diagnostics
- Subgroup robustness analysis

### 4. Vehicle Ownership and Vehicle Usage Forecasting

Prediction targets:

- `HHVEHCNT`: household vehicle count
- `ANNMILES`: annual vehicle miles
- `VEHTYPE`, `VEHAGE`: vehicle characteristics

Core question:

- Can historical household and vehicle attributes forecast 2022 vehicle ownership and usage patterns?

Why feasible:

- Vehicle table has 33 common variables across all four years.
- Useful for transportation energy, emissions, and household mobility access questions.

Methods:

- Regression or ordinal classification
- Domain adaptation across survey waves
- Error analysis by income, urban/rural category, and region

### 5. Long-Distance Travel Before and After Major Behavioral Change

Prediction targets:

- `MAINMODE`, `GCDTOT`, `NTSAWAY`

Core question:

- What changed in long-distance travel structure between 2001 and 2022?

Why feasible:

- Long-distance tables exist only for 2001 and 2022, so this is not a four-wave forecasting task.
- It is better framed as a two-period comparative study.

Methods:

- Descriptive weighted comparison
- Counterfactual reweighting
- Mode and distance distribution analysis

## Recommended First Paper Direction

Start with **Temporal Domain Adaptation for Household Travel Demand**.

Rationale:

- It matches the current project README.
- The target `CNTTDHH` is available across all four years.
- It is small enough to implement quickly, but rich enough for a strong methodological story.
- It can naturally compare direct transfer, reweighting, multi-year temporal training, and residual correction.

## Immediate Next Steps

1. Build a harmonized household modeling dataset with the 28 common household variables.
2. Define weighted and unweighted metrics for `CNTTDHH`.
3. Implement the 2017 -> 2022 baseline.
4. Add multi-year training and sample reweighting.
5. Use trip/person/vehicle tasks as secondary experiments after the household baseline is stable.

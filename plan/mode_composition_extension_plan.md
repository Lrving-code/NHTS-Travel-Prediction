# Mode Composition Extension Plan

## Goal
Extend the project from household trip-count prediction to household-level travel mode composition, while keeping the zero-label 2022 trip-count adaptation as the main result.

## Scope
- Build household-level mode composition targets from trip-level NHTS records.
- Use 2017 trip data as historical mode-composition training data and 2022 trip data as target-year evaluation data.
- Avoid using 2022 mode labels for training or calibration in the main adaptation setting.
- Use existing LLM event priors as label-free mode-share correction signals when behaviorally meaningful.

## Key Distinction
- Main project: predict `CNTTDHH`, household daily trip count.
- Extension: predict household-level mode shares derived from trip-level mode fields.
- Classifying each individual `TRPTRANS` trip is a different task and should not be directly compared with our trip-count metrics.

## Phases
- [x] Phase 1: Confirm mode-field harmonization between 2017 and 2022.
- [x] Phase 2: Build household-level mode-composition dataset.
- [x] Phase 3: Train historical mode-share predictors with CUDA where possible.
- [x] Phase 4: Apply label-free LLM mode-prior corrections.
- [x] Phase 5: Generate report and decide whether this should enter the final PPT.

## Decisions Made
- Use `TRPTRANS`, not `TRIPMODE`, as the harmonization source after checking local 2017 and 2022 codebooks.
- Map year-specific `TRPTRANS` codes into common broad categories: private vehicle, walk, bike, transit, taxi/ridehail, and other.
- Keep this as an exploratory extension because the strongest contribution remains label-free `CNTTDHH` trip-count adaptation.

## Current Results
- 2017 weighted household mode shares: private `0.818`, walk `0.105`, bike `0.010`, transit `0.041`, taxi/ridehail `0.005`, other `0.021`.
- 2022 weighted household mode shares: private `0.859`, walk `0.087`, bike `0.009`, transit `0.019`, taxi/ridehail `0.005`, other `0.022`.
- Historical XGBoost weighted total variation: `0.2008`.
- Best LLM transit-avoidance correction weighted total variation: `0.1985`.
- Best LLM transit-avoidance correction reduces transit-share weighted MAE from `0.0325` to `0.0269`.

## Risks
- `TRPTRANS` coding may not be identical across years.
- Some trip fields are post-trip variables and should not be used as forecasting features.
- Household mode-share targets are compositional; shares need clipping and renormalization.

## Status
**Complete** - extension implemented and reported in `outputs/mode_composition_extension/`.

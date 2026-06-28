# PPT Storyboard

## Slide 1: Title
- Label-Free LLM Event Adaptation for Post-Pandemic Household Travel Prediction
- Key phrase: no 2022 labels for training.

## Slide 2: Problem
- Historical routine mobility assumptions break in 2022.
- COVID changed commuting, activity participation, delivery substitution, and recovery behavior.

## Slide 3: Research Question
- Can LLM event priors repair 2022 prediction without target-year labels?

## Slide 4: Method
- Traditional supervised baseline learns routine mobility.
- LLM produces cohort-level pandemic priors.
- A fixed correction rule distills the priors into prediction adjustment.

## Slide 5: LLM Priors
- trip suppression, remote work, transit avoidance, online delivery, recovery sensitivity.
- 1,327 cohorts cover 7,893 households.

## Slide 6: Main Result
- Primary gated weighted MAE reduction: 42.31%.
- Primary gated weighted RMSE reduction: 32.22%.
- Primary gated absolute weighted-bias reduction: 99.36%.

## Slide 7: Bias/R2 Tradeoff
- Gated correction has near-zero weighted bias and highest weighted R2.

## Slide 8: Household Accuracy
- Gated MAE is about 2.47 trips per household.
- 54.5% of households are within 2 trips.

## Slide 9: Diagnostics
- Global and random controls show that event-level downscaling is the main effect.
- LLM cohort ranking adds smaller but measurable subgroup signal.

## Slide 10: Takeaway
- LLMs are useful here as event-prior generators, not direct numerical predictors.

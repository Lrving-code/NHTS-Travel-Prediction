# Temporal Transfer Validation

## Scope

This check asks whether the 2022 failure is just ordinary cross-year transfer error or a stronger post-pandemic distribution shift.

## Results

| Check | Train years | Test year | Weighted MAE | Weighted bias | R2 |
|---|---|---:|---:|---:|---:|
| pre_covid_2001_to_2009 | 2001 | 2009 | 3.5207 | +0.5818 | 0.4413 |
| pre_covid_2001_2009_to_2017 | 2001+2009 | 2017 | 4.1272 | +1.2230 | 0.2505 |
| post_covid_history_to_2022 | 2001+2009+2017 | 2022 | 4.4062 | +3.7202 | -0.5055 |

## Interpretation

- Pre-COVID temporal checks have mean absolute weighted bias `0.9024`.
- The 2022 transfer check has absolute weighted bias `3.7202`.
- This supports the framing that 2022 is an event-shift target rather than a routine transfer year.

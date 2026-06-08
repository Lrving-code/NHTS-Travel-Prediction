# Label-Free Subgroup Diagnostics

- Diagnostic alpha: `1.25`.
- Positive deltas mean the LLM trip-suppression correction is better.
- Rows with fewer than 50 households are excluded from the ranked tables.

## Largest LLM Gains vs Historical Baseline

| subgroup_column | subgroup_value | rows | llm_vs_historical_mae_delta | llm_vs_global_mae_delta | llm_vs_random_mae_delta | llm_vs_historical_bias_abs_delta |
| --- | --- | --- | --- | --- | --- | --- |
| HHSIZE | 6 | 123 | 4.1035 | -0.6907 | -0.3303 | 7.5861 |
| HHFAMINC | 1 | 300 | 3.1531 | 0.4118 | 0.5332 | 4.2075 |
| HHSIZE | 5 | 343 | 2.8515 | 0.1129 | 0.2613 | 4.7834 |
| HHSIZE | 4 | 786 | 2.6780 | 0.2420 | 0.5613 | 4.2667 |
| WRKCOUNT | 3 | 266 | 2.3173 | 0.2113 | 0.4556 | 3.5977 |
| URBRUR | 2 | 1587 | 2.2143 | 0.0498 | 0.2340 | 3.8192 |
| HHFAMINC | 4 | 552 | 2.1909 | -0.0682 | 0.1372 | 3.6458 |
| HHSIZE | 3 | 967 | 2.1804 | 0.0477 | 0.1838 | 3.5575 |
| HHFAMINC | -7 | 93 | 2.1110 | -0.0718 | 0.1753 | 3.4748 |
| HHVEHCNT | 3 | 1043 | 2.0826 | 0.0068 | 0.2068 | 4.0842 |
| HHFAMINC | 6 | 1392 | 2.0775 | -0.0762 | 0.0589 | 3.8570 |
| HHFAMINC | 5 | 824 | 2.0717 | 0.0178 | 0.1739 | 3.5066 |

## Largest LLM Gains vs Global Trip-Suppression Control

| subgroup_column | subgroup_value | rows | llm_vs_historical_mae_delta | llm_vs_global_mae_delta | llm_vs_random_mae_delta | llm_vs_historical_bias_abs_delta |
| --- | --- | --- | --- | --- | --- | --- |
| HHFAMINC | 1 | 300 | 3.1531 | 0.4118 | 0.5332 | 4.2075 |
| HHSIZE | 4 | 786 | 2.6780 | 0.2420 | 0.5613 | 4.2667 |
| WRKCOUNT | 3 | 266 | 2.3173 | 0.2113 | 0.4556 | 3.5977 |
| HHVEHCNT | 4 | 360 | 2.0519 | 0.1577 | 0.4631 | 3.6195 |
| HHFAMINC | 9 | 611 | 1.4834 | 0.1469 | 0.3503 | 2.4747 |
| HHFAMINC | 2 | 242 | 1.8918 | 0.1466 | 0.3450 | 2.5107 |
| CENSUS_R | 4 | 1754 | 1.7187 | 0.1245 | 0.2791 | 2.8698 |
| HHSIZE | 5 | 343 | 2.8515 | 0.1129 | 0.2613 | 4.7834 |
| HHVEHCNT | 5 | 122 | 1.7287 | 0.1000 | 0.3523 | 3.7138 |
| HHFAMINC | 7 | 1120 | 1.2829 | 0.0933 | 0.2155 | 1.9624 |
| WRKCOUNT | 0 | 2758 | 1.8450 | 0.0917 | 0.1773 | 3.0379 |
| HHVEHCNT | 0 | 476 | 1.8371 | 0.0880 | 0.1971 | 1.9796 |

## Groups Where Global Control Beats LLM Most

| subgroup_column | subgroup_value | rows | llm_vs_historical_mae_delta | llm_vs_global_mae_delta | llm_vs_random_mae_delta | llm_vs_historical_bias_abs_delta |
| --- | --- | --- | --- | --- | --- | --- |
| HHSIZE | 6 | 123 | 4.1035 | -0.6907 | -0.3303 | 7.5861 |
| HHFAMINC | 3 | 467 | 1.7539 | -0.1511 | -0.0502 | 2.7607 |
| HHFAMINC | 6 | 1392 | 2.0775 | -0.0762 | 0.0589 | 3.8570 |
| HHFAMINC | -7 | 93 | 2.1110 | -0.0718 | 0.1753 | 3.4748 |
| HHFAMINC | 4 | 552 | 2.1909 | -0.0682 | 0.1372 | 3.6458 |
| WRKCOUNT | 1 | 2790 | 1.7831 | -0.0507 | 0.1023 | 2.9596 |
| CENSUS_R | 3 | 2915 | 1.9845 | -0.0098 | 0.0982 | 3.3370 |
| HHVEHCNT | 3 | 1043 | 2.0826 | 0.0068 | 0.2068 | 4.0842 |
| HHSIZE | 2 | 3339 | 1.7905 | 0.0134 | 0.1170 | 3.0364 |
| HHFAMINC | 5 | 824 | 2.0717 | 0.0178 | 0.1739 | 3.5066 |
| HHVEHCNT | 1 | 2623 | 1.5876 | 0.0236 | 0.1249 | 2.4179 |
| HHSIZE | 1 | 2271 | 0.9945 | 0.0257 | 0.0779 | 1.4856 |

Interpretation:

- This diagnostic separates event-level downscaling from cohort-specific LLM assignment.
- If LLM beats global controls in a subgroup, the cohort-specific `trip_suppression_risk` ranking is adding value there.
- If global beats LLM, the subgroup mainly benefits from a common downward correction rather than differentiated LLM scores.
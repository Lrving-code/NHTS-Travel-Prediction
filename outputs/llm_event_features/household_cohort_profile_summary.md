# LLM Household Cohort Profile Summary

- Source rows: 7893
- Cohorts written: 1327
- Rows covered by written cohorts: 7893
- Minimum cohort size: 1
- Cohort columns: HHFAMINC, HHVEHCNT_BIN, WRKCOUNT_BIN, URBRUR, RAIL, CENSUS_R

Leakage controls:

- Excludes `CNTTDHH`.
- Excludes `WTHHFIN`.
- Excludes `HOUSEID`.
- Excludes aggregate 2022 target statistics.
- Prompts instruct the LLM not to predict or mention household trip counts.
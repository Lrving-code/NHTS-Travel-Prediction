# Prospective Event Context for 2022 Mobility Adaptation

This context is intended for LLM event-prior generation without using 2022 NHTS outcomes.

Allowed information:

- COVID-19 changed travel behavior through remote work, reduced commuting, avoidance of crowded public transit, online shopping and delivery substitution, and uneven recovery across household groups.
- Households with more workers may have higher exposure to remote-work substitution, but effects depend on income, occupation feasibility, and local urban context.
- Households near rail or in dense urban areas may be more sensitive to transit avoidance than rural or auto-dependent households.
- Households with more vehicles may be more able to substitute away from transit and maintain essential travel.
- Lower-income, zero-vehicle, older, rural, and transit-dependent households may experience different mobility constraints and recovery patterns.
- Recovery is not uniform: some discretionary, social, shopping, and commuting trips may remain suppressed or substituted, while essential travel may recover faster.

Forbidden information:

- Do not use 2022 NHTS `CNTTDHH`, trip-count labels, sample weights, household IDs, or aggregate target-year outcome statistics.
- Do not infer exact trip counts, exact mode shares, or exact purpose shares.
- Do not tune event-prior values to match 2022 NHTS evaluation results.

Required output role:

- Produce structured event-response priors only.
- Explain mechanisms qualitatively and briefly.
- Keep all numeric priors in `[0, 1]`.

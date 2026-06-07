# LLM Event Feature Audit Summary

- Records: 20

## Numeric Feature Distribution

|                                         |   count |   mean |    std |   min |   median |   max |
|:----------------------------------------|--------:|-------:|-------:|------:|---------:|------:|
| trip_suppression_risk                   |      20 | 0.4615 | 0.1548 |  0.28 |     0.42 |  0.78 |
| remote_work_substitution_likelihood     |      20 | 0.3685 | 0.2585 |  0.05 |     0.46 |  0.72 |
| transit_avoidance_likelihood            |      20 | 0.3185 | 0.2238 |  0.12 |     0.18 |  0.86 |
| online_delivery_substitution_likelihood |      20 | 0.504  | 0.09   |  0.36 |     0.5  |  0.64 |
| post_pandemic_recovery_sensitivity      |      20 | 0.45   | 0.1153 |  0.31 |     0.42 |  0.74 |
| confidence                              |      20 | 0.6595 | 0.0349 |  0.62 |     0.66 |  0.74 |

## Primary Event Mechanisms

| primary_event_mechanism                                                                                                                                                                                                                           |   count |
|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------:|
| Non-work discretionary activity reduction with some substitution toward delivery or remote services                                                                                                                                               |       1 |
| Low-income, one-adult, zero-vehicle rural households with no workers are more exposed to pandemic-era access constraints, service disruptions, and discretionary activity suppression than to remote-work substitution.                           |       1 |
| High-income urban households with one worker and two vehicles are most likely to respond through selective substitution—remote work where feasible and online delivery for shopping—rather than strong mobility constraints or transit avoidance. |       1 |
| Dual-worker, high-income, vehicle-rich urban household likely had meaningful remote-work and online-service substitution capacity, while abundant private vehicles limited dependence on shared modes despite rail availability.                  |       1 |
| dual-worker remote-work substitution with moderate online-service substitution and low transit-specific avoidance                                                                                                                                 |       1 |
| Non-work household with high private-vehicle access; pandemic response likely centered on selective activity avoidance and some delivery substitution rather than remote work or transit avoidance.                                               |       1 |
| Single-worker urban household with rail availability and one vehicle likely faced pandemic-era mode substitution pressures, with some ability to shift work and errands online while retaining private-vehicle mobility.                          |       1 |
| Work-schedule flexibility and selective substitution of errands with online services, moderated by high vehicle access and limited rail dependence.                                                                                               |       1 |
| Non-work activity substitution and cautious discretionary travel behavior among small, non-working, one-vehicle urban households without rail availability.                                                                                       |       1 |
| Urban rail-accessible large working household with limited vehicles relative to drivers, where pandemic-era effects are most likely mediated through remote-work substitution, transit caution, and some delivery substitution.                   |       1 |
| Non-work discretionary activity suppression with some substitution to online delivery; limited relevance of remote-work effects and comparatively low transit-avoidance exposure.                                                                 |       1 |
| Workplace flexibility and discretionary-activity substitution among two-worker, two-vehicle rural households with limited transit exposure.                                                                                                       |       1 |

## Example Explanations

### hhc_000001

- Mechanism: Non-work discretionary activity reduction with some substitution toward delivery or remote services
- Confidence: 0.62
- Explanation: Single-adult, non-worker, one-vehicle urban household with no rail availability suggests low remote-work substitution and limited transit-specific avoidance. Pandemic response is more likely to operate through reduced discretionary out-of-home activity, cautious personal exposure management, and moderate use of delivery or remote services. One vehicle and one driver provide flexibility, lowering dependence on shared modes, while December timing may add seasonal caution or disruption sensitivity.

### hhc_000071

- Mechanism: Low-income, one-adult, zero-vehicle rural households with no workers are more exposed to pandemic-era access constraints, service disruptions, and discretionary activity suppression than to remote-work substitution.
- Confidence: 0.66
- Explanation: This cohort is single-adult, very low income, zero-vehicle, nonworking, rural, and without rail availability. Pandemic effects would most likely operate through reduced access, health-risk avoidance, disrupted shared/community transport, and constrained discretionary travel. Remote work substitution is minimal because there are no workers. Transit avoidance is moderate despite no rail because dependence may shift to limited public, demand-response, rideshare, or informal modes. Delivery substitution is possible but constrained by rural coverage and low income. Recovery sensitivity is moderately high because restored services and reduced health concerns could materially affect mobility options.

### hhc_000141

- Mechanism: Non-worker, zero-vehicle urban households with no rail access are likely most affected through reduced discretionary activity, dependence on constrained shared/active/local modes, and substitution toward delivery or nearby services rather than commute replacement.
- Confidence: 0.74
- Explanation: This low-income, urban, zero-vehicle, non-working cohort has very low remote-work relevance but elevated vulnerability to pandemic-era activity suppression and mode disruption. Lack of household vehicles increases sensitivity to transit, rideshare, walking, and local access constraints, while no workers reduces commute-driven recovery effects. Delivery substitution is moderately likely given urban setting but constrained by low income.

### hhc_000210

- Mechanism: Non-work urban households without vehicles likely faced elevated mobility constraints during pandemic-era disruptions, with exposure concentrated around reduced shared-mode use, reliance on walking or rides from others, and substitution of some discretionary activity through delivery or online services.
- Confidence: 0.66
- Explanation: This small, urban, no-vehicle, no-worker cohort has very low driver availability and no household vehicles, making it vulnerable to pandemic-era access constraints and transit/shared-mode avoidance. Remote-work substitution is minimal because there are no workers. Online delivery substitution is moderately likely given urban setting and low vehicle access, though income constraints may limit uptake. Recovery sensitivity is moderate because restored services and reduced health concerns would matter, but lack of vehicles remains a structural constraint.

### hhc_000280

- Mechanism: Workplace schedule flexibility and selective substitution of errands with online services, moderated by strong private-vehicle access.
- Confidence: 0.62
- Explanation: Urban household with three workers and three adults suggests meaningful exposure to pandemic-era work and activity schedule changes. Two vehicles and two drivers reduce dependence on shared modes, so transit avoidance is likely low despite urban context and no reported rail availability. Lower-middle income and multiple workers support moderate remote-work and delivery substitution, but private-vehicle access likely preserved many out-of-home options.

### hhc_000350

- Mechanism: large two-worker household with limited vehicle access, urban/rural non-rail context, and moderate income constraints
- Confidence: 0.62
- Explanation: This cohort has a large household with two workers, two drivers, and only one vehicle, suggesting pandemic-era disruptions could affect scheduling, shared-vehicle use, and discretionary activity. Remote-work substitution is plausible but not dominant given limited income information and no occupation detail. Transit avoidance is low because rail is unavailable and the area profile is not strongly rail-oriented. Online delivery substitution is moderate due to household size and pandemic shopping adaptations, while recovery sensitivity is moderate because work, school, and household coordination pressures likely returned as restrictions eased.

### hhc_000420

- Mechanism: Urban rail-access, zero-vehicle, multi-worker household likely experienced strong pandemic-era mobility disruption through transit avoidance, workplace policy shifts, and substitution toward remote activity or delivery.
- Confidence: 0.69
- Explanation: This cohort is urban, rail-served, zero-vehicle, and has two workers with no household drivers, making it highly exposed to transit disruption and avoidance during pandemic conditions. Moderate income and larger household size support some ability and need for online substitution, while recovery sensitivity remains elevated because reopening, transit confidence, and workplace attendance policies would strongly affect behavior.

### hhc_000490

- Mechanism: Workplace schedule flexibility and household-level substitution toward at-home activities, moderated by strong private-vehicle access and limited rail availability.
- Confidence: 0.68
- Explanation: This urban, two-worker, two-vehicle cohort has moderate income and multiple adults, making remote-work and delivery substitution plausible during pandemic-era disruptions. Lack of rail availability and full vehicle access reduce transit-specific avoidance as a dominant mechanism. Recovery sensitivity is moderate because commute and discretionary patterns may rebound with workplace reopening, while some online substitution can persist.

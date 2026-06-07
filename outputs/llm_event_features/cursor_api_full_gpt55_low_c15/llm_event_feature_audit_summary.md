# LLM Event Feature Audit Summary

- Records: 1327

## Numeric Feature Distribution

|                                         |   count |   mean |    std |   min |   median |   max |
|:----------------------------------------|--------:|-------:|-------:|------:|---------:|------:|
| trip_suppression_risk                   |    1327 | 0.4559 | 0.1218 |  0.24 |     0.42 |  0.78 |
| remote_work_substitution_likelihood     |    1327 | 0.3806 | 0.2309 |  0.03 |     0.46 |  0.78 |
| transit_avoidance_likelihood            |    1327 | 0.3147 | 0.2007 |  0.04 |     0.18 |  0.83 |
| online_delivery_substitution_likelihood |    1327 | 0.5058 | 0.0873 |  0.28 |     0.5  |  0.74 |
| post_pandemic_recovery_sensitivity      |    1327 | 0.4711 | 0.1069 |  0.22 |     0.46 |  0.77 |
| confidence                              |    1327 | 0.6586 | 0.0324 |  0.42 |     0.67 |  0.78 |

## Primary Event Mechanisms

| primary_event_mechanism                                                                                                                                                                                                                                                       |   count |
|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------:|
| Non-work discretionary activity reduction with some substitution toward delivery or remote services, moderated by strong private-vehicle access and low transit dependence.                                                                                                   |       2 |
| workplace schedule flexibility and discretionary activity substitution                                                                                                                                                                                                        |       2 |
| Workplace flexibility and selective substitution of errands/services rather than transit disruption                                                                                                                                                                           |       2 |
| Non-work discretionary activity reduction with some substitution toward online services; private-vehicle access limits dependence on shared modes.                                                                                                                            |       1 |
| Urban rail-available, higher-income households with one worker and two vehicles likely had moderate pandemic-era substitution through remote work, delivery services, and selective avoidance of shared transit, buffered by strong auto access.                              |       1 |
| Moderate pandemic-era substitution toward remote work and online services, buffered by high household vehicle access and limited transit exposure.                                                                                                                            |       1 |
| Moderate activity substitution and cautious discretionary travel behavior, mainly through remote-work flexibility and online-service adoption rather than transit avoidance.                                                                                                  |       1 |
| Moderate substitution of commute and shopping activity through remote work and delivery, buffered by high household vehicle access and lack of rail availability.                                                                                                             |       1 |
| Moderate pandemic-era substitution away from discretionary out-of-home activity, mainly through remote work for the single-worker household and online delivery options, with limited transit-related effects because rail is unavailable and the household has two vehicles. |       1 |
| Auto-oriented urban households with one worker and two vehicles are more likely to adapt through selective activity suppression, remote work where feasible, and delivery substitution rather than transit avoidance.                                                         |       1 |
| Urban rail-access household with one worker, high income, home ownership, and two vehicles: pandemic response likely mixed remote-work substitution and avoidance of shared modes, buffered by strong private-vehicle access.                                                 |       1 |
| Moderate substitution away from in-person activities through remote work and delivery options, with some transit caution in an urban rail-available setting but buffered by universal two-vehicle access.                                                                     |       1 |

## Example Explanations

### hhc_000007

- Mechanism: Non-work activity substitution and cautious out-of-home participation among small, non-working urban households with one vehicle and no rail access.
- Confidence: 0.62
- Explanation: This cohort is small, urban, non-working, and has one vehicle with no rail availability. Pandemic-era impacts are therefore more likely to come from reduced discretionary activity, health caution, and delivery substitution than from remote work or transit avoidance. Vehicle access supports some resilience, while older or non-working household structure may increase caution and slower recovery.

### hhc_000015

- Mechanism: Moderate work-related substitution and cautious discretionary travel adaptation in an urban, one-worker, one-vehicle adult household without rail availability.
- Confidence: 0.62
- Explanation: This is an urban two-adult household with one worker and one vehicle, suggesting some exposure to work-pattern changes and delivery substitution, but limited rail-related avoidance because rail is not available. Home ownership and vehicle access reduce vulnerability to severe disruption, while unknown income lowers certainty.

### hhc_000006

- Mechanism: Urban rail-access household with one nonworking adult and one vehicle; pandemic response likely centered on avoiding shared modes and substituting some errands with delivery rather than work-from-home changes.
- Confidence: 0.62
- Explanation: Single-adult, zero-worker household implies very low remote-work substitution. Urban setting with rail availability raises exposure to transit-avoidance behavior, though vehicle access provides an alternative. February 2022 timing suggests partial recovery but lingering caution and service disruptions. Unknown income lowers certainty and makes delivery substitution only moderate.

### hhc_000010

- Mechanism: Non-work activity substitution and cautious discretionary behavior, moderated by private vehicle access and lack of rail availability.
- Confidence: 0.62
- Explanation: Single-adult, non-worker, urban household with one vehicle and no rail availability suggests low remote-work relevance and limited transit-avoidance exposure. Pandemic-era effects would more likely operate through discretionary outing caution and some substitution toward online delivery, while private vehicle access supports resilience and recovery.

### hhc_000002

- Mechanism: workplace flexibility and urban transit caution
- Confidence: 0.62
- Explanation: Two-adult, two-worker urban household in a rail-available large metro area suggests meaningful exposure to remote-work substitution and some continued transit avoidance. Two vehicles and two drivers reduce dependence on shared modes, while June 2022 timing implies partial behavioral recovery from earlier pandemic disruption.

### hhc_000004

- Mechanism: Non-work activity caution and selective substitution, moderated by private vehicle access in an urban rail-served setting.
- Confidence: 0.62
- Explanation: Single-adult, non-worker household with one vehicle in an urban rail-available area. Remote-work substitution is low because there are no workers. Transit avoidance is moderate due to rail availability, but vehicle access provides an alternative. Online delivery substitution is moderate, especially for discretionary shopping or errands, while overall recovery sensitivity is modest because the profile lacks commute exposure.

### hhc_000005

- Mechanism: Urban rail-access household with one adult, one vehicle, no workers, and March 2022 timing; pandemic response is more likely shaped by transit avoidance, discretionary activity caution, and some delivery substitution than by remote-work effects.
- Confidence: 0.58
- Explanation: This single-adult, non-working, urban household has limited remote-work relevance but meaningful exposure to pandemic-era transit concerns because rail is available. One vehicle provides an alternative to shared modes, lowering dependence on transit while still allowing some activity continuity. Unknown income and renter status add uncertainty around delivery substitution and recovery pace.

### hhc_000001

- Mechanism: Non-work activity substitution and cautious discretionary travel behavior in an urban, single-adult, one-vehicle household without workers and without rail availability.
- Confidence: 0.62
- Explanation: This is a one-person urban household with one driver and one vehicle, no workers, and no rail availability. Pandemic-era response is therefore unlikely to be driven by remote work or transit avoidance, and more likely to reflect reduced discretionary out-of-home activity plus some substitution toward delivery or online services. December timing may modestly increase sensitivity to illness concerns and seasonal disruptions.

### hhc_000009

- Mechanism: Non-work activity suppression and partial substitution to delivery or at-home services, with limited relevance of remote work or transit avoidance.
- Confidence: 0.63
- Explanation: This small urban cohort has no workers, one vehicle per household, mostly one to two adults, and no reported rail availability. Pandemic-era response is therefore more likely driven by reduced discretionary out-of-home activity, health-risk avoidance, and some delivery substitution than by commuting changes. Limited vehicle access may raise sensitivity to service disruptions, but lack of rail access lowers transit-specific avoidance.

### hhc_000012

- Mechanism: Single-worker urban household with rail access and one vehicle likely faced competing pandemic-era substitutions: remote work reduced commute exposure, transit caution shifted some mobility toward private vehicle use, and delivery options partially replaced discretionary out-of-home activity.
- Confidence: 0.63
- Explanation: Urban location, rail availability, and one-worker status increase exposure to pandemic-related work and mode-choice changes. One vehicle and one driver provide some resilience against transit avoidance, while July 2022 timing suggests partial recovery from earlier pandemic disruptions. Income is unknown, so substitution estimates are moderate rather than high.

### hhc_000003

- Mechanism: Non-work discretionary activity adjustment with some substitution toward delivery or at-home services; low commuting-related substitution due to zero workers and low transit relevance in a rural, no-rail household.
- Confidence: 0.62
- Explanation: Two-adult rural household with no workers, many vehicles, and no rail availability suggests pandemic response is unlikely to be driven by remote work or transit avoidance. Effects are more likely through reduced discretionary outings, health-risk caution, and modest online delivery substitution, with relatively muted recovery sensitivity because travel is car-oriented and not commute-dependent.

### hhc_000013

- Mechanism: Urban rail-access household with one worker and limited vehicle availability is most exposed to pandemic-era shifts through transit avoidance, partial remote-work substitution, and substitution of some discretionary errands with online delivery.
- Confidence: 0.58
- Explanation: Two-adult urban homeowner household in a large metro area with rail availability and one vehicle for two drivers. Pandemic response priors are moderately elevated for transit avoidance and delivery substitution, with moderate remote-work potential due to one worker. December weekend context and urban access suggest some behavioral flexibility, but limited income information and cohort size imply only moderate confidence.

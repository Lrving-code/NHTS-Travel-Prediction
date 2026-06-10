# Citation Verification Log

Last checked: 2026-06-10

Scope: citations included in `outputs/paper_draft/latex/references.bib` and used by `main.tex`.

## Verification Rules

- Paper citations were added only after checking a primary source such as DOI content negotiation, ACL Anthology, OpenReview, arXiv, or the publisher page.
- Dataset and agency-report citations use official government or agency pages.
- No BibTeX entry in `references.bib` is generated solely from memory.
- The current LaTeX draft deliberately avoids unverifiable citations; missing classical travel-demand references can be added later after verification.

## Verified Paper Citations

| Key | Work | Verification source | Status | Use in paper |
|---|---|---|---|---|
| `wang2026ellmob` | ELLMob: Event-Driven Human Mobility Generation with Self-Aligned LLM Framework | OpenReview `https://openreview.net/forum?id=MPYsaBgZIT`; arXiv `https://arxiv.org/abs/2603.07946` | Verified as ICLR 2026 / arXiv source | Event-driven LLM mobility anchor |
| `yang2025causalmob` | CausalMob: Causal Human Mobility Prediction with LLMs-derived Human Intentions toward Public Events | DOI `https://doi.org/10.1145/3690624.3709231`; arXiv `https://arxiv.org/abs/2412.02155` | DOI BibTeX fetched | LLM-derived event/intention features |
| `feng2025agentmove` | AgentMove: A Large Language Model based Agentic Framework for Zero-shot Next Location Prediction | ACL Anthology `https://aclanthology.org/2025.naacl-long.61/`; DOI `https://doi.org/10.18653/v1/2025.naacl-long.61` | DOI BibTeX fetched | Zero-shot LLM mobility baseline framing |
| `chen2026agentmob` | Towards Efficient and Evidence-grounded Mobility Prediction with LLM-Driven Agent | arXiv `https://arxiv.org/abs/2606.05130` | arXiv BibTeX fetched | Fast-path plus selective reasoning framing |
| `wang2025elpmob` | Building Efficient LLM Pipeline for Human Mobility Prediction | DOI `https://doi.org/10.1145/3748636.3771314`; GitHub `https://github.com/chwang0721/ELP-Mob` | DOI BibTeX fetched | LLM mobility efficiency / batching |
| `long2025unimob` | A Universal Model for Human Mobility Prediction | DOI `https://doi.org/10.1145/3690624.3709236`; arXiv `https://arxiv.org/abs/2412.15294` | DOI BibTeX fetched | Universal mobility model direction |
| `liang2025stfmsurvey` | Foundation Models for Spatio-Temporal Data Science | arXiv `https://arxiv.org/abs/2503.13502` | arXiv BibTeX fetched | Spatio-temporal foundation model survey |
| `yan2025transportllm` | Large Language Models for Traffic and Transportation Research | arXiv `https://arxiv.org/abs/2503.21330` | arXiv BibTeX fetched | Transportation LLM survey |
| `sharifi2026dailytravel` | Integrating hybrid recurrent neural networks and large language models for daily travel behavior prediction | DOI `https://doi.org/10.1016/j.trip.2025.101793` | DOI BibTeX fetched | NHTS / daily travel behavior related work boundary |

## Verified Data and Agency Sources

| Key | Source | Verification source | Status | Use in paper |
|---|---|---|---|---|
| `fhwa2022nhts` | 2022 National Household Travel Survey public-use data | Official NHTS downloads page `https://nhts.ornl.gov/downloads` | Official data source checked | Main dataset citation |
| `fhwa2024nhtstrends` | Summary of Travel Trends: 2022 NHTS | ROSA/National Transportation Library `https://rosap.ntl.bts.gov/view/dot/73764` | Official report page checked | Post-pandemic travel-trends context |
| `burrows2024commuting` | Commuting in the United States: 2022 | U.S. Census Bureau PDF `https://www2.census.gov/library/publications/2024/demo/acsbr-018.pdf` | Official ACS brief checked | Work-from-home and transit mechanism evidence |
| `psrc2024regionaltravelstudy` | 2023 Puget Sound Regional Travel Study Final Report | PSRC PDF `https://www.psrc.org/sites/default/files/2024-05/2023-Puget-Sound-Regional-Travel-Study-Final-Report.pdf` | Official report checked | External household-survey validation source |

## Known Citation Gaps

These are not blockers for the current skeleton, but they should be added before a real submission:

- Classical household travel-demand modeling and trip-generation references.
- Survey weighting and travel diary methodology references.
- Domain adaptation / covariate shift references for structured tabular prediction.
- Causal inference references for negative controls and leakage/audit framing.

Add them only after programmatic or official-source verification.

## Reproducibility Notes

The DOI BibTeX entries were fetched through HTTP content negotiation with:

```text
Accept: application/x-bibtex
```

The arXiv entries were fetched from:

```text
https://arxiv.org/bibtex/<arxiv_id>
```


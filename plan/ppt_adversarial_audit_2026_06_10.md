# PPT Adversarial Audit: 2026-06-10

## Target

The deck must support a 10-minute final defense and remain extensible toward a paper presentation. It should not feel like a pile of experiments. It should tell one story:

> 2022 is an event-shift target; routine household models fail; LLM event priors provide label-free mechanism adaptation; guardrails and Pareto evaluation make the result auditable for planning.

## Current Structure

The current deck has 23 slides:

1. Title
2. Content
3. Research gap
4. Related work
5. Data and target system
6. Technical route
7. LLM event prior
8. Evaluation design
9. Main trip-count results
10. Statistical validation
11. Error-bias tradeoff
12. Multi-objective selection
13. Error distribution and tolerance
14. Event heterogeneity
15. Robustness check
16. Subgroup check
17. Error insight
18. LLM role
19. Behavior outputs
20. Purpose composition
21. LLM scaling
22. Research logic
23. Conclusion

## Core Talk Path for 10 Minutes

Use these as the main spoken slides:

1. Title
2. Research gap
3. Data and target system
4. Technical route
5. LLM event prior
6. Evaluation design
7. Main trip-count results
8. Statistical validation
9. Multi-objective selection
10. LLM role
11. Behavior outputs
12. Conclusion

Slides 11, 13-17, 20-22 should be backup or fast-support slides unless the instructor asks for details.

## High-Severity PPT Issues

### Issue 1: Too many evaluation slides for 10 minutes

Risk: The audience will remember many metrics but not the core thesis.

Fix:

- Keep slide 12 as the integrator: it explains why multiple metrics exist.
- Move detailed robustness/subgroup/error-distribution slides into backup in speaker notes or appendix.
- In the live talk, summarize them as “we checked this with bootstrap, random controls, and subgroup diagnostics.”

### Issue 2: LLM role must be explained before results

Risk: Audience may think the LLM directly predicts trip counts.

Fix:

- Slide 6 and 7 must say: `LLM outputs s_event, not y`.
- Slide 18 should be pulled earlier if confusion remains.
- Every results slide should use “event prior adapter,” not “LLM prediction.”

### Issue 3: Purpose slide may weaken the story

Risk: If purpose composition is weak, it dilutes the otherwise strong trip/mode result.

Fix:

- In a 10-minute talk, mention purpose only as “extended target; current boundary result.”
- Do not put purpose in the final headline unless improved by future experiments.

### Issue 4: Multi-objective slide needs one clear message

Current message:

- Different planning preferences select different operating points.

Fix:

- Speak it as: “We choose the primary gated method because the course/report objective is balanced accuracy and calibration, not pure minimum MAE.”
- Low-cost global prior should be framed as a deployment option, not a threat.

### Issue 5: Related work should be sharpened around 2025-2026 signal

Risk: A generic related-work slide looks less research-grade.

Fix:

- Group papers by the exact design lessons:
  - event-driven mobility,
  - large-small model collaboration,
  - LLM-assisted optimization,
  - causal spatio-temporal guardrails.
- Avoid a long list of paper names.

## Medium-Severity Layout Issues

### Issue 6: Table density versus 12pt constraint

Status:

- New multi-objective table is 12pt.
- Some existing labels, footers, and chart annotations are smaller than 12pt because they are template/footer/chart labels.

Fix:

- Body tables should remain 12pt.
- Footer/page number can stay small.
- Figure axis labels may stay small if readable, but core conclusion text should be 12pt or larger.

### Issue 7: Section numbering order

Status:

- New evaluation sequence uses 08A, 08B, 08C, 08D.
- Robustness sequence uses 09, 09A, 09B.

Fix:

- Acceptable for now, but a publication-style deck should use cleaner numeric sequence and put backups at the end.

### Issue 8: Too many bilingual lines can consume space

Fix:

- Keep slide titles bilingual.
- Use bilingual story boxes only when the English helps the report.
- For dense result slides, prioritize Chinese explanation plus English metric labels.

## Recommended Next PPT Edits

1. Add a “main path vs backup” note to speaker notes.
2. Consider moving slide 18 before slide 9 if test viewers still misunderstand LLM role.
3. Convert robustness/subgroup/detail slides into appendix after the conclusion for the final 10-minute version.
4. Make related-work slide more explicitly 2025-2026 and design-lesson based.
5. Add one causal DAG figure if causal guardrails are discussed.

## Current Verdict

For course defense: usable and above ordinary class-project level.

For paper-style presentation: promising but still too broad. The next improvement should be empirical strengthening and storyline pruning, not adding more decorative slides.

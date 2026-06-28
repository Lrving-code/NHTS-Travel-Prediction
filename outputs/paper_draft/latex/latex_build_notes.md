# LaTeX Build Notes

Last checked: 2026-06-29

## Files

- `main.tex`: article-style manuscript skeleton.
- `references.bib`: verified BibTeX entries.
- `citation_verification_log.md`: verification record for each citation.

## Static Checks

The citation-key static check verifies that every `\citep{...}` key in `main.tex` exists in `references.bib`.

Current status:

- Missing citation keys: none.
- Unused BibTeX keys: none after adding the daily-travel LLM reference to the related-work paragraph.
- Core figure labels are present in `main.tex`: `fig:workflow`, `fig:metric_comparison`, `fig:permutation`, and `fig:multi_objective`.

## Local Compile Attempt

Available TeX tools on this machine:

- MiKTeX `pdflatex`
- MiKTeX `bibtex`
- MiKTeX `latexmk`

Commands attempted:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Observed result:

- `latexmk` still fails because MiKTeX cannot find the required Perl script engine.
- The manual `pdflatex -> bibtex -> pdflatex -> pdflatex -> pdflatex` sequence succeeds in this Codex shell.
- `main.pdf` is generated locally as a 10-page PDF.
- `main.log` reports no undefined citation warnings after the final run.
- Remaining TeX warnings are non-blocking: one underfull hbox and one overfull hbox around the compact results table.
- MiKTeX still prints elevated-privilege and update-check warnings; these do not stop the manual compile.

Decision:

- Do not commit generated PDF or intermediate files from this environment.
- Treat the manual compile sequence as the current reproducible source gate.
- Before external submission, rerun the same sequence in a normal, non-elevated PowerShell session or Overleaf and inspect the overfull table warning.

Suggested compile command in a normal TeX environment:

```powershell
cd outputs\paper_draft\latex
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

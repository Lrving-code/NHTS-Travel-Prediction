# LaTeX Build Notes

Last checked: 2026-06-10

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
- MiKTeX `xelatex`
- MiKTeX `bibtex`
- MiKTeX `latexmk`

Commands attempted:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
```

Observed issue:

- `latexmk`, `bibtex`, and the TeX engines return MiKTeX elevated-permission security-risk errors in this Codex shell.
- `xelatex` can write a draft PDF, but BibTeX cannot run in this shell, so citations remain unresolved in that PDF.

Decision:

- Do not commit generated PDF or intermediate files from this environment.
- Keep the verified source files and static citation-key check as the current reproducible gate.
- Re-run the full compile in a normal, non-elevated PowerShell session or Overleaf before external submission.

Suggested compile command in a normal TeX environment:

```powershell
cd outputs\paper_draft\latex
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

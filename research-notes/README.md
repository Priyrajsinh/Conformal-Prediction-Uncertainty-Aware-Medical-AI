# Research notes

Reading log for the methods and design choices behind this project. Each
note is roughly one page, written for my own future reference — what the
paper / library actually says, what surprised me, and how it informed
the code that ended up in `src/`.

Order of reading matters: 01 → 02 is foundations → tooling, 03 is the
clinical-utility lens, 04 is the project-specific design decisions, 05
is the regulatory framing.

| # | Title | Source | Date read | One-line takeaway |
|---|-------|--------|-----------|--------------------|
| 01 | [Conformal prediction foundations](01-conformal-prediction-foundations.md) | Vovk, Gammerman & Shafer (2005); Angelopoulos & Bates (2022) | 2026-05-08 | Exchangeability + quantile of nonconformity scores ⇒ finite-sample marginal coverage, no distributional assumption |
| 02 | [MAPIE library design](02-mapie-library-design.md) | Taquet et al. (2022); MAPIE 1.x source | 2026-05-09 | `prefit=True` lets you keep your XGBoost untouched and bolt conformal on top; LAC is the correct binary score |
| 03 | [Decision curve analysis](03-decision-curve-analysis.md) | Vickers & Elkin (2006); Vickers et al. (2019) | 2026-05-09 | Net benefit translates accuracy into clinical-utility-at-a-threshold — AUC alone hides bad operating points |
| 04 | [Design rationale for this project](04-design-rationale.md) | — | 2026-05-10 | Why split-LAC, three-way split, XGBoost, MAPIE: each choice traded simplicity for a specific risk reduction |
| 05 | [EU AI Act for a binary medical classifier](05-eu-ai-act-medical-ai.md) | EU AI Act consolidated text (artificialintelligenceact.eu) | 2026-05-11 | Articles 9/10/13/14 + Annex III §1 map cleanly onto coverage monitor, pandera, NL translator, Mondrian audit |

## How to use these notes

- They are *summaries with citations*, not exhaustive paraphrases. If
  you want to argue about a claim, follow the citation, not the note.
- Code references use the form `src/path/to/file.py::function_name` and
  always point to the version on `main` at the date the note was written
  — see `git blame` for evolution since.
- The notes are deliberately opinionated. They explain why a *rejected*
  option was rejected, not just what we built. That is where the value
  is when revisiting the project six months from now.

## Future entries

Things on my list that did not make this round:

- Adaptive Prediction Sets (Romano-Sesia-Candès 2020) — added briefly in
  04 but deserves its own note when I extend to multiclass.
- Jackknife+ (Barber-Candès-Ramdas-Tibshirani 2021) and why I did not
  use it here despite the theoretical appeal.
- Online conformal prediction (Gibbs-Candès 2021) — relevant for the
  drift-monitor extension.

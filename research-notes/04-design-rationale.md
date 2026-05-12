# 04 — Design rationale for this project

**Written**: 2026-05-10

These are the choices that have a credible alternative — i.e. somebody
reading the code could reasonably ask "why not X?" — together with the
reasoning that put each one in `src/` instead of its competitor.

## Why split-LAC, not RAPS or APS

RAPS (Romano-Sesia-Candès 2020) and APS (Romano-Patterson-Candès 2019)
shine when the number of classes is large enough that *ranking
candidate labels* is informative — `k_reg` and `lambda_reg` regularise
the tail of the sorted probability vector. With two classes the
ranking has length two and the regularisation has nothing to bite on.
Rule C35 records this: LAC is the only sensible binary score in MAPIE
1.x. If/when this project extends to the multiclass version of the
UCI Heart Disease dataset (`target` ∈ {0, 1, 2, 3, 4}, no
binarisation), RAPS will go back in.

## Why a three-way split, not jackknife+ or pure cross-conformal

Jackknife+ (Barber-Candès-Ramdas-Tibshirani 2021) gives you tighter
sets on small datasets by refitting the model with each row held out
in turn — leveraging more data for both training and calibration.
Theoretically attractive on 303 rows.

Why I did not use it:

1. **Audit-trail friction.** Article 10 ("data governance") wants
   each split named, checksummed and on disk. Jackknife+ has no
   "calibration set" — it has 303 leave-one-out fits. Explaining
   that to an auditor is a thing I would rather not do.
2. **Compute cost vs. benefit on this dataset.** XGBoost with 200
   trees on 240 rows is fast, but 303 leave-one-out fits is still
   ~50× the calibration cost of split. The empirical method
   comparison in `reports/results.json::method_comparison` shows
   CV+10 (a tighter approximation to jackknife+) gives mean set size
   1.20 at α = 0.10 vs split's 1.35 — a real but modest improvement
   that does not outweigh the auditability cost.
3. **Pedagogy.** This is a portfolio project. Split conformal is the
   one a hiring manager can read end-to-end in fifteen minutes.

## Why XGBoost, not logistic regression

Logistic regression is the textbook starting point on tabular
medical data and would have shipped a stronger ECE almost for free
(LR is linear and inherently better-calibrated than tree ensembles
out of the box). I picked XGBoost anyway because:

1. The non-linear interactions in cardiac risk (e.g., `oldpeak ×
   exang`, `thalach × age`) genuinely move the needle on a 303-row
   dataset — baseline accuracy 0.867 with XGBoost vs ~0.82 for a
   well-tuned LR on this split.
2. SHAP explanations are *the* artefact stakeholders want on
   Tab 1 / Tab 2 of the Streamlit dashboard. SHAP works for LR too,
   but the per-feature attributions on a non-linear model are
   visibly more informative — they actually show interactions.
3. The whole point of layering conformal prediction on top is that
   the *underlying calibration of the classifier does not matter*
   for the coverage guarantee. So I am free to pick the
   discrimination-best model and let the conformal layer carry the
   uncertainty contract. That is the trade I wanted to demonstrate.

## Why MAPIE, not `nonconformist` or a hand-rolled implementation

Already covered in `02-mapie-library-design.md`. One-liner: MAPIE is
maintained, has the formal guarantee in tested code, and exposes both
split and CV+ behind a single sklearn-compatible API. The
maintenance status of `nonconformist` made it a non-starter.

## Why `cv='prefit'` (now `prefit=True` in 1.x)

This one is non-negotiable. Letting MAPIE refit the base model on cal
data would silently move rows between the train and cal pools and
destroy the named-split audit trail. Rule C33 ("three-way split is
sacred") enforces this in code with explicit `sample_id` overlap
assertions.

## Why 60/20/20, not 70/15/15 or 80/10/10

The dominant constraint is calibration set size. With 303 rows total
and a 20% test set, the cal split has 60 rows. The Vovk finite-sample
correction is `(n+1)/n = 61/60 ≈ 1.017`, which means the quantile
shift is small. Going to 70/15/15 would have given 45 calibration
rows and a noisier guarantee. Going to 80/10/10 would have given 30,
which is at the edge of where conformal sets become noticeably wide
and the variance across calibration draws becomes a real concern.

60/20/20 is the smallest test-set size at which I am willing to
defend the headline coverage number with a straight face, given a
303-row dataset.

## Why patient-friendly NL output instead of raw sets

Returning `{0, 1}` to a clinician is fine. Returning `{0, 1}` to a
*patient* — which the Gradio Space's audience includes — is a UX
failure. Rule C45 translates set + label into "Likely heart disease
(high confidence)" / "Likely no heart disease (high confidence)" /
"Uncertain — needs human review". The translator collapses the
mathematical object into the three actions a non-expert can take.
Article 13 ("transparency to users") is the regulatory hook; UX
sanity is the actual driver.

## Things I would change with more data

- Mondrian-calibrate per (sex, age_bin) bucket once the dataset is
  large enough that each bucket has ≥ 100 cal rows. Currently 8 rows
  in the `age ≥ 65` bucket is too few.
- Run jackknife+ as a comparison method, accept the auditability
  cost, and publish the trade-off.
- Add temporal features (medication, prior procedures) — the static-
  feature limitation is the largest gap between this and a deployable
  system.

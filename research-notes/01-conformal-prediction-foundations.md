# 01 — Conformal prediction foundations

**Read**: 2026-05-08
**Sources**:
- Vovk, Gammerman & Shafer (2005), *Algorithmic Learning in a Random World*, Springer. ISBN 978-0-387-00152-4. Chapters 2–4.
- Angelopoulos & Bates (2022), *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*. arXiv:2107.07511. Sections 1–3.

## The single most important sentence

> If the calibration data and the test point are *exchangeable*, sorting
> the nonconformity scores of the calibration set gives you a quantile
> that is a valid finite-sample bound on the test score, with no
> distributional assumption.

That is the whole trick. Everything else — RAPS, APS, CV+, jackknife+ —
is bookkeeping around that one fact.

## Exchangeability in plain English

A sequence of random variables is exchangeable if any permutation has
the same joint distribution. IID is a sufficient condition, not a
necessary one. For this project the cal and test rows come from the
same underlying UCI Cleveland sample, so exchangeability holds by
construction — the moment the same patient appears in both, or the
calibration data is from a different hospital, the guarantee dies. That
is the *only* assumption.

## Split conformal in three steps

1. Train a model on a `train` split (no conformal involvement).
2. Score the `cal` split with a *nonconformity score* — anything that
   measures "how poorly does this prediction fit the true label". For
   binary classification with LAC, the score is `s(x, y) = 1 − p̂(y|x)`.
3. At test time, predict `s(x_test, y)` for each candidate label and
   include `y` in the set iff `s(x_test, y) ≤ q̂`, where `q̂` is the
   `⌈(n+1)(1−α)⌉ / n` quantile of the calibration scores.

The `(n+1)/n` correction is the finite-sample fix — it is the reason
the guarantee holds at small `n`, not just asymptotically.

## Marginal vs conditional coverage

What you get for free: `P(Y ∈ C(X)) ≥ 1 − α`, averaged over the joint
draw of `(X, Y)`. That is *marginal* coverage.

What you usually want but cannot have: `P(Y ∈ C(X) | X = x) ≥ 1 − α`
for every `x`. That is *conditional* coverage. Foygel-Barber et al.
(2021) prove that distribution-free conditional coverage is impossible
in finite samples without giving up on having a useful (small) set.

The pragmatic middle ground is *group-conditional* (Mondrian) coverage:
condition on a discrete subgroup like `sex` or `age_bin`, calibrate
inside each bucket, and you get a per-bucket marginal guarantee. That
is what the Mondrian audit in `src/evaluation/evaluate.py` reports.

## What surprised me

- The proof does not need anything about the *quality* of the
  underlying model. You can have a model that returns random labels and
  the conformal wrapper will still give you valid coverage. The
  marginal guarantee is a property of the *procedure*, not the
  classifier. The classifier only affects set *size*: bad model → big
  sets.
- The empty set is allowed. `C(x) = ∅` says "no label is plausible at
  this confidence level". For binary classification this is almost
  never what you want, and `safe_predict()` checks `n_empty_sets > 0`
  as a calibration sanity assertion (it has been zero on every run so
  far).

## How this maps onto the code

- `src/data/split.py::three_way_split` — produces the cal split.
- `src/models/conformal.py::ConformalXGBoost.fit` — trains the
  underlying XGBoost on the train split only.
- `src/models/conformal.py::ConformalXGBoost.calibrate` — runs
  MAPIE's `SplitConformalClassifier(prefit=True).conformalize(X_cal,
  y_cal)`. This is step 2 above. **The test set is never touched here.**
- `src/evaluation/coverage.py::empirical_coverage` — verifies the
  guarantee on the test split for each α.

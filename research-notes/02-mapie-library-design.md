# 02 — MAPIE library design

**Read**: 2026-05-09
**Sources**:
- Taquet, Blot, Morzadec, Lacombe & Brunel (2022). *MAPIE: an open-source library for distribution-free uncertainty quantification.* arXiv:2207.12274.
- MAPIE 1.x source, modules `mapie.classification` and `mapie.conformity_scores`.

## Why MAPIE rather than rolling my own or using `nonconformist`

- `nonconformist` (Linusson, 2018) — the original Python library, but
  unmaintained since 2021, no MAPIE-style sklearn-1.x compatibility,
  no CV+ implementations.
- Rolling my own — the *split* case is 50 lines and tempting. Tempting
  is exactly why I did not: a subtle off-by-one in the `(n+1)/n`
  quantile would silently break the coverage guarantee and the
  regression test would pass for months before someone audited it.
  Using a library with citations + maintainers means the formal
  guarantee is somebody else's job.
- MAPIE — actively maintained (last release in the few months before
  this project), sklearn-pipeline-compatible, has both split and CV+
  in the same API.

## API shape in MAPIE 1.x

The pre-1.x form `MapieClassifier(method='raps', cv='prefit')` is no
longer the public surface (and CLAUDE.md rule C35 records the migration
note). The 1.x form for this project is:

```python
from mapie.classification import SplitConformalClassifier

scc = SplitConformalClassifier(
    estimator=xgb,                          # the *prefit* base model
    confidence_level=[1 - a for a in alphas],
    conformity_score="lac",
    prefit=True,
)
scc.conformalize(X_cal, y_cal)               # calibration step
predictions, prediction_sets = scc.predict_set(X_test)
```

Three things to notice:

1. `confidence_level` takes `1 − α`, not `α`. Easy off-by-one when
   reading the API for the first time.
2. `conformity_score="lac"` — Least Ambiguous Classifier. The score is
   `s(x, y) = 1 − p̂(y|x)`. For binary classification this is the only
   sensible choice in MAPIE 1.x. **RAPS is mathematically degenerate
   on two classes** (rule C35) because the "ranking" only has one
   non-trivial position, so `k_reg` and `lambda_reg` collapse. The
   library will let you set it, but the resulting sets are equivalent
   to LAC and the regularisation parameters are wasted.
3. `prefit=True` is the critical flag. It tells MAPIE *not* to refit
   the base estimator on the calibration data — the XGBoost stays
   exactly as fit on the train split, and only the conformal layer
   sees the cal split. This matches the three-way-split contract
   (rule C33) and is the cheapest way to get a true split-conformal
   guarantee.

## Why `prefit=True` is the right default for medical AI

Without it, MAPIE re-splits or refits and you lose explicit control
over which rows the model has and has not seen. For a regulatory
audit (Article 10) you want the cal split to be a *named, checksummed
file* on disk, not an opaque internal slice. `prefit=True` makes the
contract explicit: "this XGBoost was trained on train.csv, this
calibrator was fit on cal.csv, this evaluation uses test.csv". DVC
checksums the three files. Job done.

## CV+ — what I read but did not deploy

`CrossConformalClassifier` (CV+) replaces the single cal split with
K-fold cross-validation, training K base models and aggregating their
nonconformity scores. The Romano-Sesia-Candès argument is that CV+
gives you better *conditional* coverage on small datasets because you
average over K calibration draws. The MAPIE paper benchmarks back
this up on tabular data.

On this dataset CV+ at K=10 gave **coverage 0.900 vs 0.967 for split**
(both at α=0.10, see `reports/results.json::method_comparison`). Same
target, larger mean set size (1.20 vs 1.35). For a portfolio project
where simplicity and audit-trail clarity matter more than squeezing
the last few % of efficiency, split won. For a production deployment
on a different dataset I would re-benchmark; the choice is not
universal.

## How this maps onto the code

- `src/models/conformal.py::ConformalXGBoost.__init__` —
  instantiates `SplitConformalClassifier` with the alpha list and
  `prefit=True`.
- `src/models/conformal.py::ConformalXGBoost.calibrate` —
  the `conformalize(X_cal, y_cal)` call. Wrapped in `safe_predict`
  for NaN / inf / shape checks.
- `src/evaluation/method_compare.py` — runs split vs CV+ at K=5 and
  K=10 and writes the comparison table that the README and MODEL_CARD
  quote.

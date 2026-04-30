# MODEL_CARD.md — P2 Conformal Prediction · Uncertainty-Aware Medical AI

> Template — finalised on Day 7 with real numbers from `reports/results.json`.

## Model details
- Architecture: XGBoost (sklearn API) wrapped by MAPIE `MapieClassifier(method='raps', cv='prefit')`
- Hyperparameters: see `config/config.yaml` (defaults: n_estimators=200, max_depth=4, learning_rate=0.05, seed=42)
- Calibration alphas: {0.05, 0.10, 0.20}
- Training data: UCI Heart Disease Cleveland, 303 rows × 13 features. Target binarised from 0–4 → {0, 1}.
- Three-way split: train 60% / cal 20% / test 20% (stratified by target, seed=42)

## Intended use
- Risk stratification assistant for cardiologists. NOT a diagnostic tool.
- Output is a prediction *set* with coverage guarantee — singleton sets indicate confidence; multi-class sets indicate the model is uncertain.

## Out-of-scope use
- Standalone diagnosis without clinician review.
- Patient populations outside the UCI Cleveland age range (29–77) or different demographic distributions.
- Real-time deployment without ongoing coverage monitoring (see "Training-serving skew" below).

## Coverage guarantees (filled on Day 4)
| α | Empirical coverage on test | Mean set size | Median set size |
|---|---------------------------|---------------|-----------------|
| 0.05 | TBD ≥ 0.95 | TBD | TBD |
| 0.10 | TBD ≥ 0.90 | TBD | TBD |
| 0.20 | TBD ≥ 0.80 | TBD | TBD |

ECE (Expected Calibration Error, rule U2): TBD on Day 4.

## Group-conditional coverage (rule #3)
| Subgroup | n | Empirical coverage at α=0.10 | Set size mean |
|----------|---|------------------------------|----------------|
| sex=female | TBD | TBD | TBD |
| sex=male | TBD | TBD | TBD |
| age < 50 | TBD | TBD | TBD |
| age 50–64 | TBD | TBD | TBD |
| age ≥ 65 | TBD | TBD | TBD |

## Training-serving skew
- Training feature distributions saved to `models/training_stats.json` after training.
- `safe_predict()` checks each inference input against the training stats and logs warnings if the patient is far outside the training distribution.
- DVC tracks `models/training_stats.json` so any change requires an explicit commit.

## Decision Curve Analysis (rule A)
*Net-benefit plot inserted on Day 4. Compares model vs treat-all and treat-none baselines across decision thresholds.*

## Selective classification (rule C)
*Accuracy as a function of abstain rate (singleton vs multi-class sets) on Day 4.*

## Limitations
- 303-row dataset is small. Coverage tightness has high variance across calibration set draws.
- Cleveland sample is non-representative of the broader population (especially under-represented women).
- Conformal coverage is *marginal* by default — use the Mondrian / group-conditional analysis above before deploying for any specific subgroup.

## EU AI Act framing (Day 7)
- Article 9 (Risk Management): coverage drift monitor + coverage_violations_total counter satisfies ongoing risk monitoring.
- Article 10 (Data and data governance): pandera schema + DVC + checksums.
- Annex III §1 (Medical devices): high-risk AI system. Documentation here serves as the technical-documentation requirement.

## Citations (rule M3)
- Romano, Sesia & Candès (2020) — Classification with Valid and Adaptive Coverage (RAPS). arXiv:2006.02544
- Vovk, Gammerman & Shafer (2005) — Algorithmic Learning in a Random World. Springer.
- Taquet, Blot, Morzadec, Lacombe & Brunel (2022) — MAPIE: an open-source library for distribution-free uncertainty quantification. arXiv:2207.12274
- Vickers & Elkin (2006) — Decision curve analysis: a novel method for evaluating prediction models. Med Decis Making 26(6):565-574.

## Maintenance
- Owner: Priyrajsinh Parmar | priyrajsinh03@gmail.com
- Recalibration trigger: empirical coverage on rolling 1000-prediction window drops below `1 - alpha - 0.02`.
- Refit cadence: when DVC sees a new `data/raw/heart.csv` checksum.

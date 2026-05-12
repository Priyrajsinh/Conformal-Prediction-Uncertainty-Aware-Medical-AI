# 05 — EU AI Act for a binary medical classifier

**Read**: 2026-05-11
**Sources**:
- EU AI Act consolidated text via [artificialintelligenceact.eu](https://artificialintelligenceact.eu/). Articles 9, 10, 11, 13, 14 and Annex III.
- ENISA / JRC explanatory notes on Annex III §1.

This is not legal advice. It is engineering framing — the mapping
between code I actually wrote and the requirements the Act actually
imposes on a system like this one.

## Why this system is high-risk

Annex III §1 lists "AI systems intended to be used as safety
components in the management and operation of medical devices, or
that are themselves medical devices under Regulation (EU) 2017/745".
A binary classifier whose output a cardiologist consults before
ordering further investigation falls squarely inside that. The
"safety component" framing matters even though I am not selling a
medical device: an *engineered* portfolio version is being built so
that the path to a regulated one is well-trodden.

High-risk classification triggers Articles 9–15 in full. The five
that bind hardest on the engineering surface are 9, 10, 11, 13, 14.
The rest (8: compliance, 12: record-keeping, 15: accuracy and
robustness) overlap heavily with the five below and are mostly
satisfied as a side effect.

## Article 9 — Risk management throughout lifecycle

> Providers shall establish, implement, document and maintain a risk
> management system in relation to high-risk AI systems.

What this looks like for a conformal classifier: *coverage is the
risk metric*. The day the empirical coverage on a rolling window
drops below `1 − α − 0.02`, the system is operating outside its
contracted reliability and either needs recalibration or a wider α.

Implementation in this project:

- `src/monitoring/coverage_drift.py::CoverageDriftMonitor` —
  maintains a 1000-prediction rolling window, exposes
  `mean_set_size_gauge`, `coverage_violations_total` (counter) and a
  `coverage_drift_alarm` gauge (0/1) on `/metrics`. (Rule C42.)
- `config/config.yaml::monitoring.alarm_threshold` — the
  configurable trigger.
- The alarm is the regulatory artefact: it is what a post-market
  monitoring obligation under Article 9 will hook into. Today it
  triggers Prometheus Alertmanager; tomorrow it triggers a human
  audit workflow.

## Article 10 — Data and data governance

> Training, validation and testing data sets shall be relevant,
> representative, free of errors and complete.

Three lines of defence:

1. **Schema.** `src/data/schema.py::HEART_DISEASE_SCHEMA` declares
   the type and range of every one of the 13 features (and the
   binary target). pandera runs this *before* the three-way split
   (rule C34) — drift is caught at the source, not three pipeline
   steps later.
2. **Lineage.** DVC tracks `data/raw/heart.csv` with a SHA-256
   checksum at `data/raw/heart.csv.sha256` (rule C41). Any change to
   the raw data forces a new checksum and a new model retrain
   (rule C42's refit cadence).
3. **Stratification.** The three-way split is stratified by target
   with seed=42, and zero-overlap is asserted on `sample_id` sets.
   No row appears in more than one split.

What is *not* implemented (and is fairly called out in the
limitations section of MODEL_CARD.md): representativeness of the UCI
Cleveland cohort. 303 rows, mostly male, mostly from one hospital, is
not a representative sample of "patients presenting with chest
pain" in any global sense. Article 10 would require a deployment-
population sample before a CE-marked release.

## Article 11 — Technical documentation

> The technical documentation of a high-risk AI system shall be
> drawn up before that system is placed on the market.

That document is `MODEL_CARD.md`. It carries the architecture,
hyperparameters, three-way-split provenance, headline coverage,
Mondrian audit, DCA, selective classification, training-serving skew
note, recalibration trigger, and references. Annex IV gives the full
required content list — for a portfolio project the model card hits
the essentials; a real submission would add a quality-management
description and a list of harmonised standards followed.

## Article 13 — Transparency to users

> High-risk AI systems shall be designed and developed in such a way
> as to ensure that their operation is sufficiently transparent to
> enable users to interpret the system's output.

Three artefacts, each at a different audience:

- **Clinician (Streamlit dashboard).** Tab 1 shows the SHAP
  waterfall — *why* the model produced this prediction for this
  patient. Tab 2 shows the global beeswarm — *what features* the
  model relies on across the population.
- **Patient (Gradio Space NL translator, rule C45).** "Likely heart
  disease (high confidence)" / "Likely no heart disease (high
  confidence)" / "Uncertain — needs human review" instead of the
  raw mathematical set.
- **Auditor (Streamlit Tab 4 + MODEL_CARD).** Mondrian group-
  conditional coverage with Bonferroni-corrected p-values.

Article 13 also requires that the *limitations* of the system be
communicated. The MODEL_CARD has a full Limitations section that
explicitly flags the small sample, the under-represented women, the
8-row `age ≥ 65` bucket, and the marginal-coverage caveat.

## Article 14 — Human oversight

> High-risk AI systems shall be designed and developed in such a way
> that they can be effectively overseen by natural persons during the
> period in which the AI system is in use.

The conformal set size is the human-oversight signal. `len(set) ==
1` means "model is confident, you may use this label as a prior";
`len(set) > 1` means "model is *not* confident, defer to your own
judgement". The Gradio Space surfaces this directly via the NL
translator's "Uncertain — needs human review" branch.

α is the human-oversight dial. The deployed default α = 0.10 abstains
on ~35% of patients (see selective classification in MODEL_CARD).
A safety-critical deployment would lower α; a triage-only system
might raise it. *Exposing this dial is the engineering form of
Article 14.*

## Annex III §1 — Medical device safety components

The trigger condition. Marks the system as high-risk and pulls in
the entire Articles 8–15 obligation set. Nothing to implement here —
it is the legal hook, not an engineering requirement. But it is the
reason the other five articles apply at all.

## What this framing deliberately is not

It is not a conformity assessment. A CE-marked deployment under
Article 43 needs a notified body, post-market monitoring, an ISO
13485 QMS, and clinical evaluation. This document and the model card
are the *technical-documentation* slice of that — Article 11 — and
nothing more. The point of the exercise is to demonstrate that the
engineering surface (coverage monitor, pandera, NL translator,
Mondrian audit) maps cleanly onto the regulatory surface, so that a
real deployment would not have to redesign the pipeline to comply.

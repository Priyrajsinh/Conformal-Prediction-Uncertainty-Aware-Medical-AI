# 03 — Decision curve analysis

**Read**: 2026-05-09
**Sources**:
- Vickers & Elkin (2006). *Decision curve analysis: a novel method for evaluating prediction models.* Medical Decision Making 26(6): 565–574. doi:10.1177/0272989X06295361.
- Vickers, van Calster & Steyerberg (2019). *A simple, step-by-step guide to interpreting decision curve analysis.* Diagnostic and Prognostic Research 3:18.

## The question DCA answers

Discrimination metrics (AUC, accuracy, F1) answer "how good is the
classifier as a statistical object". They do not answer "is the
classifier good *enough that I should act on it*". DCA is the bridge.

Concrete version of the bridge: pick a decision threshold `p_t`. Treat
patients whose predicted risk is `≥ p_t`, do nothing for the rest.
Define

```
net_benefit(p_t) = TP / N − FP / N * (p_t / (1 − p_t))
```

The second term is the *cost* of a false positive expressed in
true-positive-equivalents, where the exchange rate `p_t / (1 − p_t)`
comes from the clinician's own threshold preference. A clinician who
acts at `p_t = 0.2` has implicitly said "I am willing to tolerate four
false positives to catch one true positive" — and net benefit
arithmetic respects that.

Two baselines on every DCA plot:

- **Treat all** — net benefit `= prevalence − (1 − prevalence) *
  (p_t / (1 − p_t))`. Drops below zero past the threshold where the
  prevalence is no longer worth the false-positive cost.
- **Treat none** — net benefit `= 0` everywhere.

A model is *useful* in the clinical sense when its DCA curve lies
above both baselines across the threshold range the clinician would
actually use. Above the curves but at thresholds nobody would ever use
in practice (say, `p_t = 0.85`) is not enough — that part of the
x-axis is decorative.

## On this project's data

`reports/figures/dca_net_benefit.png` plots the three curves on the
60-patient test set with prevalence ≈ 0.467. Numbers from
`reports/results.json::dca`:

| Threshold p_t | Model | Treat-all | Treat-none |
|---------------|------:|----------:|-----------:|
| 0.10 | 0.435 | 0.407 | 0 |
| 0.20 | 0.417 | 0.333 | 0 |
| 0.30 | 0.371 | 0.238 | 0 |
| 0.40 | 0.356 | 0.111 | 0 |
| 0.50 | 0.333 | −0.067 | 0 |

The model dominates both baselines from `p_t = 0.05` to roughly
`p_t = 0.85` (above `0.85` everything converges as the curves get
noisy). The clinically interesting range for cardiac risk
stratification is `0.10–0.40`; the model adds the equivalent of
≈ 0.05–0.08 *extra true positives per patient* over treat-all in that
band.

## Common misuses (Vickers et al. 2019)

1. **Reporting net benefit at a single threshold and stopping there.**
   The point of DCA is the curve. A model could win at `p_t = 0.5`
   and lose at `p_t = 0.2`, which is where the clinician actually
   operates.
2. **Comparing AUC and saying "we did DCA too".** The two metrics
   answer different questions. If they disagree, DCA wins for the
   deployment decision.
3. **DCA on a population whose prevalence does not match
   deployment.** Net benefit is prevalence-sensitive (the treat-all
   line literally is the prevalence at `p_t = 0`). Run it on the
   target population, not just the original validation cohort.

## Why DCA belongs in the model card

EU AI Act Article 13 ("transparency") and Article 14 ("human
oversight") both lean on the clinician understanding *when* the model
is useful. DCA is the visual form of that — one image, three lines,
the clinician can see at which threshold their judgement should
defer to the model and at which it should not. AUC cannot give them
that.

## How this maps onto the code

- `src/evaluation/dca.py::decision_curve` — sweeps `p_t` from 0.01 to
  0.99, returns the three curves, writes them into
  `reports/results.json::dca`.
- `src/evaluation/plot_dca.py::plot_dca` — renders
  `reports/figures/dca_net_benefit.png`. Embedded in both the README
  and the MODEL_CARD.
- `tests/test_evaluation.py::test_dca_dominates_in_clinical_range` —
  regression test: model net benefit ≥ treat-all net benefit at every
  threshold in `[0.05, 0.40]`.

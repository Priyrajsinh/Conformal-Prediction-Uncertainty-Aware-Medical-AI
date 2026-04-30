## Summary
<!-- 1–3 bullets — what changed, why -->

Closes #

## Type
- [ ] feat
- [ ] fix
- [ ] test
- [ ] chore
- [ ] refactor
- [ ] docs

## Local CI (rule C46)
Ran `make ci` locally — every step green:
- [ ] black --check
- [ ] isort --check-only
- [ ] flake8
- [ ] mypy
- [ ] bandit (zero findings)
- [ ] radon (no grade C+ functions)
- [ ] interrogate (>=80% docstrings)
- [ ] pip-audit (zero CVEs)
- [ ] detect-secrets (no new secrets)
- [ ] pytest (coverage >= 70%)

## Conformal-prediction guarantees (when touching model/eval code)
- [ ] THREE-WAY split unchanged (no train/cal/test leakage)
- [ ] Calibration still uses `cv='prefit'` on cal split only
- [ ] Empirical coverage >= 1−α verified for all alphas
- [ ] `safe_predict()` still wraps every `MAPIE.predict_sets()` call
- [ ] `pandera HEART_DISEASE_SCHEMA.validate(df)` still runs before split

## Test plan
<!-- How a reviewer can verify this manually -->

1.
2.
3.

# Calibration tuning: reducing overconfidence (optimism)

The dunning recovery model outputs probabilities that are used for ranking and for business rules (e.g. “retry when P(success) > 3× baseline”). If the model is **too optimistic** (probabilities systematically higher than actual success rates), EV calculations and opportunity thresholds become misleading. This document summarizes what is implemented and gives tuning suggestions.

---

## What is implemented (post–March 2026 refresh)

1. **Dedicated calibration set**  
   Calibration (isotonic regression) is fit on a **held-out temporal slice** (e.g. 2026-02-15 to 2026-02-28), not on the same validation set used for early stopping. That reduces overfitting of the calibrator to the eval set and usually improves reliability of probabilities.

2. **Temperature scaling**  
   After isotonic calibration, an optional **temperature** \(T > 1\) is applied:  
   `p_cal = sigmoid(logit(p_cal) / T)`.  
   This pulls probabilities toward 0.5 and reduces optimism. Default in training: `CALIBRATION_TEMPERATURE=1.25` (set via env). Increase (e.g. 1.5 or 2.0) if validation reliability plots still show overconfidence.

3. **Train / Cal / Val split**  
   - **Train:** CatBoost fit and early-stopping eval.  
   - **Cal:** Isotonic (and temperature) fit only.  
   - **Val:** Reporting only (AUC, PR-AUC, reliability).  

   So validation metrics reflect how well the *calibrated* model generalizes, and the calibrator is not tuned to the same data used for reporting.

---

## Tuning suggestions (when the model is still too optimistic)

### 1. Increase temperature (simplest)

- **Env:** `CALIBRATION_TEMPERATURE=1.5` or `2.0` before running the training script.  
- **Effect:** Stronger pull toward 0.5; high probabilities are reduced more.  
- **Trade-off:** Too high a value can over-dampen and hurt ranking (AUC may drop slightly). Tune on validation reliability (e.g. reliability diagram, ECE) and business thresholds.

### 2. Use more calibration data

- Extend the calibration window (e.g. `CAL_START` / `CAL_END`) so the isotonic curve is estimated with more points.  
- More data usually gives a more stable and less overfit calibrator.

### 3. Cross-validated calibration (advanced)

- Fit the calibrator on **out-of-fold predicted probabilities** from the training set (e.g. 5-fold by `linked_invoice_id`), then apply to a single final model.  
- This is similar to `sklearn.calibration.CalibratedClassifierCV(cv=5, method='isotonic')`.  
- Pros: less overfitting to one temporal slice. Cons: more code and compute; need to ensure temporal consistency if you care about time-based leakage.

### 4. Sigmoid (Platt) instead of isotonic

- **When:** Calibration set is small or noisy; isotonic can overfit.  
- **How:** Replace `IsotonicRegression` with a single logistic regression of \(y\) on \(\logit(p)\) (or use `CalibratedClassifierCV(..., method='sigmoid')`).  
- **Effect:** Smoother, more conservative curve; often less extreme probabilities than isotonic on small data.

### 5. Reliability diagram and ECE

- After training, plot **reliability diagram** on validation: bin predicted probabilities, plot mean predicted vs mean actual per bin. If the curve lies below the diagonal for high probabilities, the model is optimistic.  
- **ECE (Expected Calibration Error):** weight-average of |mean pred − mean actual| per bin. Use it to compare temperature or calibration-window choices.  
- The script `compare_shadow_vs_actual_20260206.py` and related reports can feed into this.

### 6. Clip or cap probabilities (operational guardrail)

- In production or shadow, cap max probability (e.g. `min(p_cal, 0.5)`) for certain segments if you only need ranking and want to avoid overconfident actions.  
- Prefer fixing calibration (temperature, more cal data, sigmoid) first; use capping only as a last resort.

### 7. Per-segment calibration (future)

- If overconfidence is concentrated in certain segments (e.g. a country or decline code), consider separate calibration curves or temperatures per segment.  
- Start with global temperature; only add segment-specific calibration if validation/reliability plots show clear segment bias.

---

## Quick reference: env vars for training

| Env var | Default | Effect |
|--------|--------|--------|
| `FORCE_QUERY` | (unset) | Set to `1` to re-fetch raw data from BigQuery. |
| `CALIBRATION_TEMPERATURE` | `1.25` | Temperature applied after isotonic. Increase to reduce optimism. |

---

## References

- Training script: `scripts/train_dunning_v2_20260301.py` (train/cal/val split, isotonic on cal set, temperature).
- Model class: `src/model.py` — `IsotonicCalibratedClassifier(temperature=...)`.
- **References:** See **docs/EVALUATION.md** for how to evaluate model and calibration (train-time and shadow vs actuals).
- Shadow vs actual (reliability / calibration): `scripts/compare_shadow_vs_actual_20260206.py`, `docs/SHADOW_VS_ACTUAL_COMPARISON.md`.

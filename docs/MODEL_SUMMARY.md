# Dunning Recovery Model — Summary

This document summarizes the **dunning recovery model** used to predict the probability of payment success on the *next* retry attempt, and to recommend an optimal retry time (24–120 hours from "now") for invoices in a soft-decline dunning cycle.

---

## 1. Model Overview

| Item | Description |
|------|-------------|
| **Purpose** | Predict P(success) on the next attempt and choose the best retry slot (4-hour resolution, 24–120h ahead) to maximize expected recovery. |
| **Algorithm** | **CatBoost** classifier (binary: success vs no success on next attempt) with **isotonic regression** calibration on the validation set. |
| **Target** | `is_success` — 1 if the attempt (row) has `status == 'success'`, else 0. |
| **Scope** | **Soft-decline** dunning attempts only (filter: `prev_decline_type == "Soft decline"`, `is_attached_invoice_1st_attempt == "Dunning attempt"`). |
| **Production artifact** | Calibrated model: `models/catboost_dunning_calibrated_20260224.joblib`. Raw CatBoost: `models/catboost_dunning_20260224.joblib`. |

---

## 2. Data and Temporal Split

- **Source:** BigQuery view `aa-datamart.billing_dm.MISc_vw_txn_enriched_subID_fallback` (via `src.data.pipeline.run_pipeline()`). Raw data is cached to `data/raw/dunning_raw.parquet`; set `FORCE_QUERY=1` to refresh.
- **Training:** `2025-01-02` through `2026-02-07` (inclusive).
- **Validation:** `2026-02-08` through `2026-02-14` (inclusive). Used for early stopping, calibration fit, and reported AUC/PR-AUC.
- **Holdout / shadow:** No data from `2026-02-15` onward is used for training or validation; that period is reserved for shadow monitoring and comparison vs actual outcomes.

---

## 3. Features

The model uses **20 features** (same set in `train_dunning_v2_20260206.py`, `shadow_monitoring_20260206.py`, and `dunning_modeling.ipynb`). All are derived from the **current** attempt’s context and the **previous** attempt’s outcome (for “next attempt” prediction).

### 3.1 Feature list

| Feature | Type | Description |
|---------|------|-------------|
| `prev_decline_code` | Categorical | Normalized decline code of the *previous* attempt (e.g. insufficient_funds, do_not_honor). At inference: latest attempt’s `Decline_code_norm` → “previous” for next prediction. |
| `hour_sin`, `hour_cos` | Numeric | Cyclic encoding of **local** hour of the attempt (or of the candidate retry time in shadow). |
| `dow_sin`, `dow_cos` | Numeric | Cyclic encoding of day of week (local). |
| `day_sin`, `day_cos` | Numeric | Cyclic encoding of day of month (local). |
| `dist_to_payday` | Numeric | Min distance to common paydays (1, 15, 30). |
| `log_charge_amount` | Numeric | `log1p(amount)`. |
| `is_debit` | Numeric | 1 if funding type is debit, else 0. |
| `amt_per_attempt` | Numeric | `amount / (invoice_attempt_no + 1)`. |
| `time_since_prev_attempt` | Numeric | Hours since the previous attempt (`updated_at - prev_attempt_time` in training; at inference: hours since latest failure). |
| `cumulative_delay_hours` | Numeric | Hours since the first attempt on the invoice. |
| `billing_country` | Categorical | Billing country. |
| `gateway` | Categorical | Payment gateway (e.g. stripe, adyen). |
| `funding_type_norm` | Categorical | Normalized funding type (e.g. credit, debit). |
| `card_brand` | Categorical | Card brand (e.g. visa, mastercard). |
| `prev_card_status` | Categorical | Card status from the *previous* attempt. At inference: latest attempt’s `card_status`. |
| `Domain_category` | Categorical | Domain category (e.g. Work, Gmail). |
| `invoice_attempt_no` | Numeric | Attempt number on the invoice (from latest record in shadow). |

**Excluded from model input:** `linked_invoice_id` (group/id only), `is_success` (target).

### 3.2 Categorical features (CatBoost)

```text
prev_decline_code, billing_country, gateway, funding_type_norm, card_brand, Domain_category, prev_card_status
```

---

## 4. Training Configuration

- **CatBoost (no class weighting):** Calibration is applied later, so the classifier is trained with default class handling.
- **Hyperparameters:**  
  `iterations=1000`, `learning_rate=0.05`, `depth=6`, `eval_metric="AUC"`, `early_stopping_rounds=50`, `random_seed=42`.
- **Calibration:** Isotonic regression fitted on **validation set** predictions; out-of-bounds predictions are clipped to [0, 1]. The production artifact is the **calibrated** model.

---

## 5. Training / Validation / Testing Metrics

- **Validation set:** The only formal evaluation in the training script is on the **validation** period (`2026-02-08`–`2026-02-14`). The script prints:
  - **AUC** (ROC-AUC) on validation (calibrated probabilities).
  - **PR-AUC** (average precision) on validation (calibrated probabilities).
- **No separate test set in code:** Data from `2026-02-15` onward is not used for training or validation; it is used for **shadow** comparison only (see `compare_shadow_vs_actual_20260206.py` and `SHADOW_VS_ACTUAL_COMPARISON.md`).
- **Notebook reference:** In `dunning_modeling.ipynb`, a time-based holdout (train on past, test on future) reports metrics in the **~0.74–0.75 AUC** and **~0.09 PR-AUC** range; actual values depend on the exact split and data. Running `train_dunning_v2_20260206.py` gives the authoritative validation AUC and PR-AUC for the production train/val split.
- **Shadow evaluation:** Post-deployment evaluation is done via `compare_shadow_vs_actual_20260206.py` (decile lift, TTR, EV lift, calibration, etc.), not as a single “test” metric.

**Summary table (conceptual):**

| Split | Date range | Use |
|-------|------------|-----|
| Train | 2025-01-02 – 2026-02-07 | Fit CatBoost |
| Validation | 2026-02-08 – 2026-02-14 | Early stopping, calibration fit, AUC/PR-AUC |
| Holdout / shadow | 2026-02-15+ | Shadow monitoring and comparison vs actuals |

---

## 6. Outputs

| Output | Path | Description |
|--------|------|-------------|
| Raw model | `models/catboost_dunning_20260224.joblib` | CatBoost only (no calibration). |
| Calibrated model | `models/catboost_dunning_calibrated_20260224.joblib` | CatBoost + isotonic calibrator; **used in production and shadow**. |
| PR curve (validation) | `eval_plots_20260224.png` | Precision–recall curve on validation set. |

---

## 7. Downstream Use

- **Shadow monitoring** (`shadow_monitoring_20260206.py`): Loads the calibrated model, scores candidate retry slots (24–120h, 4-hour steps) per invoice, and writes `artifacts/shadow_log.csv` and `artifacts/shadow_slot_log.csv`.
- **Comparison vs actuals** (`compare_shadow_vs_actual_20260206.py`): Compares shadow predictions to BigQuery outcomes (recovery, TTR, decile lift, EV lift, decline-code matrix). See `SHADOW_VS_ACTUAL_COMPARISON.md`.

---

## 8. Why the Chosen Model Is Optimal for This Use Case

From a data science management perspective, the choice of **CatBoost + isotonic calibration** is well aligned with the dunning recovery problem in several ways.

**1. Mixed feature types and categoricals**  
We have a mix of numeric features (time since last attempt, amount, cyclic time) and high-cardinality categoricals (decline code, country, gateway, card brand, domain). Tree-based models like CatBoost handle both natively without one-hot explosion, and CatBoost’s ordered boosting and built-in handling of categories reduce overfitting and tuning burden. That fits our feature set and data volume.

**2. Ranking and rare positives**  
The business need is to **rank** which invoices (and which retry times) are most likely to succeed, not to classify every case. AUC is a ranking metric, and we optimize it directly with early stopping on a validation set. Recovery is rare (low positive rate), so we care about PR-AUC and lift in the top deciles; CatBoost performs well on imbalanced ranking, and we add **isotonic calibration** so that the predicted probabilities are usable for expected-value and “high-value opportunity” thresholds, not only for ordering.

**3. Interpretability and robustness**  
Trees give feature importance and are robust to scale and moderate noise. We have many categoricals that can be noisy (e.g. decline codes); CatBoost’s regularization and ordered boosting help. We don’t need a black box for this use case; stakeholders can inspect which factors (decline code, time since last attempt, country, etc.) drive the score.

**4. Production and shadow alignment**  
The same feature set is used in training and in shadow monitoring; we score many candidate slots per invoice (24–120h) by changing only the time features. CatBoost is fast at inference and easy to ship (single artifact plus calibrator). No need for a separate preprocessing pipeline that would drift from training.

**5. Calibration for business metrics**  
Raw CatBoost probabilities are often overconfident or underconfident. We use **isotonic regression** on the validation set to map raw scores to better-calibrated probabilities. That makes “P(success) &gt; 3× baseline” and “expected value = amount × (P − baseline)” meaningful for opportunity gap and EV lift, rather than relying on uncalibrated scores.

**6. Why not alternatives (brief)**  
- **Logistic regression:** Would require more feature engineering and might underfit the interactions (e.g. decline code × time since attempt).  
- **Neural nets:** Unnecessary complexity and harder to deploy for this tabular, medium-size data; no clear accuracy gain.  
- **Other gradient boosting (XGBoost, LightGBM):** CatBoost’s handling of categories and ordered boosting are a good fit here; the team standard is CatBoost for this pipeline.

Overall, the combination gives strong ranking (AUC), interpretable drivers, calibrated probabilities for business rules and EV, and a simple path to production and shadow evaluation.

---

## 9. What Is Isotonic Regression? (Plain-Language Explanation)

**For someone with little ML/statistics background:**

**The problem**  
The model (CatBoost) outputs a number between 0 and 1 for each attempt—a “raw score” that we treat as a probability. But often these raw scores are not true chances. For example, when the model says “10% chance,” we might see that in reality 5% of such cases succeed, or 20%. So the **ordering** (who has higher chance) is good, but the **scale** (the actual percentage) is off.

**What isotonic regression does**  
We take the model’s raw scores and the **actual** outcomes (success / no success) on a separate set of data (our validation set). We then fit a **monotone** curve: “if the raw score goes up, the corrected probability must go up (or stay flat), never down.” So we’re not changing who is ranked first or last; we’re **adjusting the numbers** so that, on average, when the model says “X%,” the true success rate in that bucket is closer to X%. It’s like recalibrating a thermometer so that the numbers match reality, without changing the order of “warmer” vs “colder.”

**Why “isotonic”?**  
“Isotonic” means “same order”: the correction preserves the order of the model’s scores. So we keep the good ranking we trained for, and fix the scale so that the probabilities are more trustworthy for business rules (e.g. “retry when P(success) &gt; 3× baseline”) and for expected-value calculations.

**In one sentence:**  
Isotonic regression is a way to fix the model’s probabilities so they better match real success rates, while keeping the same ordering of who is more or less likely to succeed.

---

## 10. References

- **Training script:** `train_dunning_v2_20260206.py`
- **Feature alignment (shadow):** `shadow_monitoring_20260206.py` — “previous” = latest attempt’s `Decline_code_norm` / `card_status`; `time_since_prev_attempt` = inference_run_at − latest_failure_updated_at; `cumulative_delay_hours` = inference_run_at − first_attempt_at.
- **Notebook (exploration / alternate splits):** `notebook/dunning_modeling.ipynb`
- **Shadow vs actual:** `SHADOW_VS_ACTUAL_COMPARISON.md`

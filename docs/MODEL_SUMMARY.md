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

The model uses **21 features** (same set in `train_dunning_v2_20260206.py`, `shadow_monitoring_20260206.py`, and `dunning_modeling.ipynb`). All are derived from the **current** attempt’s context and the **previous** attempt’s outcome (for “next attempt” prediction).

### 3.1 Feature list

| Feature | Type | Description |
|---------|------|-------------|
| `prev_decline_code` | Categorical | Normalized decline code of the *previous* attempt (e.g. insufficient_funds, do_not_honor). At inference: latest attempt’s `Decline_code_norm` → “previous” for next prediction. |
| `prev_advice_code_group` | Categorical | Advice code group of the *previous* attempt. At inference: latest attempt’s `advice_code_group` → "previous" for next prediction. Processed like decline code (fillna UNKNOWN). |
| `hour_sin`, `hour_cos` | Numeric | Cyclic encoding of **local** hour of the attempt (or of the candidate retry time in shadow). |
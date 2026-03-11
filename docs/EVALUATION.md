# Model and calibration evaluation

This document describes **how to evaluate** the dunning recovery model's **discriminative performance** (ranking, AUC, PR-AUC) and **calibration performance** (whether predicted probabilities match actual success rates). There are two evaluation paths: at **training time** (validation set) and **post-deployment** (shadow vs actuals).

---

## 1. At training time (validation set)

After you run the training script, it evaluates the **calibrated** model on the **validation** set (no data used for calibration). You get discriminative and calibration metrics plus two plots.

### 1.1 Commands

```bash
cd aa_dunning_modeling
python scripts/train_dunning_v2_20260301.py
# Optional: FORCE_QUERY=1 to re-fetch data; CALIBRATION_TEMPERATURE=1.5 to tune calibration
```

### 1.2 Discriminative metrics (ranking)

| Metric | Meaning | Use |
|--------|--------|-----|
| **AUC (ROC-AUC)** | Ability to rank success vs no-success. 0.5 = random, 1.0 = perfect. | Main ranking metric; higher is better. |
| **PR-AUC (average precision)** | Precision-recall area under curve; better for imbalanced data. | Complements AUC when positives are rare. |

Interpretation: high AUC/PR-AUC means the model orders "likely to recover" vs "unlikely" well. These do **not** tell you whether the probability values are correct (e.g. whether "30%" really corresponds to 30% success rate); that is calibration.

### 1.3 Calibration metrics (probability quality)

| Metric | Meaning | Use |
|--------|--------|-----|
| **Brier score** | Mean squared error between predicted probability and 0/1 outcome. | Lower is better; combines discrimination and calibration. |
| **ECE (Expected Calibration Error)** | Weighted average over probability bins of |mean predicted - actual rate|. | Lower = better calibration. |
| **MCE (Max Calibration Error)** | Worst bin's |mean predicted - actual rate|. | Highlights worst miscalibration. |

Interpretation: if the model is **too optimistic**, you'll see high predicted probabilities but actual success rates lower (e.g. predicted 40%, actual 15%). That shows up as high ECE/MCE and in the reliability diagram (predicted bar above actual). See **docs/CALIBRATION_TUNING.md** for how to reduce optimism (e.g. temperature, more calibration data).

### 1.4 Outputs

| Output | Path | Description |
|--------|------|-------------|
| Validation metrics | Console | AUC, PR-AUC, Brier, ECE, MCE. |
| PR curve | `reports/eval_plots_20260311.png` | Precision-recall curve on validation set. |
| Reliability diagram | `reports/reliability_20260311.png` | Deciles: mean predicted vs actual rate. Below diagonal in high deciles = optimistic. |

---

## 2. Post-deployment (shadow vs actuals)

Once the model is running in **shadow mode** (logging suggested retry times and probabilities without changing Chargebee behavior), you can compare its predictions to **actual** outcomes from BigQuery. This gives calibration and discriminative metrics on **real** post-deployment data.

### 2.1 Prerequisites

- Shadow log: `artifacts/shadow_log.csv` (from `scripts/shadow_monitoring_20260206.py`).
- Actual outcomes: from BigQuery (default) or a CSV with columns such as `invoice_id`, `recovered`, `recovered_at`, `last_attempt_at`.

### 2.2 Commands

```bash
cd aa_dunning_modeling
python scripts/compare_shadow_vs_actual_20260206.py

# Optional overrides (env or args):
# --shadow artifacts/shadow_log.csv
# --output artifacts/shadow_vs_actual_comparison.csv
# --report artifacts/shadow_vs_actual_report.txt
# --cal-plot artifacts/calibration_plot.png
# --gains-plot artifacts/cumulative_gains_plot.png
# --actual-csv path/to/actuals.csv   # use CSV instead of BQ
```

### 2.3 What the script does

1. Loads the shadow log and actual outcomes (BQ or CSV).
2. Merges on `invoice_id` (last-mile: latest record per invoice).
3. Computes:
   - **Calibration:** Brier, ECE, MCE (same as training; lower is better).
   - **Discriminative:** AUC-ROC, AUC-PR.
   - **Decile table:** count, recovered, mean predicted, actual rate, calibration error per decile.
   - **Top decile lift:** how much higher recovery rate is in the top decile vs global rate.
   - **TTR (time to recover), EV lift, temporal windows (±6h, ±12h, ±24h), decline-code matrix.**
4. Writes:
   - Comparison CSV (one row per invoice with predictions and actuals).
   - Text report with all sections above.
   - **Calibration (reliability) plot:** mean predicted vs actual rate by decile.
   - **Cumulative gains plot:** % of recoveries captured vs % of invoices (sorted by model probability).

### 2.4 Outputs

| Output | Default path | Description |
|--------|---------------|-------------|
| Comparison CSV | `artifacts/shadow_vs_actual_comparison.csv` | Merged shadow + actuals; for further analysis. |
| Text report | `artifacts/shadow_vs_actual_report.txt` | Executive summary, calibration metrics, discriminative metrics, deciles, TTR, EV, etc. |
| Calibration plot | `artifacts/calibration_plot.png` | Reliability diagram (predicted vs actual by decile). |
| Cumulative gains | `artifacts/cumulative_gains_plot.png` | Lift curve: % of recoveries captured vs % of invoices. |

### 2.5 Interpreting the calibration plot

- **Perfect calibration:** bars for "Mean predicted" and "Actual rate" align (diagonal).
- **Optimistic:** "Mean predicted" above "Actual rate" in high deciles — model overestimates success probability; consider increasing `CALIBRATION_TEMPERATURE` or other steps in **docs/CALIBRATION_TUNING.md**.
- **Pessimistic:** "Mean predicted" below "Actual rate" in high deciles — model underestimates.

---

## 3. Quick reference

| Goal | Where | What to run / look at |
|------|--------|------------------------|
| Ranking (AUC, PR-AUC) at train time | Validation set | `python scripts/train_dunning_v2_20260301.py` — console + `reports/eval_plots_*.png` |
| Calibration at train time | Validation set | Same run — Brier, ECE, MCE on console + `reports/reliability_*.png` |
| Ranking + calibration on live data | Shadow vs actuals | `python scripts/compare_shadow_vs_actual_20260206.py` — report + `artifacts/calibration_plot.png` |
| Reduce overconfidence | Training + tuning | **docs/CALIBRATION_TUNING.md** (temperature, cal set, etc.) |

---

## 4. References

- **Training script:** `scripts/train_dunning_v2_20260301.py` — train/cal/val split, validation metrics and reliability diagram.
- **Shadow comparison:** `scripts/compare_shadow_vs_actual_20260206.py`; **docs/SHADOW_VS_ACTUAL_COMPARISON.md** for full pipeline description.
- **Calibration tuning:** **docs/CALIBRATION_TUNING.md** — how to improve calibration when the model is too optimistic or pessimistic.

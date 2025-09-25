# Uncertainty-Aware Medical AI for Glaucoma Diagnosis

This project explores the limitations of using **accuracy as the sole evaluation metric** in medical AI, with a focus on the risk of **false negatives** in glaucoma diagnosis. To address this, we apply **Conformal Prediction (CP)** to partition predictions into *confident* and *uncertain* groups, allowing ambiguous cases to be flagged for expert review. We benchmark two scenarios: an **upper bound** with a perfect oracle expert and a **lower bound** using a GPT proxy, illustrating how uncertainty-aware methods improve patient safety beyond what accuracy alone can capture.

---

## 📂 Project Structure

- `notebooks/`
  - `03_conformal_prediction.ipynb` → main notebook for running experiments and reproducing results
- `src/`
  - `data/` → preprocessing, dataset splitting, inspection
  - `models/` → baseline logistic regression, CP routines, expert integration
  - `visualization/` → plotting utilities for calibration, probability distributions, CP results
- `reports/figures/` → all generated plots
- `data/`
  - `raw/` → expected raw data (`ds_whole.csv`)
  - `processed/` → train / calibration / test splits (auto-generated)

---

## 🚀 How to Run

The easiest way to reproduce everything is to run the main notebook:

```bash
notebooks/03_conformal_prediction.ipynb

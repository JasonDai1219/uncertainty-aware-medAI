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

## ⚙️ Requirements
```bash
pip install -r requirements.txt

## 📊 Outputs / Results

Figures will be saved automatically in reports/figures/, including:
	•	baseline_confusion.png – baseline confusion matrix
	•	baseline_calibration.png – calibration curve showing uncertainty zone
	•	cp_fn_coverage.png – false negatives vs alpha under oracle and GPT proxy
	•	cp_fn_gpt_measured_bootstrap.png – FN reduction curve using GPT-measured sensitivity

Study Results are inside the notebook.


## ⚠ Disclaimer

This is a research prototype, not a clinical tool. The GPT proxy is used only as a pragmatic lower bound and should not be interpreted as a medical expert.


## 📜 License
MIT License
Copyright (c) 2025 Ruizhe (Jason) Dai

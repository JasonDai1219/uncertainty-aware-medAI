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
```

## ⚙️ Requirements
```bash
pip install -r requirements.txt
```

## 📊 Outputs / Results

All figures are automatically saved under `reports/figures/`:

- **baseline_confusion.png** → Confusion matrix of the logistic regression baseline  
- **baseline_calibration.png** → Calibration curve highlighting the uncertain zone  
- **cp_fn_coverage.png** → False negatives vs. α, comparing oracle expert and GPT proxy scenarios  
- **cp_fn_gpt_measured_bootstrap.png** → FN reduction curve using GPT-measured sensitivity (bootstrap)

Additional quantitative results of the study are available directly inside the notebook (`conformal_prediction.ipynb`).


## ⚠ Disclaimer

This is a research prototype, not a clinical tool. The GPT proxy is used only as a pragmatic lower bound and should not be interpreted as a medical expert.


## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

# Uncertainty-Aware Medical AI for Glaucoma Diagnosis

This project explores the limitations of accuracy as a sole metric in medical AI, focusing on the risk of false negatives in glaucoma diagnosis. We apply **Conformal Prediction (CP)** to separate confident from uncertain predictions, enabling expert review of ambiguous cases. The study quantifies both an **upper bound** (oracle expert) and a **lower bound** (LLM proxy) of performance improvement when uncertainty is considered.

---

## 🚀 How to Run

The easiest way to reproduce everything is to run the main notebook:

```bash
notebooks/03_conformal_prediction.ipynb

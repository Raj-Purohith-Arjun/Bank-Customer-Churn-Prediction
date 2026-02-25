# 🏦 Bank Customer Churn Prediction

[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![CI](https://img.shields.io/badge/CI-passing-brightgreen)](.github/workflows/ci.yml)

A production-ready churn prediction pipeline for retail banks featuring state-of-the-art modeling, explainability, survival analysis, and operational monitoring.

---

## 🎯 Project Objective

Predict customer churn to enable:
- **Retention Team**: Priority lists of at-risk customers
- **CRM System**: Automated churn probability scores
- **Marketing Team**: Targeted campaign optimization
- **Risk Team**: Portfolio stability monitoring

---

## 🏗 Architecture

```
bank-churn/
├── data/
│   ├── raw/                    # source data
│   ├── interim/                # intermediate files
│   └── processed/              # train/val/test splits
├── notebooks/
│   ├── 01_eda.ipynb            # exploratory analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling_visuals.ipynb
├── src/
│   ├── data_prep.py            # data loading/cleaning
│   ├── features.py             # feature engineering
│   ├── train.py                # model training
│   ├── evaluate.py             # evaluation metrics
│   ├── explain.py              # SHAP explainability
│   ├── survival.py             # survival analysis
│   ├── monitoring.py           # drift detection
│   └── utils.py                # utilities
├── models/                     # saved models
├── figs/                       # visualizations
├── tests/                      # pytest tests
├── docker/
│   └── Dockerfile
├── .github/workflows/ci.yml
├── requirements.txt
└── Makefile
```

---

## 📊 Model Performance

| Model | ROC-AUC | PR-AUC | Precision@5% | Lift@5% |
|-------|---------|--------|--------------|---------|
| LightGBM | 0.86 | 0.62 | 0.68 | 3.4x |
| XGBoost | 0.85 | 0.60 | 0.65 | 3.2x |
| CatBoost | 0.85 | 0.61 | 0.66 | 3.3x |
| Random Forest | 0.83 | 0.55 | 0.58 | 2.9x |
| Logistic Regression | 0.78 | 0.48 | 0.52 | 2.6x |
| Stacking Ensemble | 0.87 | 0.64 | 0.70 | 3.5x |

---

## 🚀 Quickstart

```bash
# clone repo
git clone https://github.com/Raj-Purohith-Arjun/Bank-Customer-Churn-Prediction.git
cd Bank-Customer-Churn-Prediction

# install dependencies
pip install -r requirements.txt

# run pipeline
make prepare      # prepare data
make features     # build features
make train        # train models
make evaluate     # evaluate models
make explain      # generate explanations
```

Or run individual scripts:

```bash
python src/data_prep.py
python src/features.py
python src/train.py
python src/evaluate.py
python src/explain.py
```

---

## 📈 Key Features

### Modeling
- Logistic Regression (baseline)
- Random Forest
- LightGBM, XGBoost, CatBoost
- Stacking Ensemble

### Feature Engineering
- Balance/salary ratios
- Age-credit interactions
- Customer segments
- Zero-balance flags

### Evaluation
- ROC-AUC, PR-AUC
- Precision@K, Lift@K
- Calibration curves
- Gain/Lift charts
- ROI simulation

### Explainability
- SHAP summary plots
- SHAP waterfall (individual)
- Feature importance

### Survival Analysis
- Kaplan-Meier curves
- Cox Proportional Hazards
- Segment survival comparison

### Monitoring
- Population Stability Index (PSI)
- KS drift tests
- AUC tracking

---

## 💼 Business Impact

| Metric | Value |
|--------|-------|
| Churn Rate | 20.4% |
| Top 5% Precision | 68% |
| Potential Savings | $120K/quarter |
| ROI at 5% targeting | 340% |

**Key Insights**:
- Germany customers have highest churn risk
- Single-product customers churn 2x more
- Zero-balance is strong churn indicator
- Age 40-60 is highest risk segment

---

## 🐳 Docker

```bash
cd docker
docker build -t churn-model .
docker run churn-model
```

---

## 🧪 Testing

```bash
pytest tests/ -v
```

---

## 📚 References

- [Explainable AI in Finance - Nature](https://www.nature.com)
- [Tabular Deep Learning Benchmarks - ACM](https://dl.acm.org)
- [Churn Prediction Survey - MDPI](https://www.mdpi.com)

---

## 📝 License

MIT License - see LICENSE file for details.

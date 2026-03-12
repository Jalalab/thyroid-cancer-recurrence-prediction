# 🏥 A Machine Learning Framework for Predicting Thyroid Cancer Recurrence

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![SMOTE](https://img.shields.io/badge/Imbalance-SMOTE-green)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-red)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

A reproducible machine learning pipeline to predict **thyroid cancer recurrence** 
using routinely collected clinical and pathological features from 5,351 patients.

> 📄 **Course:** CSE445 — North South University, Dhaka, Bangladesh (2025)  
> 👨‍💻 **Author:** Jalal Abedin Pial  
> 🎓 **Supervisor:** Dr. Sifat Momen

---

## 📌 Problem Statement

Thyroid cancer recurrence after initial treatment is a critical clinical challenge.
Early identification of high-risk patients enables:
- More intensive surveillance
- Earlier diagnostic work-up
- Consideration of adjuvant therapy

This project builds a **binary classifier** to predict whether a patient will experience 
recurrence (1) or not (0) based on pre-treatment clinical features.

---

## 📂 Dataset

- **Source:** Soegaard Ballester, J., & Wachtel, H. (2022). *Institutional Thyroid Cancer Dataset*. Mendeley Data, V1. [DOI: 10.17632/57726vkm48.1](https://doi.org/10.17632/57726vkm48.1)
- **Size:** 5,351 patients × 59 features
- **Period:** January 2013 – December 2015
- **Target:** `recurred` — binary label (recurrence vs. no recurrence)
- **Class Imbalance:** Only **2.7% recurrence cases** (140 out of 5,351 patients)
- **Note:** Dataset is fully de-identified/anonymized for patient privacy

---

## 🧠 ML Pipeline
```
Raw Clinical Data (5,351 patients × 59 features)
        ↓
Remove leaky/ID columns (post-treatment variables)
        ↓
Auto-select pre-operative safe features
        ↓
Preprocessing Pipeline (ColumnTransformer)
  ├── Numeric  → Median Imputation → StandardScaler
  └── Categorical → Mode Imputation → OneHotEncoder
        ↓
Stratified Train/Test Split (80/20)
        ↓
SMOTE Oversampling (training set only)
        ↓
4 Models × Two-Stage Hyperparameter Tuning
  ├── RandomizedSearchCV (broad search)
  └── GridSearchCV (fine-tuning)
        ↓
Evaluation: Accuracy + ROC-AUC + F1 + Confusion Matrix
        ↓
SHAP Explainability (Random Forest)
        ↓
Save Models (.joblib)
```

---

## 🤖 Models & Results

| Model | Test Accuracy | ROC-AUC |
|---|---|---|
| 🥇 **Random Forest** | **97.5%** | **0.813** |
| 🥈 SVM | 96.5% | 0.663 |
| 🥉 Decision Tree | 94.8% | 0.590 |
| Logistic Regression | 82.3% | 0.739 |

**Winner: Random Forest** — highest accuracy and best AUC score.

---

## ⚙️ Key Techniques

| Challenge | Solution |
|---|---|
| Severe class imbalance (2.7% recurrence) | SMOTE + class-weighted learning |
| Mixed data types (numeric + categorical) | ColumnTransformer pipeline |
| Missing values | Median (numeric) / Mode (categorical) imputation |
| Data leakage | Strict removal of post-treatment variables |
| Hyperparameter tuning | RandomizedSearchCV → GridSearchCV (2-stage) |
| Model explainability | SHAP TreeExplainer + feature importance |

---

## 🔍 Top Predictive Features (Random Forest)

Based on feature importance and SHAP analysis:
1. `ClinN_c0` — Lymph node staging
2. `surg.to.radtx.days` — Days from surgery to radiation therapy
3. `LvInvasion` — Lymphovascular invasion
4. `dx.to.surg.days` — Days from diagnosis to surgery
5. `firstcontact.to.surg.days` — Time from first contact to surgery
6. `DateFirstContact.x` — Date of first clinical contact
7. `Sex_M` — Patient sex
8. `SummaryStage` — Cancer summary stage

---

## 📁 Repository Structure
```
├── Final_project.ipynb              # Main notebook (full pipeline)
├── Project_Report.pdf               # Full academic report
├── thyroidcancerupstaging_          # Dataset (anonymized)
│   blindeddata_final.csv
├── preprocessor_rec.joblib          # Saved preprocessing pipeline
├── best_rf_rec.joblib               # Best Random Forest model
├── best_dt_rec.joblib               # Best Decision Tree model
├── best_svm_rec.joblib              # Best SVM model
└── best_log_rec.joblib              # Best Logistic Regression model
```

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/Jalalab/thyroid-cancer-recurrence-prediction.git
cd thyroid-cancer-recurrence-prediction
```

### 2. Install dependencies
```bash
pip install pandas numpy scikit-learn imbalanced-learn shap matplotlib seaborn joblib
```

### 3. Open the notebook
```bash
jupyter notebook Final_project.ipynb
```
> Or upload to **Google Colab** and run all cells.



## ⚠️ Limitations & Future Work

**Limitations:**
- Retrospective single-cohort dataset
- No external validation cohort
- Binary outcome only (no time-to-event modeling)

**Future Work:**
- Survival analysis (Cox proportional hazards)
- Deep learning time-to-event prediction
- External validation on independent cohorts
- Patient-level SHAP force plots

---

## 📚 References

1. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
2. Molnar, C. (2019). *Interpretable Machine Learning*. Lulu.com.
3. Soegaard Ballester, J., & Wachtel, H. (2022). Institutional Thyroid Cancer Dataset. *Mendeley Data*, V1. DOI: 10.17632/57726vkm48.1

---

## 🏫 Academic Context

This project was submitted in partial fulfillment of the requirements for the Bachelor of 
Science in Computer Science and Engineering at **North South University**, Dhaka, Bangladesh.



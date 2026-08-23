# IEEE SSCS AUSC — Task 9: Stacking + Optuna

**Location:** `Machine-Learning/Tasks/Task9/`

## Overview

| Part | Topic | Dataset |
| :--- | :---- | :------ |
| **1** | **Stacking** ensemble: Logistic Regression + kNN + Decision Tree | [Heart Disease UCI (Kaggle)](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data) |
| **2** | **Optuna** tuning of Decision Tree `max_depth` | [Red Wine Quality (Kaggle)](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009) |

> **AI Assistance Note:** Some notebook markdown explanations were drafted with AI assistance, then reviewed.

---

## Project Structure

```
Task9/
├── task.ipynb                 # Full notebook (stacking + Optuna)
├── heart.csv                  # UCI Heart Disease (Kaggle-equivalent)
├── winequality-red.csv        # UCI Red Wine Quality (Kaggle-equivalent)
├── plots/
│   ├── stacking_comparison.png
│   ├── stacking_confusion_matrix.png
│   ├── optuna_max_depth.png
│   └── dt_f1_vs_depth_wine.png
├── results/
│   ├── stacking_comparison.csv
│   ├── optuna_dt_max_depth_trials.csv
│   └── optuna_depth_test_eval.csv
├── requirements.txt
└── README.md
```

---

## Task 1 — Stacking

Base learners:
- `LogisticRegression`
- `KNeighborsClassifier` (k=7)
- `DecisionTreeClassifier` (max_depth=5)

Meta-learner: `LogisticRegression` via `sklearn.ensemble.StackingClassifier` (5-fold OOF `predict_proba`).

Target: binary heart disease (`target > 0`).

### Results (executed)

| Model | Accuracy | F1 |
| :---- | -------: | -: |
| Logistic Regression | 0.868 | 0.861 |
| kNN (k=7) | 0.855 | 0.845 |
| Stacking (LR+kNN+DT) | 0.842 | 0.838 |
| Decision Tree (depth=5) | 0.711 | 0.694 |

---

## Task 2 — Optuna (`max_depth`)

- Objective: maximize **5-fold CV F1** on wine train split  
- Search space: `max_depth ∈ [1, 30]`  
- Sampler: TPE · **40 trials**  
- Label: good wine if `quality >= 6`

### Results (executed)

| Metric | Value |
| :----- | ----: |
| Best `max_depth` | **15** |
| Best CV F1 | 0.750 |
| Test F1 @ best | 0.759 |
| Test Accuracy @ best | 0.735 |

---

## How to Run

```bash
cd Machine-Learning/Tasks/Task9
pip install -r requirements.txt
jupyter notebook task.ipynb
# Run all cells
```

---

## Requirements

```bash
pip install numpy pandas matplotlib scikit-learn optuna
```

---

## Author

**Abdlrhman Hisham Ismail** — IEEE SSCS AUSC, AI Team

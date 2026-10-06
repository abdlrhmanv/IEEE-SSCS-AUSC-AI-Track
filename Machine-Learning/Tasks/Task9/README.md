# Task 9 — Stacking and Optuna

[Portfolio](../../../README.md) · [Tasks](../README.md)

## Overview

| Part | Topic | Dataset |
| :--- | :---- | :------ |
| **1** | **Stacking** ensemble: Logistic Regression + kNN + Decision Tree | [Heart Disease UCI (Kaggle)](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data) |
| **2** | **Optuna** tuning of Decision Tree `max_depth` | [Red Wine Quality (Kaggle)](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009) |

> **AI Assistance Note:** Some notebook markdown explanations were drafted with AI assistance, then reviewed.

<a id="project-structure"></a>

## Contents

Folder guides: [plots](plots/README.md) · [results](results/README.md).

```text
Task9/
├── plots/
│   ├── dt_f1_vs_depth_wine.png
│   ├── optuna_max_depth.png
│   ├── stacking_comparison.png
│   └── stacking_confusion_matrix.png
├── results/
│   ├── optuna_depth_test_eval.csv
│   ├── optuna_dt_max_depth_trials.csv
│   └── stacking_comparison.csv
├── heart.csv
├── README.md
├── requirements.txt
├── task.ipynb
└── winequality-red.csv
```

<a id="how-to-run"></a>

## Run locally

```bash
# From the repository root, with your environment activated
cd Machine-Learning/Tasks/Task9
python -m pip install -r requirements.txt
python -m jupyter notebook task.ipynb
# Run all cells
```

## Workflow

### Task 1 — Stacking

Base learners:
- `LogisticRegression`
- `KNeighborsClassifier` (k=7)
- `DecisionTreeClassifier` (max_depth=5)

Meta-learner: `LogisticRegression` via `sklearn.ensemble.StackingClassifier` (5-fold OOF `predict_proba`).

Target: binary heart disease (`target > 0`).

### Task 2 — Optuna (`max_depth`)

- Objective: maximize **5-fold CV F1** on wine train split
- Search space: `max_depth ∈ [1, 30]`
- Sampler: TPE · **40 trials**
- Label: good wine if `quality >= 6`

## Results

### Stacking results

| Model | Accuracy | F1 |
| :---- | -------: | -: |
| Logistic Regression | 0.868 | 0.861 |
| kNN (k=7) | 0.855 | 0.845 |
| Stacking (LR+kNN+DT) | 0.842 | 0.838 |
| Decision Tree (depth=5) | 0.711 | 0.694 |

### Optuna results

| Metric | Value |
| :----- | ----: |
| Best `max_depth` | **15** |
| Best CV F1 | 0.750 |
| Test F1 @ best | 0.759 |
| Test Accuracy @ best | 0.735 |

## Requirements

Install the packages in [requirements.txt](requirements.txt), including Jupyter for notebook tasks. Create and activate a virtual environment first; see the [root setup notes](../../../README.md#getting-started).

<a id="course-materials"></a>

## Resources

Assignment and reference resources: [Task 9 materials](../../../course-materials/shared/ml-nlp-session.pdf). See the [course-materials index](../../../course-materials/README.md) for all weeks.

## Author

**Abdlrhman Hisham Ismail** — IEEE SSCS AUSC AI Team

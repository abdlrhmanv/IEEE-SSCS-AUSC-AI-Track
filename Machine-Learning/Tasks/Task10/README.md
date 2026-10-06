# Task 10 — Classification (Large Task + Research)

**Location:** `Machine-Learning/Tasks/Task10/`

IEEE SSCS AUSC · AI Sub-Team  
**Author:** Abdlrhman Hisham Ismail (AI2617)

Large classification project covering Decision Trees, SVMs (linear & RBF), Random Forest with Optuna, and a research report on SVM kernels.

**Shared dataset:** [Red Wine Quality (Kaggle / UCI)](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009)  
Binary label: **good** if `quality >= 6`.

---

## Folder Structure

Part of the main course repo: `IEEE-SSCS-AUSC-AI-Track`.

```text
Task10/
├── data/
│   ├── heart.csv
│   └── winequality-red.csv
├── Task 0 - Decision Tree/
│   ├── plots/
│   │   ├── test_f1_vs_max_depth.png
│   │   └── train_f1_vs_max_depth.png
│   ├── decision_tree.ipynb
│   └── results_f1_vs_depth.csv
├── Task 1 - SVM linear/
│   ├── plots/
│   │   └── svm_linear_confusion.png
│   ├── results_svm_linear.csv
│   └── svm_linear.ipynb
├── Task 2 - SVM RBF/
│   ├── plots/
│   │   └── svm_rbf_confusion.png
│   ├── results_svm_rbf.csv
│   └── svm_rbf.ipynb
├── Task 3 - Randomforest/
│   ├── plots/
│   │   ├── optuna_rf_history_importance.png
│   │   └── rf_confusion.png
│   ├── results/
│   │   ├── best_rf_params.csv
│   │   └── optuna_trials.csv
│   └── random_forest_optuna.ipynb
├── Task 4 - Research/
│   └── SVM_Kernels_Report.pdf
├── .gitignore
├── data_utils.py
├── README.md
└── requirements.txt
```

---

## Task Overview

| Folder | Deliverable |
| :----- | :---------- |
| **Task 0 — Decision Tree** | Train DT; plot **train F1** and **test F1** vs `max_depth` ∈ **[3, 25]** (two separate graphs) |
| **Task 1 — SVM linear** | Train `SVC(kernel="linear")` |
| **Task 2 — SVM RBF** | Train `SVC(kernel="rbf")` |
| **Task 3 — Randomforest** | Optuna search for best RF hyperparameter **combination** |
| **Task 4 — Research** | **5-page** report on types of kernels in SVM |

---

## Key Results (executed)

### Decision Tree
- Best test F1 near shallow depths (~0.75 at depth 3); train F1 rises with depth (overfitting trend).
- Plots: `Task 0 - Decision Tree/plots/train_f1_vs_max_depth.png`, `test_f1_vs_max_depth.png`

### SVM
| Kernel | Test Acc | Test F1 | # Support Vectors |
| :----- | -------: | ------: | ----------------: |
| Linear | 0.738 | 0.748 | 717 |
| RBF | 0.755 | 0.764 | 732 |

### Random Forest (Optuna, 50 trials)
- Best CV F1 ≈ **0.822**
- Example best combo: `max_depth=30`, `n_estimators=324`, `min_samples_split=8`, `min_samples_leaf=1`, `max_features=None`, `criterion=entropy`
- Test F1 ≈ **0.819** · Test Acc ≈ **0.805**

### Research
- `Task 4 - Research/SVM_Kernels_Report.pdf` — **5 pages** (linear, polynomial, RBF, sigmoid, comparison & references)

---

## How to Run

```bash
# From the repository root, with your environment activated
cd Machine-Learning/Tasks/Task10
python -m pip install -r requirements.txt

# Choose one notebook; its kernel working directory must be its own child folder
python -m jupyter notebook "Task 0 - Decision Tree/decision_tree.ipynb"
python -m jupyter notebook "Task 1 - SVM linear/svm_linear.ipynb"
python -m jupyter notebook "Task 2 - SVM RBF/svm_rbf.ipynb"
python -m jupyter notebook "Task 3 - Randomforest/random_forest_optuna.ipynb"
```

For example, `decision_tree.ipynb` must run with its kernel directory set to `Task 0 - Decision Tree/`, so the first cell can import the parent `data_utils.py`. Jupyter normally starts kernels in the notebook folder; configure the same directory when using an editor.

---

## Requirements

Install the packages in [requirements.txt](requirements.txt), including Jupyter for notebook tasks. Use the [coursework setup guide](../../../docs/running-coursework.md) to create an environment first.

---

> **AI Assistance Note:** Notebook explanations and the research report draft were prepared with AI assistance, then reviewed for this submission.

# IEEE SSCS AUSC — Task 8: ML Task 2 (Level 1)

**Location:** `Machine-Learning/Tasks/Task8/`

## Overview

Hyperparameter experiments on the **Mushroom Classification** dataset (edible vs poisonous):

| Part | Model | Sweep | Metric |
| :--- | :---- | :---- | :----- |
| **1** | KNN (`sklearn`) | K = 2 … 150 | Train / test **F1** |
| **2** | Logistic Regression via `SGDClassifier(log_loss)` | η = 0.001 … 1.000 (step 0.001) | Train / test **F1** |
| **3** | `LogisticRegression` | `max_iter` = 1,000 … 100,000 (step 1,000) | Train / test **F1** |

> **AI Assistance Note:** Some notebook markdown explanations were drafted with AI assistance, then reviewed.

---

## Project Structure

```
Task8/
├── task.ipynb                      # Full workflow + plots + analysis
├── mushrooms.csv                   # UCI mushroom dataset (Kaggle-equivalent)
├── mushrooms_raw.data              # Original UCI file
├── plots/
│   ├── knn_f1_vs_k.png
│   ├── logreg_f1_vs_learning_rate.png
│   └── logreg_f1_vs_iterations.png
├── results/                        # CSV tables of every sweep point
├── ML_2_IEEE_Level_1.pdf           # Assignment sheet
└── README.md
```

Dataset source: [Kaggle — Mushroom Classification](https://www.kaggle.com/datasets/uciml/mushroom-classification) (same as UCI Agaricus–Lepiota).

---

## Preprocessing

1. Load CSV; treat stalk-root `?` as category `"missing"`
2. Label-encode target + all categorical features
3. Stratified **80/20** train/test split
4. `StandardScaler` inside each `Pipeline` before the classifier

---

## Key Results (executed notebook)

| Experiment | Best setting | Train F1 | Test F1 |
| :--------- | :----------- | -------: | ------: |
| KNN | **K = 2** | 1.000 | 1.000 |
| Learning rate | **η ≈ 0.754** | 0.958 | 0.969 |
| Iterations | **max_iter = 1,000** | 0.960 | 0.961 |

Notes:
- Mushrooms is a highly separable dataset → very small K still generalizes.
- LR F1 plateaus by 1k iterations; larger `max_iter` adds cost only.
- Task 2 uses SGD logistic regression because classic GD **learning rate** is not a `LogisticRegression` hyperparameter in scikit-learn.

---

## How to Run

```bash
cd Machine-Learning/Tasks/Task8
jupyter notebook task.ipynb
# Run all cells top to bottom (~1 minute)
```

---

## Requirements

```bash
pip install numpy pandas matplotlib scikit-learn
```

---

## Author

**Abdlrhman Hisham Ismail** — IEEE SSCS AUSC, AI Team

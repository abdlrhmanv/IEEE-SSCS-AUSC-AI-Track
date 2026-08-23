# IEEE SSCS AUSC — Task 6: Polynomial Regression

**Location:** `Machine-Learning/Tasks/Task6/`

## Overview

This task for the **IEEE SSCS AUSC AI Team** focuses on **polynomial regression** with scikit-learn using the **UCI Auto MPG** dataset (predict `mpg`), plus a small synthetic quadratic check.

Implemented workflow:
- Linear regression **baseline**
- **PolynomialFeatures** at degrees 2–5 with train/validation curves
- **Ridge** and **Lasso** with `GridSearchCV` alpha tuning
- Full metric suite: **MSE, MAE, RMSE, MAPE, R²** (+ Adjusted R² where defined)
- Residual diagnostics and written analysis

> **AI Assistance Note:** Some notebook markdown explanations and documentation comments were generated with AI assistance, then reviewed in the final workflow.

---

## Project Structure

```
Task6/
├── task.ipynb                          # Main notebook (full workflow + outputs)
├── Python Code/
│   └── task.ipynb                      # Same notebook (submission layout)
├── Written Report/
│   ├── Polynomial_Regression_Report.pdf
│   └── report.html                     # Source used to generate the PDF
├── auto-mpg.data                       # Local copy of UCI Auto MPG
├── poly_regression_task.pdf            # Assignment sheet
├── poly_regression_study_plan.pdf      # Study plan / resources
└── README.md
```

---

## Notebook Workflow

1. **Load & explore** Auto MPG (missing horsepower, MPG distribution, feature–target plots)
2. **Split** 70% train / 15% validation / 15% test
3. Fit **linear baseline**; report all five metrics
4. Fit **polynomial models** (deg 2–5); plot train vs validation RMSE & R²
5. Tune **Ridge / Lasso** at the best degree with cross-validated `alpha`
6. Build a **test-set comparison table** across all models
7. **Residual analysis** for the best model
8. Write **insights** + optional synthetic quadratic experiment

---

## Key Results (executed notebook)

| Model | Test RMSE | Test R² |
| :---- | --------: | ------: |
| Lasso (deg 2) | **2.73** | **0.849** |
| Ridge (deg 2) | 2.77 | 0.845 |
| Poly deg 2 | 2.82 | 0.839 |
| Linear baseline | 3.02 | 0.816 |

Best unregularized degree by validation RMSE: **2**. Degrees 4–5 overfit severely.

---

## How to Run

```bash
cd Machine-Learning/Tasks/Task6
jupyter notebook task.ipynb
# Run all cells top to bottom
```

Requires network access only if `auto-mpg.data` is missing (notebook falls back to the UCI URL).

---

## Requirements

- Python 3.x
- NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## Author

**Abdlrhman Hisham Ismail** — IEEE SSCS AUSC, AI Team

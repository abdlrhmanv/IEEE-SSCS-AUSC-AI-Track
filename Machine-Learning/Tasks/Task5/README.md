# IEEE SSCS AUSC — Task 5: Regression (California Housing)

**Location:** `Machine-Learning/Tasks/Task5/`

## 📋 Overview

This task for the **IEEE SSCS AUSC AI Team** focuses on **supervised regression** using the California Housing dataset.  
The objective is to predict **`median_house_value`** and compare multiple linear regression implementations.

Implemented models:
- **From scratch — Normal Equation** (closed-form solution)
- **From scratch — Gradient Descent** (iterative optimization)
- **Scikit-learn — `LinearRegression`**

> **AI Assistance Note:** Some notebook markdown explanations and documentation comments were generated with AI assistance, then reviewed in the final workflow.

---

## 📂 Project Structure

```text
Task5/
├── housing.csv
├── README.md
├── requirements.txt
└── task.ipynb
```

---

## 🔍 Notebook Workflow

The notebook follows the assignment requirements step-by-step:

1. **Load and inspect** California Housing data
2. **Preprocess** (handle missing values, one-hot encode categorical feature)
3. **Split** data into **70% train / 15% validation / 15% test**
4. **Scale features** (for stable Gradient Descent)
5. Implement **MSE** and **MAE** **from scratch**
6. Train **Normal Equation** model from scratch
7. Train **Gradient Descent** model from scratch and tune learning rate on validation set
8. Train **Scikit-learn LinearRegression**
9. Compare all models on **Train / Validation / Test** using custom MSE/MAE
10. Add markdown discussion and final comparison summary

---

## ▶️ How to Run

```bash
# From the repository root, with your environment activated
cd Machine-Learning/Tasks/Task5
python -m pip install -r requirements.txt
python -m jupyter notebook task.ipynb
# Run all cells top to bottom
```

---

## 🛠️ Requirements

Install the packages in [requirements.txt](requirements.txt), including Jupyter for notebook tasks. Create and activate a virtual environment first; see the [root setup notes](../../../README.md#getting-started).

---

## 👤 Author

**Abdlrhman** — IEEE SSCS AUSC, AI Team


## Course materials

Assignment and reference resources: [Task 5 materials](../../../course-materials/Machine-Learning/Task5). See the [course-materials index](../../../course-materials/README.md) for all weeks.

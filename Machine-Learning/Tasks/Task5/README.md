# Task 5 — California Housing regression

[Portfolio](../../../README.md) · [Tasks](../README.md)

## Overview

This task for the **IEEE SSCS AUSC AI Team** focuses on **supervised regression** using the California Housing dataset.
The objective is to predict **`median_house_value`** and compare multiple linear regression implementations.

Implemented models:
- **From scratch — Normal Equation** (closed-form solution)
- **From scratch — Gradient Descent** (iterative optimization)
- **Scikit-learn — `LinearRegression`**

> **AI Assistance Note:** Some notebook markdown explanations and documentation comments were generated with AI assistance, then reviewed in the final workflow.

<a id="project-structure"></a>

## Contents

```text
Task5/
├── housing.csv
├── README.md
├── requirements.txt
└── task.ipynb
```

<a id="how-to-run"></a>

## Run locally

```bash
# From the repository root, with your environment activated
cd Machine-Learning/Tasks/Task5
python -m pip install -r requirements.txt
python -m jupyter notebook task.ipynb
# Run all cells top to bottom
```

<a id="notebook-workflow"></a>

## Workflow

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

## Requirements

Install the packages in [requirements.txt](requirements.txt), including Jupyter for notebook tasks. Create and activate a virtual environment first; see the [root setup notes](../../../README.md#getting-started).

<a id="course-materials"></a>

## Resources

Assignment and reference resources: [Task 5 materials](../../../course-materials/Machine-Learning/Task5). See the [course-materials index](../../../course-materials/README.md) for all weeks.

## Author

**Abdlrhman Hisham Ismail** — IEEE SSCS AUSC AI Team

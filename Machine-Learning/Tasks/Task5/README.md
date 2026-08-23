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

```
Task5/
├── task.ipynb                                      # Main notebook (full regression workflow)
├── housing.csv                                     # California housing dataset
├── Level-1_Task-5.pdf                              # Level 1 task sheet
├── Machine Learning - sheet 1 - AI Team - IEEE.pdf
├── Material/
│   ├── Session5.pdf
│   └── *.jpeg                                      # Supporting material/screenshots
└── README.md
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
cd Machine-Learning/Tasks/Task5
jupyter notebook task.ipynb
# Run all cells top to bottom
```

---

## 🛠️ Requirements

- **Python 3.x**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Scikit-learn**

Install dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn
```

---

## 👤 Author

**Abdlrhman** — IEEE SSCS AUSC, AI Team


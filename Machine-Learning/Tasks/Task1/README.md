# IEEE SSCS AUSC — Task 1: NumPy Assignment

**Location:** `Machine-Learning/Tasks/Task1/`

## 📋 Overview

This task for the **IEEE SSCS AUSC AI Team** focuses on **NumPy**. It consists of three parts:

1. **Core NumPy Operations** — Array creation, shape, axis-based statistics, indexing, broadcasting, normalization, and flattening.
2. **Linear Regression from Scratch** — Implementing the **Normal Equation** (θ = (XᵀX)⁻¹Xᵀy) with NumPy to predict house prices.
3. **Research** — A written research document in PDF format.

---

## 📂 Project Structure

```text
Task1/
├── NumPy Assignment/
│   ├── Code/
│   │   ├── Core NumPy Operations/
│   │   │   └── main.py
│   │   └── Linear Regression from Scratch (Normal Equation)/
│   │       └── main.py
│   └── Research/
│       └── The Normal Equation, and Handling Multicollinearity.pdf
├── README.md
└── requirements.txt
```

---

## 🧮 Part 1 — Core NumPy Operations

NumPy-based operations on a grades matrix (students × subjects):

| Step | Description |
| ---- | ----------- |
| Convert list to array, print shape | `np.array`, `.shape` |
| Mean per student | `mean(axis=1)` |
| Mean per subject | `mean(axis=0)` |
| Filter students | Boolean indexing (e.g. mean > 85) |
| Add bonus | Broadcasting (`grades + 5`) |
| Min–Max normalization | Per-column normalize to [0, 1] |
| Flatten | `.flatten()` |

### ▶️ How to Run

```bash
# From the repository root, with your environment activated
python -m pip install -r Machine-Learning/Tasks/Task1/requirements.txt
cd Machine-Learning/Tasks/Task1/"NumPy Assignment/Code/Core NumPy Operations"
python main.py
```

---

## 📈 Part 2 — Linear Regression (Normal Equation)

Predict house price from size (square meters) using the closed-form solution:

- Build design matrix **X** with a bias column (`np.hstack`, `np.ones`).
- Compute **θ** = (XᵀX)⁻¹Xᵀy with `np.linalg.inv` and matrix multiplication.
- Predict price for a new house size (e.g. 90 m²).

### ▶️ How to Run

```bash
# From the repository root, with your environment activated
python -m pip install -r Machine-Learning/Tasks/Task1/requirements.txt
cd Machine-Learning/Tasks/Task1/"NumPy Assignment/Code/Linear Regression from Scratch (Normal Equation)"
python main.py
```

---

## 📝 Part 3 — Research

Research document:

| Document | Location |
| -------- | -------- |
| **The Normal Equation, and Handling Multicollinearity.pdf** | `NumPy Assignment/Research/` |

---

## 🛠️ Requirements

Install the packages in [requirements.txt](requirements.txt), including Jupyter for notebook tasks. Create and activate a virtual environment first; see the [root setup notes](../../../README.md#getting-started).

---

## 👤 Author

**Abdlrhman** — IEEE SSCS AUSC, AI Team

## Course materials

Assignment and reference resources: [Task 1 materials](../../../course-materials/Machine-Learning/Task1). See the [course-materials index](../../../course-materials/README.md) for all weeks.

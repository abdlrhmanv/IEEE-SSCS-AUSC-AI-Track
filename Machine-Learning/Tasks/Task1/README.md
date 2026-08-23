# IEEE SSCS AUSC — Task 1: NumPy Assignment

**Location:** `Machine-Learning/Tasks/Task1/`

## 📋 Overview

This task for the **IEEE SSCS AUSC AI Team** focuses on **NumPy**. It consists of three parts:

1. **Core NumPy Operations** — Array creation, shape, axis-based statistics, indexing, broadcasting, normalization, and flattening.
2. **Linear Regression from Scratch** — Implementing the **Normal Equation** (θ = (XᵀX)⁻¹Xᵀy) with NumPy to predict house prices.
3. **Research** — A written research document in PDF format.

---

## 📂 Project Structure

```
Task1/
├── NumPy Assignment/
│   ├── Code/
│   │   ├── Core NumPy Operations/
│   │   │   └── main.py          # Grades array: means, filtering, bonus, normalization, flatten
│   │   └── Linear Regression from Scratch (Normal Equation)/
│   │       └── main.py          # House price prediction via normal equation
│   └── Research/
│       └── Research.pdf
└── README.md
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
cd Machine-Learning/Tasks/Task1/"NumPy Assignment/Code/Linear Regression from Scratch (Normal Equation)"
python main.py
```

---

## 📝 Part 3 — Research

Research document:

| Document | Location |
| -------- | -------- |
| **Research.pdf** | `NumPy Assignment/Research/` |

---

## 🛠️ Requirements

- **Python 3.x**
- **NumPy** (`pip install numpy`)

---

## 👤 Author

**Abdlrhman** — IEEE SSCS AUSC, AI Team

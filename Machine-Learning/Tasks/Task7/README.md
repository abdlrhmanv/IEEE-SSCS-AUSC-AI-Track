# Task 7 — Logistic regression from scratch

[Portfolio](../../../README.md) · [Tasks](../README.md)

## Overview

This folder covers **binary classification / logistic regression** for the IEEE SSCS AUSC AI Team (Level 1).

| Sub-task | Difficulty | Deliverable |
| :------- | :--------- | :---------- |
| **1** | Easy | Handwritten derivation of **∂J/∂wⱼ** for Binary Cross-Entropy |
| **2** | Easy | NumPy **sigmoid** + labeled plot |
| **3** | Medium | Logistic Regression **from scratch** (`classification.py` + `main.py`) |
| **4** | Medium | Train/test on the **4D XOR** truth table |

> **AI Assistance Note:** Some documentation comments and the typed BCE derivation were drafted with AI assistance, then reviewed. The assignment requests a handwritten derivation; the PDF here is a typed reference, not evidence of a handwritten submission.

<a id="project-structure"></a>

## Contents

Folder guides: [Task1_BCE_Derivation](Task1_BCE_Derivation/README.md) · [Task2_Sigmoid](Task2_Sigmoid/README.md) · [plots](plots/README.md).

```text
Task7/
├── plots/
│   ├── and_training_loss.png
│   └── xor_training_loss.png
├── Task1_BCE_Derivation/
│   └── BCE_Gradient_Derivation.pdf
├── Task2_Sigmoid/
│   ├── sigmoid_plot.png
│   └── sigmoid_plot.py
├── classification.py
├── main.py
├── README.md
└── requirements.txt
```

Required modular layout from the assignment:

```
main.py
classification.py   # contains LogisticRegression class
```

## Run locally

### Sigmoid Plot

```bash
# From the repository root, with your environment activated
python -m pip install -r Machine-Learning/Tasks/Task7/requirements.txt
cd Machine-Learning/Tasks/Task7
python3 Task2_Sigmoid/sigmoid_plot.py
```

### Logistic Regression + 4D XOR

```bash
# From the repository root, with your environment activated
python -m pip install -r Machine-Learning/Tasks/Task7/requirements.txt
cd Machine-Learning/Tasks/Task7
python3 main.py
```

## Workflow

### Task 1 — BCE Gradient (handwritten)

Final result used in code:

$$
\frac{\partial J}{\partial w_j}
= \frac{1}{m}\sum_{i=1}^{m}\big(\hat{y}^{(i)} - y^{(i)}\big)\,x_j^{(i)}
\quad\Rightarrow\quad
\nabla_w J = \tfrac{1}{m}\,X^\top(\hat{y}-y)
$$

See `Task1_BCE_Derivation/BCE_Gradient_Derivation.pdf` for the typed derivation. The original brief specifies a handwritten submission.

### Task 2 — Sigmoid Plot

Produces `Task2_Sigmoid/sigmoid_plot.png`.

### Tasks 3 & 4 — Logistic Regression + 4D XOR

`classification.py` implements (NumPy only):
- `sigmoid`
- `LogisticRegression(iterations, lr)`
- `fit`, `predict`, `predict_proba`, `evaluate`

`main.py` loads the required **4D XOR** table and reports predictions + metrics.
It also runs a **4D AND** sanity check (linearly separable) to show the same code can learn when a linear boundary exists.

#### Expected XOR behavior

4-bit parity (**XOR**) is **not linearly separable**. A single logistic unit therefore stays near chance (~50% accuracy, weights ≈ 0). That is a correct outcome for this model class — not a bug.

## Requirements

Install the packages in [requirements.txt](requirements.txt). Create and activate a virtual environment first; see the [root setup notes](../../../README.md#getting-started).

<a id="course-materials"></a>

## Resources

Assignment and reference resources: [Task 7 materials](../../../course-materials/Machine-Learning/Task7). See the [course-materials index](../../../course-materials/README.md) for all weeks.

## Author

**Abdlrhman Hisham Ismail** — IEEE SSCS AUSC AI Team

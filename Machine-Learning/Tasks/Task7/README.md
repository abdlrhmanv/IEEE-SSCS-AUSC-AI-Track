# IEEE SSCS AUSC — Task 7: Classification (ML Task 1 · Level 1)

**Location:** `Machine-Learning/Tasks/Task7/`

## Overview

This folder covers **binary classification / logistic regression** for the IEEE SSCS AUSC AI Team (Level 1).

| Sub-task | Difficulty | Deliverable |
| :------- | :--------- | :---------- |
| **1** | Easy | Handwritten derivation of **∂J/∂wⱼ** for Binary Cross-Entropy |
| **2** | Easy | NumPy **sigmoid** + labeled plot |
| **3** | Medium | Logistic Regression **from scratch** (`classification.py` + `main.py`) |
| **4** | Medium | Train/test on the **4D XOR** truth table |

> **AI Assistance Note:** Some documentation comments and the typed BCE derivation were drafted with AI assistance, then reviewed. Task 1 must still be **copied by hand onto paper** for submission.

---

## Project Structure

```
Task7/
├── classification.py                 # LogisticRegression class (fit / predict / evaluate)
├── main.py                           # Train & evaluate on 4D XOR (+ AND sanity check)
├── plots/
│   ├── xor_training_loss.png
│   └── and_training_loss.png
├── Task1_BCE_Derivation/
│   ├── BCE_Gradient_Derivation.pdf   # Typed reference — rewrite by hand
│   └── BCE_Gradient_Derivation.html
├── Task2_Sigmoid/
│   ├── sigmoid_plot.py
│   └── sigmoid_plot.png
├── ML_1_IEEE_Level_1.pdf             # Assignment sheet
├── Classification_IEEE_part 1.pdf    # Lecture / study material
└── README.md
```

Required modular layout from the assignment:

```
main.py
classification.py   # contains LogisticRegression class
```

---

## Task 1 — BCE Gradient (handwritten)

Final result used in code:

\[
\frac{\partial J}{\partial w_j}
= \frac{1}{m}\sum_{i=1}^{m}\big(\hat{y}^{(i)} - y^{(i)}\big)\,x_j^{(i)}
\quad\Rightarrow\quad
\nabla_w J = \tfrac{1}{m}\,X^\top(\hat{y}-y)
\]

See `Task1_BCE_Derivation/BCE_Gradient_Derivation.pdf` for every intermediate step, then **rewrite it by hand**.

---

## Task 2 — Sigmoid Plot

```bash
cd Machine-Learning/Tasks/Task7
python3 Task2_Sigmoid/sigmoid_plot.py
```

Produces `Task2_Sigmoid/sigmoid_plot.png`.

---

## Tasks 3 & 4 — Logistic Regression + 4D XOR

```bash
cd Machine-Learning/Tasks/Task7
python3 main.py
```

`classification.py` implements (NumPy only):
- `sigmoid`
- `LogisticRegression(iterations, lr)`
- `fit`, `predict`, `predict_proba`, `evaluate`

`main.py` loads the required **4D XOR** table and reports predictions + metrics.  
It also runs a **4D AND** sanity check (linearly separable) to show the same code can learn when a linear boundary exists.

### Expected XOR behavior

4-bit parity (**XOR**) is **not linearly separable**. A single logistic unit therefore stays near chance (~50% accuracy, weights ≈ 0). That is a correct outcome for this model class — not a bug.

---

## Requirements

- Python 3.x
- NumPy
- Matplotlib (plots only)

```bash
pip install numpy matplotlib
```

---

## Author

**Abdlrhman Hisham Ismail** — IEEE SSCS AUSC, AI Team

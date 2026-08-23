# Phase 1 — Machine Learning

This folder is the **Machine Learning** phase of the IEEE SSCS AUSC AI track: weekly tasks, then two larger projects.

Tasks stay **independent submissions**. They are not one application.

## Layout

```text
Machine-Learning/
├── Tasks/
│   ├── Task0/ … Task10/
├── Project1/          # portfolio stub — Uber price prediction (team repo)
└── Project2/          # portfolio stub — Smart Home voice control (standalone repo)
```

## Tasks 0–10

| Task | Topic |
| :--: | :---- |
| 0 | Matrix operations from scratch + research notes |
| 1 | NumPy + linear regression (normal equation) |
| 2 | Pandas EDA (Titanic) |
| 3 | Matplotlib & Seaborn |
| 4 | Zara sales EDA |
| 5 | California Housing regression (from scratch + sklearn) |
| 6 | Polynomial regression (Auto MPG) |
| 7 | Logistic regression from scratch (4D XOR) |
| 8 | KNN / logistic hyperparameter sweeps (mushrooms) |
| 9 | Stacking + Optuna (`max_depth`) |
| 10 | Decision tree, SVM, random forest + SVM kernels report |

Each task has its own README and install notes. There is **no** shared `requirements.txt` for this phase.

From the repository root:

```bash
cd Machine-Learning/Tasks/TaskN
```

## Project 1 — Uber price prediction

Team ML project (Streamlit). The **source lives in a separate GitHub repository**. This portfolio only keeps a short write-up and the assignment PDF.

See [`Project1/README.md`](Project1/README.md).

## Project 2 — Smart Home voice control

Production-style voice control (Streamlit + Arduino + speaker/command models). Maintained as a **standalone repository**. This portfolio only links to it.

See [`Project2/README.md`](Project2/README.md).

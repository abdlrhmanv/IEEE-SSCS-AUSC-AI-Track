<h1 align="center">IEEE SSCS AUSC — AI Track Portfolio</h1>

<p align="center">
  <img src="assets/ieee-logo.jpeg" alt="IEEE SSCS AUSC logo" width="140">
</p>

<p align="center">
  <em>Coursework and projects from the IEEE Solid-State Circuits Society<br/>
  Alexandria University Student Chapter AI track</em>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white" alt="Python"></a>
  <a href="https://numpy.org/"><img src="https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white" alt="NumPy"></a>
  <a href="https://pandas.pydata.org/"><img src="https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white" alt="Pandas"></a>
  <a href="https://matplotlib.org/"><img src="https://img.shields.io/badge/Matplotlib-11557C?logo=plotly&logoColor=white" alt="Matplotlib"></a>
  <a href="https://scikit-learn.org/"><img src="https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikitlearn&logoColor=white" alt="Scikit-Learn"></a>
</p>

---

## About

I'm **Abdlrhman Ismail**, a Senior Computer Engineering student at **Ain Shams University (ASU)** and a member of the **AI Committee** at IEEE SSCS AUSC.

This repository is a **learning portfolio**, not a single application. It follows two chronological phases:

1. **Machine Learning** — weekly tasks, then Project 1 and Project 2  
2. **Natural Language Processing** — NLP team work after the ML phase

---

## Repository structure

```text
IEEE-SSCS-AUSC-AI-Track/
├── assets/ieee-logo.jpeg
├── Machine-Learning/
│   ├── Tasks/Task0 … Task10
│   ├── Project1/          # stub → team Uber price-prediction repo
│   └── Project2/          # stub → Smart Home voice-control repo
└── NLP/
    └── Week1/
        ├── Task0/         # vector rotation notebook
        └── Project1/      # Neurova NLP
```

Details: [`Machine-Learning/README.md`](Machine-Learning/README.md) · [`NLP/README.md`](NLP/README.md) · [Neurova NLP](NLP/Week1/Project1)

---

## Phase 1 — Machine Learning

|  #  | Item | Description | Status |
| :-: | :--- | :---------- | :----: |
| 00 | [Task 0](Machine-Learning/Tasks/Task0) | Matrix operations from scratch + research | Done |
| 01 | [Task 1](Machine-Learning/Tasks/Task1) | NumPy + linear regression (normal equation) | Done |
| 02 | [Task 2](Machine-Learning/Tasks/Task2) | Titanic EDA (Pandas) | Done |
| 03 | [Task 3](Machine-Learning/Tasks/Task3) | Matplotlib & Seaborn | Done |
| 04 | [Task 4](Machine-Learning/Tasks/Task4) | Zara sales EDA | Done |
| 05 | [Task 5](Machine-Learning/Tasks/Task5) | California Housing regression | Done |
| 06 | [Task 6](Machine-Learning/Tasks/Task6) | Polynomial regression (Auto MPG) | Done |
| 07 | [Task 7](Machine-Learning/Tasks/Task7) | Logistic regression from scratch | Done |
| 08 | [Task 8](Machine-Learning/Tasks/Task8) | KNN / logistic hyperparameter sweeps | Done |
| 09 | [Task 9](Machine-Learning/Tasks/Task9) | Stacking + Optuna | Done |
| 10 | [Task 10](Machine-Learning/Tasks/Task10) | DT / SVM / RF + SVM kernels report | Done |
| P1 | [Project 1](Machine-Learning/Project1) | Uber price prediction (team repo) | Separate repo |
| P2 | [Project 2](Machine-Learning/Project2) | Smart Home voice control | Separate repo |

---

## Phase 2 — Natural Language Processing

| Item | Description | Status |
| :--- | :---------- | :----: |
| [Task 0](NLP/Week1/Task0) | Vector rotation notebook | In progress |
| [Week 1](NLP/Week1) | Lecture notebooks + [Neurova Project 1](NLP/Week1/Project1) | In progress |

---

## Getting started

There is **no** root `requirements.txt`. Each task or project has its own dependencies.

```bash
git clone https://github.com/abdlrhmanv/IEEE-SSCS-AUSC-AI-Track.git
cd IEEE-SSCS-AUSC-AI-Track

python3 -m venv .venv
source .venv/bin/activate

# Example: Task 10
pip install -r Machine-Learning/Tasks/Task10/requirements.txt

# Neurova NLP
# cd NLP/Week1/Project1 && pip install -r requirements.txt && streamlit run app.py
```

Task 0 (ML) uses only the Python standard library.

Smart Home (Project 2) is **not** vendored here. Clone and install from [smart-home-voice-control](https://github.com/abdlrhmanv/smart-home-voice-control).

---

## AI assistance

Some notebook markdown explanations and documentation comments were drafted with AI assistance, then reviewed for submission.

---

## Contact

| | |
| --- | --- |
| Email | [abdlrhmanv@icloud.com](mailto:abdlrhmanv@icloud.com) |
| LinkedIn | [Abdlrhman Ismail](https://www.linkedin.com/in/abdlrhmanv) |
| GitHub | [@abdlrhmanv](https://github.com/abdlrhmanv) |

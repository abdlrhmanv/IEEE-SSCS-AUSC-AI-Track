# IEEE SSCS AUSC — Task 2: Pandas — Titanic EDA

**Location:** `Machine-Learning/Tasks/Task2/`

## 📋 Overview

This task for the **IEEE SSCS AUSC AI Team** focuses on **Pandas** and exploratory data analysis. Using the classic Titanic dataset, we walk through the full data-analysis pipeline:

1. **Load & Inspect** — Read the CSV, examine structure, types, and basic statistics.
2. **Data Cleaning** — Handle missing values (`Age`, `Embarked`, `Cabin`) and check for duplicates.
3. **Exploratory Data Analysis** — Visualize distributions (Age, Fare, Class, Gender) and compute survival rates across groups.
4. **Analysis & Insights** — Ask and answer four focused questions with code, plots, and written interpretations.
5. **Conclusion** — Summarize key findings, limitations, and potential next steps.

---

## 📂 Project Structure

```text
Task2/
├── Python Code/
│   ├── DataSet/
│   │   ├── coulms.png
│   │   └── titanic.csv
│   └── task.ipynb
├── Written Report/
│   └── Titanic Dataset – Exploratory Data Analysis.pdf
├── README.md
└── requirements.txt
```

---

## 🔍 Notebook Walkthrough

### Step 1 — Load the Data

Import `pandas`, `numpy`, and `matplotlib`. Load the CSV and inspect with `head()`, `info()`, and `describe()`.

### Step 2 — Data Cleaning

| Column     | Issue                   | Action                          |
| ---------- | ----------------------- | ------------------------------- |
| `Age`      | 177 missing values      | Imputed with **median** age     |
| `Cabin`    | 687 missing (77% null)  | **Dropped** entirely            |
| `Embarked` | 2 missing values        | Filled with **mode** (most frequent port) |

Verified: zero remaining nulls, zero duplicate rows.

### Step 3 — Exploratory Data Analysis

- Age & Fare distributions (histograms with mean/median lines)
- Passenger Class & Gender counts (bar charts)
- Overall survival rate (~38%)
- Survival rate grouped by Gender, Class, and Embarkation Port

### Step 4 — Analysis & Insights

| # | Question | Key Finding |
|---|----------|-------------|
| Q1 | Did women survive more than men? | Women ~74% vs men ~19% |
| Q2 | Survival by passenger class? | 1st ~63%, 2nd ~47%, 3rd ~24% |
| Q3 | Age groups and survival? | Children (0–12) had the highest survival rate |
| Q4 | Higher fare → higher survival? | Yes — fare is a proxy for class |

Also includes a Gender × Class cross-tabulation showing women in 1st/2nd class had near-perfect survival.

### Step 5 — Conclusion

Key takeaways: gender and class were the dominant survival factors, consistent with the "women and children first" policy. Limitations include simple median imputation for Age and dropping the Cabin column.

---

## ▶️ How to Run

```bash
# From the repository root, with your environment activated
cd Machine-Learning/Tasks/Task2/"Python Code"
python -m pip install -r ../requirements.txt
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

Assignment and reference resources: [Task 2 materials](../../../course-materials/Machine-Learning/Task2). See the [course-materials index](../../../course-materials/README.md) for all weeks.

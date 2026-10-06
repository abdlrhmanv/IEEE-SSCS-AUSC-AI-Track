# IEEE SSCS AUSC — Task 3: Data Visualization (Matplotlib & Seaborn)

**Location:** `Machine-Learning/Tasks/Task3/`

## 📋 Overview

This task for the **IEEE SSCS AUSC AI Team** focuses on **data visualization** using **Matplotlib** and **Seaborn**. The goal is to build meaningful charts, explore patterns across real datasets, and communicate insights clearly.

Datasets used (via `seaborn.load_dataset`; network access is needed on the first run unless cached):
- **Iris** — sepal & petal measurements across 3 species
- **Tips** — restaurant bill & tip data
- **Flights** — monthly passenger counts (1949–1960)

---

## 📂 Project Structure

```text
Task3/
├── Python Code/
│   └── task.ipynb
├── Written Report/
│   └── Data Visualization Report using Matplotlib and Seaborn.pdf
├── Data_Visualization_Matplotlib_Seaborn.ipynb
├── README.md
└── requirements.txt
```

---

## 🔍 Notebook Sections (Steps 1–6)

- **Step 1 — Line & Scatter Plots**: basic relationships and trends (Matplotlib + Seaborn)
- **Step 2 — Styling & Customization**: themes, palettes, grids, fonts, layout
- **Step 3 — Distribution Plots**: histogram + KDE, grouped distributions, ECDF, pair plot
- **Step 4 — Categorical & Comparison Plots**: bar, box, violin
- **Step 5 — Heatmaps & Subplots (OOP API)**: correlation heatmap, pivot heatmap, 2×2 dashboard
- **Step 6 — Insights & Storytelling**: 2–3 clear observations per dataset

---

## ▶️ How to Run

```bash
# From the repository root, with your environment activated
cd Machine-Learning/Tasks/Task3/"Python Code"
python -m pip install -r ../requirements.txt
python -m jupyter notebook task.ipynb
# Run all cells top to bottom
```

---

## 🛠️ Requirements

Install the packages in [requirements.txt](requirements.txt), including Jupyter for notebook tasks. Use the [coursework setup guide](../../../docs/running-coursework.md) to create an environment first.

---

## 👤 Author

**Abdlrhman** — IEEE SSCS AUSC, AI Team


## Course materials

Assignment and reference resources: [Task 3 materials](../../../course-materials/Machine-Learning/Task3). See the [course-materials index](../../../course-materials/README.md) for all weeks.

# Phase 2 — Natural Language Processing

Work from the NLP phase after the Machine Learning coursework. This index covers the Week 1 implementations currently presented in the portfolio; later local drafts are not listed as completed submissions.

## Layout

```text
NLP/
└── Week1/
    ├── Task0/       # Vector rotation notebook + Streamlit interface
    └── Project1/    # Neurova: Arabic/English sentiment + language routing
```

Assignment briefs and lecture notebooks are under [Week 1 course materials](../course-materials/NLP/Week1). The [materials index](../course-materials/README.md) also links the shared ML/NLP session.

## Task 0 — Vector rotation

[Task notes and setup](Week1/Task0/README.md) cover `task0.ipynb`, `vector_rotation.py`, and the local image inputs.

## Project 1 — Neurova NLP

Solo project with Arabic and English sentiment models, language detection, evaluation notes, tests, and a Streamlit interface.

From the repository root:

```bash
cd NLP/Week1/Project1
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest tests/ -q
python -m streamlit run app.py
```

Saved models are included. Datasets are not required for inference. See the [project results and demo](Week1/Project1/README.md). Keep `.venv/` out of Git.

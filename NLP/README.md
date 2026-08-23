# Phase 2 — Natural Language Processing

Work from the **NLP team**, started after the Machine Learning phase.

There is no Week 2 or Week 3 material in this repository yet. Empty week folders are not created.

## Layout

```text
NLP/
├── Task0/                 # Vector rotation (images + notebook)
└── Week1/
    ├── 1672026.pdf
    ├── project_1_neurova_nlp.pdf
    ├── code/              # Week 1 lecture / practice notebooks
    └── Project1/          # Neurova NLP project (Arabic + English + language ID)
```

## Task 0

Introductory NLP notebook (`task0.ipynb`) and `vector_rotation.py`, with `desert.jpg` / `forest.jpg`.

## Week 1

- `code/` — lecture notebooks and a small `channel.py` helper.
- `Project1/` — sentiment / classification project (Arabic model, English model, language classifier, Streamlit `app.py`).

### Run Project 1

```bash
cd NLP/Week1/Project1
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Do **not** commit `.venv/`.

# Project 1 — Neurova NLP

Classical NLP system that **detects whether user text is Arabic or English**, then runs the matching sentiment classifier. No Transformers.

The app shows exactly three fields from the assignment PDF:

| Field | Values |
| :--- | :--- |
| User Text | original input |
| Language | `Arabic` or `English` |
| Sentiment Classification | `Positive` / `Negative` (Arabic also `Neutral`) |

## Architecture

```text
User Text
    → Language Detector  (char TF-IDF + Linear SVM)
         ├─ Arabic  → Arabic sentiment model
         └─ English → English sentiment model
    → { User Text, Language, Sentiment Classification }
```

Mixed-language policy (not specified in the PDF): **if both Arabic and Latin letters are present, route to Arabic** (loanwords such as `iPhone` / `AI` are common in Arabic reviews). Latin only → English. Digits or emoji only → English.

## Datasets

Place the CSVs under `data/raw/` (they are gitignored — too large for Git):

| Language | File | Source |
| :--- | :--- | :--- |
| Arabic | `data/raw/arabic/Final_Data.csv` | [Arabic customer reviews](https://www.kaggle.com/datasets/mohamedramadan2040/arabic-customer-reviews) |
| English | `data/raw/english/MovieReviewTrainingDatabase.csv` | [IMDb binary sentiment](https://www.kaggle.com/datasets/mwallerphunware/imbd-movie-reviews-for-binary-sentiment-analysis) |

English is **binary** (`Positive` / `Negative`). Arabic is **3-class** (`Positive` / `Negative` / `Neutral`). Neutral is not assumed for English.

## Project structure

```text
Project1/
├── app.py                      # Streamlit UI
├── requirements.txt
├── src/
│   ├── Preprocessing_pipeline.py
│   ├── labels.py
│   ├── English_model.py
│   ├── Arabic_model.py
│   ├── language_model.py
│   ├── pipeline.py             # NeurovaNLPPipeline
│   └── validation.py
├── notebooks/
│   ├── Training_english_model.ipynb
│   ├── Training_arabic_model.ipynb
│   └── Training_language_classifier.ipynb
├── models/                     # *.pkl (gitignored)
├── data/raw/{arabic,english}/
└── tests/
```

## Preprocessing

Three **separate** pipelines — not one cleaner for all models:

| Function | Steps |
| :--- | :--- |
| `preprocess_english` | lowercase → URLs → HTML → unwanted chars → whitespace |
| `preprocess_arabic` | URLs → punctuation → tashkeel → tatweel → Alef → Ya → whitespace |
| `preprocess_language_detection` | URLs → HTML → keep letters (any script) |

Arabic stopwords stay off by default so `مش حلو` does not become `حلو`.

## Feature extraction

Classical only: Bag of Words, word TF-IDF, word n-grams, character n-grams.

## Models tried

| Pipeline | Classifiers | Features |
| :--- | :--- | :--- |
| English sentiment | Naive Bayes, Logistic Regression, Linear SVM | TF-IDF uni, TF-IDF uni+bi, Bag of Words |
| Arabic sentiment | Naive Bayes, Logistic Regression, Linear SVM | Word TF-IDF, word n-grams, char n-grams (`char_wb` 3–5) |
| Language ID | Linear SVM | Character TF-IDF (2–5) |

Winner per pipeline is chosen on a **validation** split (test is held out).

## Best models

Chosen on validation, then scored once on the held-out test set:

| Pipeline | Model | Features | Test F1 |
| :--- | :--- | :--- | ---: |
| Language | Linear SVM | Char TF-IDF (2–5) | 0.999 |
| English | Linear SVM | Word TF-IDF (1–2 grams) | 0.892 |
| Arabic | Linear SVM | Character n-grams (`char_wb` 3–5) | 0.589 (macro) |

English F1 is binary (Positive). Arabic F1 is **macro** because Neutral is rare. Language F1 is macro over Arabic/English.

Pickles store the **entire** sklearn Pipeline (vectorizer + classifier).

## Evaluation

Reported in each notebook: accuracy, precision, recall, F1, confusion matrix.

Macro-F1 is used for Arabic because Neutral is rare (~5%). Binary F1 (Positive class) is used for English.

## Installation

```bash
cd NLP/Week1/Project1
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the two CSVs into `data/raw/` as above, then run the three notebooks in order (English → Arabic → language) to write `models/*.pkl`.

## How to run

```bash
streamlit run app.py
pytest tests/ -q
```

## Example usage

```python
from src.pipeline import NeurovaNLPPipeline

nlp = NeurovaNLPPipeline.load()
print(nlp.analyze("I absolutely loved this movie."))
# {'User Text': '...', 'Language': 'English', 'Sentiment Classification': 'Positive'}

print(nlp.analyze("الخدمة سيئة جدا ولن أكرر التجربة"))
# {'User Text': '...', 'Language': 'Arabic', 'Sentiment Classification': 'Negative'}
```

Demo cases (also buttons in the UI):

| Input | Language | Sentiment |
| :--- | :--- | :--- |
| I absolutely loved this movie. | English | Positive |
| This was one of the worst movies I've ever watched. | English | Negative |
| الخدمة ممتازة والتجربة كانت رائعة | Arabic | Positive |
| الخدمة سيئة جدا ولن أكرر التجربة | Arabic | Negative |

Empty text raises `EmptyTextError`. Corrupted pickles raise `RuntimeError`.

## Limitations

- **Negation scope** — `The movie wasn't bad` and `الفيلم مش وحش` are often predicted Negative.
- **Sarcasm** — `Great, another terrible movie.` is not understood as a rhetorical device.
- **Very short text** — `ok` / `تمام` have little signal.
- **Slang and dialect** — sparse in TF-IDF.
- **Spelling variation** — Arabic char n-grams help; they do not fix everything.
- **Mixed language** — routed to Arabic by policy, which can be wrong for English-majority mixed text.
- **Arabic Neutral** — rare class; macro-F1 is much lower than accuracy.

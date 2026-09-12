# Project 1 — Neurova NLP

Classical NLP system that detects whether user text is **Arabic or English**, then runs the matching sentiment classifier. No Transformers and no neural networks.

The app shows the three fields required by the assignment brief:

| Field | Values |
| :--- | :--- |
| User Text | original input |
| Language | `Arabic` or `English` |
| Sentiment Classification | `Positive` / `Negative` (Arabic also `Neutral`) |

## Architecture

```text
User Text
    → validate (empty / no letters → error)
    → Language preprocessing          preprocess_language_detection
    → Fitted char TF-IDF (2–5)
    → Linear SVM
         ├─ Arabic  → preprocess_arabic  → fitted word+char TF-IDF → Logistic Regression
         └─ English → preprocess_english → fitted word TF-IDF (1–2) → Linear SVM
    → { User Text, Language, Sentiment Classification }
```

Runtime language detection is the saved sklearn Pipeline. There is no Unicode/script shortcut for Arabic vs English.

The brief does not define mixed Arabic+English input. Mixed-script strings are classified by the same SVM. Training includes a small set of mixed rows labeled Arabic because Arabic reviews in this dataset often contain Latin product names (`iPhone`, `AI`). Observed examples after the leak-free split: `iPhone ممتاز` → Arabic, `hello مرحبا` → English.

Digits, punctuation, or emoji with **no letters** are rejected before classification. That check is a robustness addition, not a brief requirement.

## Datasets

The CSVs are gitignored (too large for Git). Download them only if you want to **retrain**:

| Language | Path | Source |
| :--- | :--- | :--- |
| Arabic | `data/raw/arabic/Final_Data.csv` | [Arabic customer reviews](https://www.kaggle.com/datasets/mohamedramadan2040/arabic-customer-reviews) |
| English | `data/raw/english/MovieReviewTrainingDatabase.csv` | [IMDb binary sentiment](https://www.kaggle.com/datasets/mwallerphunware/imbd-movie-reviews-for-binary-sentiment-analysis) |

English is **binary** (`Positive` / `Negative`). Arabic is **3-class** (`Positive` / `Negative` / `Neutral`). Neutral is not a class for English.

Inference does **not** need the CSVs. The three `models/*.pkl` files are tracked in Git (about 1.4–2.6 MB each, under GitHub’s 100 MB file limit). Git LFS is not used.

## Repository structure

```text
NLP/Week1/Project1/
├── app.py
├── requirements.txt
├── src/
│   ├── Preprocessing_pipeline.py   # required
│   ├── English_model.py            # required
│   ├── Arabic_model.py             # required
│   ├── language_model.py
│   ├── language_dataset.py
│   ├── pipeline.py
│   └── validation.py
├── notebooks/
│   ├── Training_english_model.ipynb            # required
│   ├── Training_arabic_model.ipynb             # required
│   └── Training_language_classifier.ipynb      # required
├── models/
│   ├── Arabic_model_weights.pkl                # required
│   ├── English_model_weights.pkl               # required
│   └── Language_classifier_weights.pkl         # required
├── data/raw/{arabic,english}/                  # CSVs not in Git
├── scripts/retrain.py
└── tests/
```

Clone path in this portfolio repo:

```bash
git clone https://github.com/abdlrhmanv/IEEE-SSCS-AUSC-AI-Track.git
cd IEEE-SSCS-AUSC-AI-Track/NLP/Week1/Project1
```

## Preprocessing

Training and inference import the same functions:

| Function | Steps |
| :--- | :--- |
| `preprocess_english` | lowercase → URLs/HTML → contraction expansion (`wasn't` → `was not`) → unwanted chars → whitespace |
| `preprocess_arabic` | URLs → emoji → punctuation → tashkeel → tatweel → Alef → Ya → whitespace |
| `preprocess_language_detection` | lowercase → URLs/HTML → keep letters (any script) |

Arabic stopwords stay off by default so `مش حلو` does not become `حلو`.

## Feature extraction and models

Classical only: Bag of Words, word TF-IDF, word n-grams, character n-grams, FeatureUnion.

Winner per pipeline is chosen on a **validation** split (test is held out). Language-ID sources are split **before** derived examples are created.

| Pipeline | Selected model | Features | Held-out test |
| :--- | :--- | :--- | :--- |
| Language | Linear SVM | Char TF-IDF (2–5) | Accuracy / macro-F1 **0.9996** |
| English | Linear SVM | Word TF-IDF (1–2 grams) | Accuracy **0.9010**, Positive F1 **0.9016** |
| Arabic | Logistic Regression (`class_weight=balanced`) | Word n-grams (1–2) + char_wb 3–5 | Accuracy **0.8130**, **macro-F1 0.6240** |

Pickles store the **entire** sklearn Pipeline (fitted vectorizer + classifier). Inference never fits a new vocabulary.

### Language confusion matrix

Labels `['Arabic', 'English']`:

|  | Pred Arabic | Pred English |
| :--- | ---: | ---: |
| **True Arabic** | 14177 | 2 |
| **True English** | 9 | 14170 |

### English confusion matrix

Labels `['Negative', 'Positive']`:

|  | Pred Negative | Pred Positive |
| :--- | ---: | ---: |
| **True Negative** | 2229 | 257 |
| **True Positive** | 236 | 2259 |

### Arabic confusion matrix and Neutral

Labels `['Negative', 'Neutral', 'Positive']`:

|  | Pred Negative | Pred Neutral | Pred Positive |
| :--- | ---: | ---: | ---: |
| **True Negative** | 2316 | 122 | 324 |
| **True Neutral** | 158 | 77 | 141 |
| **True Positive** | 427 | 259 | 3827 |

Neutral is about 5% of the Arabic data and the labels are noisy. After class weighting and train-fold oversampling:

| Class | Precision | Recall | F1 |
| :--- | ---: | ---: | ---: |
| Negative | 0.798 | 0.839 | 0.818 |
| Neutral | 0.168 | 0.205 | 0.185 |
| Positive | 0.892 | 0.848 | 0.869 |

Neutral is still not reliable. Macro-F1 is the honest metric; accuracy overstates quality.

## Installation

Python **3.13** was used for the saved weights (`scikit-learn==1.9.0`). Use the pinned versions in `requirements.txt` so pickle load stays compatible.

```bash
cd NLP/Week1/Project1
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If `python3.13` is not on your PATH, `python3 -m venv .venv` is fine when that interpreter is 3.13.

## How to run

```bash
streamlit run app.py
# or: python -m streamlit run app.py
python -m pytest tests/ -q
```

The app: type Arabic or English text, click **Analyze**. Demo buttons fill the text box; click Analyze afterward.

## Example usage

```python
from src.pipeline import NeurovaNLPPipeline

nlp = NeurovaNLPPipeline.load()
print(nlp.analyze("I absolutely loved this movie."))
# {'User Text': '...', 'Language': 'English', 'Sentiment Classification': 'Positive'}

print(nlp.analyze("الخدمة سيئة جدا ولن أكرر التجربة"))
# {'User Text': '...', 'Language': 'Arabic', 'Sentiment Classification': 'Negative'}
```

| Input | Language | Sentiment |
| :--- | :--- | :--- |
| I absolutely loved this movie. | English | Positive |
| This was one of the worst movies I've ever watched. | English | Negative |
| الخدمة ممتازة والتجربة كانت رائعة | Arabic | Positive |
| الخدمة سيئة جدا ولن أكرر التجربة | Arabic | Negative |

Empty text raises `EmptyTextError`. Input with no Arabic or English letters raises `NonLinguisticTextError`. Corrupted pickles raise `RuntimeError`.

## Retraining

Place the two CSVs under `data/raw/` as in the table above, then:

```bash
# notebooks (open from the notebooks/ directory, or in Jupyter)
# Training_english_model.ipynb
# Training_arabic_model.ipynb
# Training_language_classifier.ipynb

python scripts/retrain.py                 # all three models
python scripts/retrain.py --language-only # language classifier only
```

`models/training_metrics.json` is rewritten by the script. Sentiment notebooks must be re-run if you change those models.

## Limitations

- **Negation scope** — `The movie wasn't bad` and `الفيلم مش وحش` are often still predicted Negative. Expanding English contractions helps tokenization; it does not model “not + bad = good.”
- **Sarcasm** — `Great, another terrible movie.` is not understood as a rhetorical device.
- **Very short text** — `ok` / `تمام` have little sentiment signal (language ID works; polarity is weak).
- **Arabic Neutral** — rare and noisy; macro-F1 remains much lower than accuracy.
- **Mixed language** — classified by the SVM; Latin-heavy mixed strings may be English, product-name mixes may be Arabic.
- **Slang and dialect** — sparse in TF-IDF.
- **Unsupported scripts** — Chinese/Cyrillic/etc. with no Arabic or Latin letters are rejected as non-linguistic. French/German look like English to the detector.

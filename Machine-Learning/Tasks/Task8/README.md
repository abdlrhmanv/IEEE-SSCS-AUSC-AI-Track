# Task 8 — Validation-based classification experiments

Mushroom classification with KNN, SGD logistic regression, and LBFGS logistic regression. The corrected evaluation selects settings on validation data and compares ordinal and one-hot categorical features on the same partitions and search grids.

## Evaluation protocol

1. Map edible to 0 and poisonous to 1; represent `?`/missing feature values as a category.
2. Split deterministically and stratify into **4,874 train / 1,625 validation / 1,625 test rows** (60/20/20, seed 42).
3. Fit the encoder and scaler on training rows only. Both encodings use `StandardScaler`; unknown categories are handled without refitting.
4. Sweep **K=2…150**, **SGD learning rate=0.001…1.000**, and **LBFGS max_iter=1,000…100,000**. Record train and validation F1 only.
5. Choose the highest validation F1, breaking ties with the smallest parameter. Freeze all settings for both encodings before testing.
6. Refit the selected pipelines on train + validation and report test F1 once per model. One-hot is the predefined primary representation.

The tuning API cannot receive test data. F1's positive class is poisonous. Standardizing one-hot indicators weights rare categories more strongly; this is the specified comparison, not a claim that every one-hot distance choice is optimal.

## Recorded results

| Encoding | Model | Validation-selected setting | Validation F1 | Final test F1 |
| :--- | :--- | :--- | ---: | ---: |
| Ordinal | KNN | K=2 | 1.0000 | 1.0000 |
| One-hot | KNN | K=2 | 1.0000 | 0.9987 |
| Ordinal | SGD logistic | eta=0.028 | 0.9646 | 0.9602 |
| One-hot | SGD logistic | eta=0.001 | 1.0000 | 0.9987 |
| Ordinal | LBFGS logistic | max_iter=1,000 | 0.9617 | 0.9610 |
| One-hot | LBFGS logistic | max_iter=1,000 | 1.0000 | 1.0000 |

Validation scores are from fits on the 60% train partition; test scores come after refitting on the 80% development partition. One-hot improved the linear models on this split; KNN decreased slightly. There are **2,498 candidate evaluations** and six final fits. Convergence warnings are counted in the CSVs; none occurred in the six final fits.

The old submission had already inspected this test partition. These corrected results demonstrate a selection protocol that does not use test scores; they are **not a new external blind benchmark**. They describe one split without confidence intervals.

## Inspect or reproduce

From the repository root, in a separate activated Python 3.13 environment:

```bash
cd Machine-Learning/Tasks/Task8
python -m pip install -r requirements.txt
python -m jupyter notebook task.ipynb
```

The notebook displays the recorded run by default. Set `RECOMPUTE = True` to retrain the full grids, or run the module directly:

```bash
python evaluation.py
python -m unittest discover -s tests -v
```

Keep the notebook kernel directory in this Task8 folder. [requirements-evaluation.txt](requirements-evaluation.txt) pins the packages used for the saved evaluation; [requirements.txt](requirements.txt) also installs Jupyter. Headless evaluation does not need Jupyter.

## Files and evidence

- [task.ipynb](task.ipynb): executable report, six validation figures, and final comparison.
- [evaluation.py](evaluation.py): shared selection, preprocessing, final refitting, and export code.
- [tests/test_evaluation.py](tests/test_evaluation.py): isolation and preprocessing regression checks.
- [results/encoding_comparison.csv](results/encoding_comparison.csv): six frozen-model results.
- [results/evaluation_protocol.json](results/evaluation_protocol.json): versions, counts, seeds, dimensions, and settings.
- `results/{ordinal,onehot}_*.csv`: full train/validation sweeps; no test metric columns.
- `plots/{ordinal,onehot}_*_validation.png`: six training/validation curves.
- [legacy/README.md](legacy/README.md): original submission CSVs and plots, preserved with their test-selection limitation clearly identified.

## Sources

[Mushroom (1981), UCI](https://archive.ics.uci.edu/dataset/73/mushroom), DOI [10.24432/C5959T](https://doi.org/10.24432/C5959T); [Kaggle mirror](https://www.kaggle.com/datasets/uciml/mushroom-classification). UCI publishes CC BY 4.0. The local CSV gives the raw features readable names and stores 2,480 missing stalk-root markers as empty fields; evaluation transforms missing values and targets as described above. See [third-party notices](../../../THIRD_PARTY_NOTICES.md).

Method references: [scikit-learn validation guidance](https://scikit-learn.org/stable/modules/cross_validation.html) and [categorical preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features).

Assignment resources: [Task 8 materials](../../../course-materials/Machine-Learning/Task8) · [Materials index](../../../course-materials/README.md).

**Author:** Abdlrhman Hisham Ismail — IEEE SSCS AUSC AI Team. Some explanatory text was drafted with AI assistance and reviewed against the executed results.

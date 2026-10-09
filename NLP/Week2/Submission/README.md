# Neurova NLP Week 2

## Files

- `reports/Task_0_Weight_Initialization.pdf`: zero, random, Xavier, He and LeCun initialization.
- `reports/Task_1_ReLU_Activation_Functions.pdf`: ReLU and seven activation variants.
- `reports/Task_2_LSTM_Project_Report.pdf`: toxicity classification experiment and analysis.
- `task2/notebooks/Toxic_Comment_LSTM.ipynb`: notebook with saved outputs.
- `task2/src/toxic_lstm.py`: preprocessing, training and evaluation.
- `task2/models/selected_model.pt`: selected model, vocabulary and decision thresholds.

## Experiment

The Jigsaw training dataset contains 159,571 comments with six overlapping labels: toxic, severe_toxic, obscene, threat, insult and identity_hate. After collapsing normalized duplicates, 159,298 comments are split into 127,438 training, 15,930 validation and 15,930 test examples.

The model uses a vocabulary of 30,000 tokens, a 64-dimensional embedding, a 64-unit LSTM, dropout of 0.3 and six output logits. Sequences are limited to 128 tokens. Unweighted and positive-weighted binary cross-entropy are compared using the same split and seed. Weighted BCE is selected by validation macro average precision; per-label thresholds are tuned on validation before evaluating the test set.

| Test metric | Value |
|---|---:|
| Micro F1 | 0.7082 |
| Macro F1 | 0.5709 |
| Macro ROC-AUC | 0.9779 |
| Macro PR-AUC | 0.5753 |
| Macro average precision | 0.5790 |

These results use one seed and a local held-out split. They are not competition leaderboard scores.

## Run

The notebook contains saved outputs and can be read without retraining. To execute it, use an environment with the packages in `task2/requirements.txt`.

From this directory on Linux or macOS:

```bash
bash setup_cpu.sh
.venv/bin/python task2/src/toxic_lstm.py --epochs 6 --threads 4
.venv/bin/python -m unittest discover -s task2/src -p 'test_*.py' -v
```

Set `RUN_TRAINING = True` in the notebook to rerun both training configurations. Otherwise it loads the saved results. The original training CSV is included. If needed, `task2/src/download_data.py` downloads and checks it.

For prediction:

```bash
.venv/bin/python task2/src/predict.py "Thank you for improving this article."
```

Package versions are recorded in `task2/environment-lock.txt`. Training histories, plots, split assignments, metrics, predictions and error examples are in `task2/results/`.

## Dataset

[Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge)

Only the original training CSV is used for the local train, validation and test split. Competition test files are not used in this experiment.

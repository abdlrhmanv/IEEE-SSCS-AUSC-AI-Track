# Neurova NLP — Training data

[Portfolio](../../../../README.md) · [Neurova NLP](../README.md)

## Overview

Training CSVs are excluded from Git. Saved pipelines are included under `models/`, so inference does not need these datasets.

## Contents

Paths below are relative to this `data/` directory.

| Dataset | Required path | Source |
| :--- | :--- | :--- |
| Arabic customer reviews | `raw/arabic/Final_Data.csv` | [Kaggle source](https://www.kaggle.com/datasets/mohamedramadan2040/arabic-customer-reviews) |
| English movie reviews | `raw/english/MovieReviewTrainingDatabase.csv` | [Kaggle source](https://www.kaggle.com/datasets/mwallerphunware/imbd-movie-reviews-for-binary-sentiment-analysis) |

## Usage

Download the files only for retraining. Keep their filenames and place them in the raw-data folders, then follow the [project retraining steps](../README.md#retraining).

## Notes

These datasets retain their upstream terms; the portfolio's MIT grant covers original code only. See [source attribution](../../../../THIRD_PARTY_NOTICES.md).

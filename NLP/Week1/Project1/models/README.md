# Neurova NLP — Saved models

[Portfolio](../../../../README.md) · [Neurova NLP](../README.md)

## Overview

This folder contains saved models for Neurova NLP. The [main project notes](../README.md) explain the complete workflow and evaluation context.

## Contents

| Resource | Description |
| :--- | :--- |
| [.gitkeep](.gitkeep) | Directory placeholder; no dataset content is committed |
| [Arabic_model_weights.pkl](Arabic_model_weights.pkl) | Saved inference pipeline |
| [English_model_weights.pkl](English_model_weights.pkl) | Saved inference pipeline |
| [Language_classifier_weights.pkl](Language_classifier_weights.pkl) | Saved inference pipeline |
| [training_metrics.json](training_metrics.json) | Recorded metadata |

## Usage

Run the Neurova application from its project folder using the [project instructions](../README.md). Use the pinned Python 3.13 environment to load the saved pipelines.

## Notes

Pickles contain fitted vectorizers and classifiers; inference does not fit a new vocabulary. Model artifacts are excluded from the MIT code-license grant.

# Original educational submission results

[Portfolio](../../../../README.md) · [Task 8](../README.md)

## Overview

These three CSVs and three plots are preserved byte for byte from the earlier notebook run. They sweep train/test F1 and select settings using the test set; categorical features were encoded before the split. They are historical coursework references and should not be used as unbiased final evaluation evidence.

The corrected [notebook](../task.ipynb) and [evaluation module](../evaluation.py) use train/validation sweeps, fit preprocessing only on training rows, and evaluate fixed settings on test. Revised ordinal and one-hot comparisons use the same new split and grids; their scores should not be treated as an isolated encoding comparison against these historical outputs.

## Contents

| Resource | Description |
| :--- | :--- |
| [Plots](plots/README.md) | Original train/test curves |
| [Result tables](results/README.md) | Original numeric sweeps |

## Usage

Use these files as historical references. Follow the [current task notes](../README.md) for validation-only selection and the revised encoding comparison.

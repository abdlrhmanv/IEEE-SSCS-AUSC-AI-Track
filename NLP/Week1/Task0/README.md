# NLP Week 1 — Vector rotation

A notebook exploring vector/image transformations and a Streamlit interface for rotating a 3D vector around the X, Y, or Z axis. The image inputs (`desert.jpg` and `forest.jpg`) are included locally.

## Run the interface

From the repository root, after creating and activating an environment using the [root setup notes](../../../README.md#getting-started):

```bash
cd NLP/Week1/Task0
python -m pip install -r requirements.txt
python -m streamlit run vector_rotation.py
```

## Inspect the notebook

From the same task directory and environment:

```bash
python -m jupyter notebook task0.ipynb
```

Run the cells in order. Keep the kernel working directory in this folder so the image paths resolve.

## Files

- [task0.ipynb](task0.ipynb): vector/image transformation exercises.
- [vector_rotation.py](vector_rotation.py): interactive Streamlit application.
- [requirements.txt](requirements.txt): task dependencies.

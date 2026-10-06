# Running the coursework

Each task is an independent submission. Create a separate environment for the task you want to inspect; Neurova needs the pinned dependencies in its own requirements file to load its saved models.

## Create an environment

Clone the repository if needed, then open a terminal at its root:

```bash
git clone https://github.com/abdlrhmanv/IEEE-SSCS-AUSC-AI-Track.git
cd IEEE-SSCS-AUSC-AI-Track
python3 -m venv .venv-coursework
source .venv-coursework/bin/activate
```

On Windows, create the environment with `py -m venv .venv-coursework` and activate it with `.venv-coursework\Scripts\activate` in Command Prompt, or `.\.venv-coursework\Scripts\Activate.ps1` in PowerShell. Use `python` inside the activated environment.

Most coursework dependency files list packages needed by the source rather than historical version locks. Task 8 has a separate pinned evaluation requirements file for its recorded run. The matrix/NumPy smoke checks and Task 8 evaluation were run on Python 3.13. Neurova's pinned runtime is tested on Python 3.13. Full notebook retraining on every package combination has not been verified.

## Choose an entry point

All working directories below are relative to the repository root. Run `cd` to the listed directory first. Except for ML Task 0, install the linked requirements before launching. Notebook kernels must use the listed working directory so relative dataset paths resolve.

| Task | Working directory | Entry point | Dependencies |
| :--- | :--- | :--- | :--- |
| ML 0 | `Machine-Learning/Tasks/Task0/Hello IEEE/Code` | `python main.py` | Standard library |
| ML 1 | `Machine-Learning/Tasks/Task1` | See [two script commands](../Machine-Learning/Tasks/Task1/README.md) | [requirements](../Machine-Learning/Tasks/Task1/requirements.txt) |
| ML 2 | `Machine-Learning/Tasks/Task2/Python Code` | `python -m jupyter notebook task.ipynb` | `python -m pip install -r ../requirements.txt` |
| ML 3 | `Machine-Learning/Tasks/Task3/Python Code` | `python -m jupyter notebook task.ipynb` | `python -m pip install -r ../requirements.txt` |
| ML 4 | `Machine-Learning/Tasks/Task4` | `python -m jupyter notebook zara_eda_task.ipynb` | [requirements](../Machine-Learning/Tasks/Task4/requirements.txt) |
| ML 5 | `Machine-Learning/Tasks/Task5` | `python -m jupyter notebook task.ipynb` | [requirements](../Machine-Learning/Tasks/Task5/requirements.txt) |
| ML 6 | `Machine-Learning/Tasks/Task6` | `python -m jupyter notebook task.ipynb` | [requirements](../Machine-Learning/Tasks/Task6/requirements.txt) |
| ML 7 | `Machine-Learning/Tasks/Task7` | `python main.py` or `python Task2_Sigmoid/sigmoid_plot.py` | [requirements](../Machine-Learning/Tasks/Task7/requirements.txt) |
| ML 8 | `Machine-Learning/Tasks/Task8` | `python -m jupyter notebook task.ipynb` | [requirements](../Machine-Learning/Tasks/Task8/requirements.txt) |
| ML 9 | `Machine-Learning/Tasks/Task9` | `python -m jupyter notebook task.ipynb` | [requirements](../Machine-Learning/Tasks/Task9/requirements.txt) |
| ML 10 | Each notebook's child folder inside `Machine-Learning/Tasks/Task10` | See [four notebook commands](../Machine-Learning/Tasks/Task10/README.md) | Install `requirements.txt` from the Task10 parent first |
| NLP Week 1 Task 0 | `NLP/Week1/Task0` | `python -m streamlit run vector_rotation.py` or `python -m jupyter notebook task0.ipynb` | [requirements](../NLP/Week1/Task0/requirements.txt) |
| Neurova | `NLP/Week1/Project1` | `python -m streamlit run app.py` | [Python 3.13 quick start](../NLP/Week1/Project1/README.md#try-the-demo) |

For rows linking a requirements file, run `python -m pip install -r requirements.txt` in that task's root. ML Task 1's examples install first, then enter each script folder. ML Task 10's notebooks import `data_utils.py` from their parent, so use the individual notebook folder as the kernel directory.

ML Task 3 downloads Seaborn example datasets on the first run unless they are cached. ML Task 6 has a local Auto MPG file and a network fallback if that file is missing. Existing result tables describe saved notebook runs; launching Jupyter alone does not reproduce them.

## Materials and team projects

The [course-materials index](../course-materials/README.md) contains briefs, slides, reference exports, and lecture examples. Each ML task README links its materials. The [move manifest](material-moves.json) records original paths, new paths, and content hashes.

[FareCast](../Machine-Learning/Project1/README.md) and [Smart Home](../Machine-Learning/Project2/README.md) have separate source repositories and runtime requirements; follow their walkthroughs.

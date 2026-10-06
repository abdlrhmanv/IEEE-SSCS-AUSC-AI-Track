# FareCast — Uber fare prediction

[Portfolio](../../README.md) · [Machine Learning](../README.md)

## Overview

**Team project · Machine Learning · Python / scikit-learn / Streamlit**

Estimate a ride's fare from pickup/dropoff coordinates, passenger count, and engineered time and distance features. The team application combines regression models with a map-based interface and trip history.

[Team repository](https://github.com/AliMohamed3122005/UberPricePredictionProject) · [Model implementation](https://github.com/AliMohamed3122005/UberPricePredictionProject/tree/main/Model)

## At a glance

| Item | Detail |
| :--- | :--- |
| Ownership | Team project; individual contribution documented below |
| My role | Regression models and preprocessing |
| Interface | Streamlit fare planner |
| Demo scope | Local Models preview; artifact and map fixes needed |

## My contribution

**Abdlrhman Ismail — regression models and preprocessing refactor.** My recorded contribution includes:

- Added Random Forest and Gradient Boosting regressors with optional cross-validation tuning and feature-importance access.
- Added shared model/evaluator interfaces and MAE, RMSE, and R² calculation so regressors use a consistent evaluation contract.
- Added the shared feature-column contract to align model training with engineered preprocessing outputs.
- Refactored cleaning, feature engineering, and visualization into reusable steps, and added training/artifact-writing support.

Evidence: [my implementation commit and file changes](https://github.com/AliMohamed3122005/UberPricePredictionProject/commit/09df36cb6e73dd460b08d660514348260b561bb5).

The map interface and SQLite trip-history layer are team features. This write-up attributes my work to the changes above.

## Demo walkthrough

![FareCast model comparison screen](../../assets/projects/farecast-models.jpg)

Actual local **Models** tab, captured on **6 October 2026**, using the existing saved artifacts. The incompatible default Gradient Boosting artifact was omitted from the temporary preview, so its row reads Not trained. The displayed metrics come from the other artifacts; this screenshot is not a fresh evaluation or a guarantee of fare accuracy.

After following [the setup below](#run-the-team-application):

1. Open **Models** to inspect the available saved regressors and their reported metrics.
2. Open **Plan trip**, click pickup and dropoff locations, and choose passengers, ride type, and pickup time.
3. Click **See prices** when a model is loaded. After the scaling issue is resolved, compare predicted fares against held-out examples before using the estimates.
4. Open **Trip history** to inspect recorded estimates.

The published ngrok demo was **offline on 6 October 2026**. During local preview, the CARTO map tiles returned an API-key-required notice, and the default Gradient Boosting artifact failed to load in the Python 3.13 / scikit-learn 1.9 environment. Its recorded training version is scikit-learn 1.8.0. The team needs to verify artifact compatibility and configure a working tile provider before a full prediction demo. The screenshot above is a limited interface preview.

<a id="current-result-and-limitation"></a>

## Results

The repository contains the model implementations and an interactive fare-planning application. The current [team README](https://github.com/AliMohamed3122005/UberPricePredictionProject/blob/main/README.md) reports a scaling issue in the loaded model artifact. Fare estimates should be revalidated after that issue is resolved; this portfolio does not claim a verified fare-error metric.

<a id="run-the-team-application"></a>

## Run locally

The application source lives in the team repository. This folder contains the portfolio write-up. The [original assignment brief](../../course-materials/Machine-Learning/Project1/assignment.pdf) is archived with the course materials.

```bash
git clone https://github.com/AliMohamed3122005/UberPricePredictionProject.git
cd UberPricePredictionProject
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r App/requirements.txt
python -m streamlit run App/app/main.py
```

For Windows, activate with `.venv\Scripts\activate`. See the team repository for trained model artifacts and current setup notes.

# License scope and third-party materials

The [MIT license](LICENSE) applies only to original code and associated documentation owned by Abdlrhman Hisham Ismail. This includes his original solution code in `Machine-Learning/Tasks/`, `NLP/Week1/Task0/`, and `NLP/Week1/Project1/`, plus original portfolio write-ups. Third-party excerpts retain their own notices and terms.

The following are excluded from that grant: datasets, `course-materials/`, lecture examples, images, IEEE branding, pretrained model artifacts, and code in linked team repositories. Inclusion in this portfolio does not transfer ownership or grant additional reuse rights. The user selected MIT for original code only on 6 October 2026.

## Dataset attribution

| Material | Location / use | Source and status |
| :--- | :--- | :--- |
| Mushroom / Agaricus–Lepiota | Task 8: `mushrooms_raw.data`, `mushrooms.csv` | [UCI Mushroom](https://archive.ics.uci.edu/dataset/73/mushroom), DOI [10.24432/C5959T](https://doi.org/10.24432/C5959T); corresponding [Kaggle mirror](https://www.kaggle.com/datasets/uciml/mushroom-classification). UCI publishes CC BY 4.0. Local CSV adds readable column names and stores the 2,480 missing stalk-root markers as empty fields; the evaluation represents `?` as a missing-value category and maps edible/poisonous to 0/1. |
| Auto MPG | Task 6: `auto-mpg.data` | R. Quinlan (1993), [UCI Auto MPG](https://archive.ics.uci.edu/dataset/9/auto+mpg), DOI [10.24432/C5859H](https://doi.org/10.24432/C5859H). UCI publishes CC BY 4.0. |
| Red Wine Quality | Tasks 9 and 10: `winequality-red.csv` | Cortez, Cerdeira, Almeida, Matos & Reis (2009), [UCI Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality), DOI [10.24432/C56S3T](https://doi.org/10.24432/C56S3T); corresponding [Kaggle mirror](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009). UCI publishes CC BY 4.0. Notebooks derive the binary target `quality >= 6`. |
| Heart disease | Tasks 9 and 10: `heart.csv` | Task 9 cites [this Kaggle source](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data). Exact local-file provenance and redistribution terms have not been independently verified. |
| Titanic passengers | Task 2: `Python Code/DataSet/titanic.csv` | Course-provided local dataset. The submitted notebook does not record the exact download source or license; provenance remains unverified. |
| Zara sales | Task 4: `Zara_sales_EDA.csv` | Course-provided local dataset. Exact publisher/download source and license are not recorded in the submitted notebook. |
| California housing | Task 5: `housing.csv` | Local coursework dataset. Exact export source and redistribution terms are not recorded in the submitted notebook. |
| Iris, Tips, Flights | Task 3: downloaded with `seaborn.load_dataset` | [Seaborn example-data repository](https://github.com/mwaskom/seaborn-data). These are third-party datasets; consult the upstream dataset-specific terms. |
| Arabic customer reviews | Neurova training data; CSV excluded from Git | [Kaggle Arabic customer reviews](https://www.kaggle.com/datasets/mohamedramadan2040/arabic-customer-reviews), as recorded in Neurova's data notes. Upstream terms apply. |
| English movie reviews | Neurova training data; CSV excluded from Git | [Kaggle movie reviews](https://www.kaggle.com/datasets/mwallerphunware/imbd-movie-reviews-for-binary-sentiment-analysis), as recorded in Neurova's data notes. Upstream terms apply. |

UCI source pages were checked on 6 October 2026. Their [CC BY 4.0 terms](https://creativecommons.org/licenses/by/4.0/) require attribution and indication of adaptations; this notice identifies the dataset sources and relevant local transformations. The repository's MIT license does not replace dataset licenses. Unverified rows are documented as unresolved provenance rather than assigned an invented license.

## Course references and media

- `course-materials/` holds assignment briefs, slides, study plans, and lecture examples supplied in the IEEE SSCS AUSC AI track. These are attributed to their course/source context; a redistribution license is not established here. Original names and hashes are recorded in the [move manifest](docs/material-moves.json).
- The IEEE logo in `assets/` is chapter context and remains outside the MIT grant. This portfolio does not claim ownership of IEEE branding.
- The `assets/projects/` screenshots show the actual Neurova, FareCast, and Smart Home interfaces. Their provenance and capture conditions are in the [screenshot notes](assets/projects/README.md); external interface assets and branding retain their respective rights.
- `desert.jpg` and `forest.jpg` in the vector task, and other third-party reference images, have no independently verified image license recorded here.
- Neurova's `.pkl` artifacts are included for inference and excluded from this code-license grant. Their training datasets and third-party components retain their own terms.

## Linked team work and dependencies

[FareCast](https://github.com/AliMohamed3122005/UberPricePredictionProject) and [Smart Home](https://github.com/abdlrhmanv/smart-home-voice-control) are separate team repositories. The portfolio's MIT license does not relicense them; contribution write-ups link to their source and commit evidence.

Installed dependencies are not vendored here. Python, NumPy, pandas, Matplotlib, scikit-learn, Streamlit, and other packages retain their upstream licenses. Follow the dependencies' own notices when redistributing a packaged application.

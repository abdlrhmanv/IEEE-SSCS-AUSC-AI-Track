# Portfolio presentation iterations

Date: **6 October 2026**

Scope: the first nine portfolio iterations — featured projects, personal contributions, visual/demo access, GitHub presentation, material organization, accurate setup/documentation, validation-only model selection, categorical encoding comparison, and license/source attribution. GitHub About metadata is updated remotely. File changes remain in this portfolio checkout; linked team repositories are read as evidence.

## Iteration 1 — Featured projects

**GOAL**

A visitor should encounter the three strongest projects before the course chronology and see each project's purpose, scope, result, and next link.

**CHANGES**

- Added Featured projects above the repository/coursework sections.
- Added concise project descriptions, truthful metrics/limitations, and direct links.
- Kept the learning journey and coursework index available below the project showcase.

**TESTING — PASS**

- Checked that Featured projects precedes Repository structure and the ML task index.
- Checked local project links and matched Neurova's English accuracy (0.8996) and Arabic macro-F1 (0.6221) to its evaluation documentation.
- Confirmed the team/source boundaries for FareCast and Smart Home.

## Iteration 2 — Personal contribution

**GOAL**

Make Abdlrhman Ismail's specific role easy to distinguish from the overall team application.

**CHANGES**

- Neurova: sole-developer ownership, confirmed directly by the user; preprocessing, model training/routing, Streamlit, evaluation, validation, and CI.
- FareCast: regression implementations, shared evaluation/feature contracts, training support, and preprocessing refactor.
- Smart Home: core/adapters architecture, service tests, command rejection/upload validation, STT command override, and serial/mapping/temperature fixes.
- Added evidence links to the project notes and short role labels in the root showcase.

**TESTING — PASS**

Read the actual file changes, not only commit titles:

- [FareCast contribution](https://github.com/AliMohamed3122005/UberPricePredictionProject/commit/09df36cb6e73dd460b08d660514348260b561bb5).
- [Smart Home architecture/tests](https://github.com/abdlrhmanv/smart-home-voice-control/commit/a637ef250c5e32d5059609629bbdc64835b4e882).
- [Smart Home command override](https://github.com/abdlrhmanv/smart-home-voice-control/commit/628f40dd9f55e6520ab0ebb7f94a48b01e5a0d5b).
- [Smart Home mapping/temperature](https://github.com/abdlrhmanv/smart-home-voice-control/commit/8884c6bfebdf78ed7a3e10af840340b793036c17).

Confirmed contribution sections exist for all three projects, contain evidence links, and preserve team attribution.

## Iteration 3 — Visuals and demo access

**GOAL**

Show real project interfaces and give a visitor a clear path to inspect or run them.

**CHANGES**

- Added English/Arabic Neurova screenshots and a quick start with sample-button instructions.
- Added a real Smart Home dashboard screenshot and separate interface/hardware walkthrough steps.
- Added a FareCast Models-screen preview and a local walkthrough.
- Added a linked project-preview gallery to the root README.

**TESTING — PASS for presentation, with runtime limits below**

- Neurova browser flow: English Positive sample → Analyze → English / Positive; Arabic Negative sample → Analyze → Arabic / Negative.
- Existing Neurova suite: **62 passed**, with 30 non-failing NumPy deprecation warnings.
- Smart Home: launched the actual dashboard in a temporary copy, with dark theme, hardware disconnected, locked access, and light/music off. No microphone, password inference, or physical actuation test was performed.
- FareCast: the published ngrok tunnel returned ERR_NGROK_3200 (offline). CARTO tiles returned an API-key-required notice. The local artifact metadata records scikit-learn 1.8.0; using Neurova's 1.9 environment exposed an artifact-loading incompatibility.

- Saved and decoded all **four** JPEG screenshots successfully (approximately 30–62 KB each).
- Checked **42 local links, image references, and heading anchors** across the four edited READMEs; all resolved.
- `git diff --check` passed.
- Rendered the README with Markdown tables, fenced code, and heading anchors; inspected the featured-project table and gallery in the browser. All local previews and badges loaded, and the project descriptions, roles, and links were readable at the normal desktop viewport. This is a local rendering preview, not a claim of a published GitHub update or mobile verification.
- FareCast's limited Models view showed five saved artifacts as Ready. Its incompatible default Gradient Boosting artifact was omitted only from the temporary preview and visibly appeared as Not trained. A matching-version install was attempted but could not be completed because the package index timed out. Full fare inference remains unverified.

## Iteration 4 — GitHub presentation

**GOAL**

Make the repository's About panel identify the strongest applications and relevant technologies, with a useful project destination.

**CHANGES**

- Updated the live GitHub description to: “AI/ML portfolio: Arabic-English sentiment analysis, smart-home voice control, and Uber fare prediction. Python, scikit-learn and Streamlit.”
- Added ten topics: `arduino`, `ieee`, `machine-learning`, `natural-language-processing`, `portfolio`, `python`, `scikit-learn`, `sentiment-analysis`, `streamlit`, `student-project`.
- Set the homepage field to the existing Neurova source folder. This is a project link, not a hosted demo.

**TESTING — PASS**

Read back the repository metadata through the connected GitHub API after saving. The description, homepage, and all ten topics matched the requested values.

## Iteration 5 — Course materials and duplicates

**GOAL**

Keep solutions, results, and research reports easy to find while collecting assignment briefs, lecture examples, and reference exports in one navigable archive.

**CHANGES**

- Reorganized 37 original paths: moved 35 files to `course-materials/` and removed two byte-identical redundant copies.
- Kept one shared ML/NLP session PDF; kept Task 6's root notebook as the canonical copy.
- Renamed briefs and slides descriptively; gave ten reference images date-based filenames.
- Added the [materials index](../course-materials/README.md), task-level material links, and a [move manifest](material-moves.json) with original paths, destinations, and SHA-256 hashes.
- Kept solution code, datasets, model artifacts, written research reports intact. The Task 0 submission ZIP is kept locally and excluded from Git at the user’s request.

**TESTING — PASS**

- All 37 destination content hashes match their originals; all 37 superseded paths are absent.
- The two removed copies match their retained canonical files byte for byte. This saves **15,059,690 bytes (~15.06 MB)** in the working tree; Git history was not rewritten.
- All 79 code/notebook/dataset/model files captured before this iteration retained their content hashes, resolving relocated paths through the manifest.
- All six existing untracked Week 2 files retained their hashes.
- Opened the materials index through the root README link in the browser and inspected its rendered table. It links the briefs, lecture examples, and shared session.

## Iteration 6 — Documentation and setup

**GOAL**

Give a visitor correct file locations, dependency lists, entry points, and implementation statuses without requiring them to infer the setup from source code.

**CHANGES**

- Updated ML task trees to reflect actual files after the material move; corrected the Task 1 research filename and removed nonexistent HTML report entries.
- Added nine task dependency files and updated two existing files. Notebook tasks now include Jupyter; dependencies remain separate by task.
- Added NLP vector-rotation task notes and a [coursework setup guide](running-coursework.md) covering all ML tasks and both NLP Week 1 implementations.
- Standardized environment-aware installation and launch commands, documented Task 10 kernel directories, and documented Task 3's first-run dataset download.
- Updated root and phase indexes: NLP implementations are no longer labeled “In progress”; reference material is listed separately. Removed the stale repository name and outdated material paths.

**TESTING — PASS for documentation and focused smoke checks**

- Checked 24 Markdown documents after the iteration record was added: **160 local links, image references, and heading anchors** resolved.
- Checked all eleven task dependency lists against the source imports and notebook launcher requirements.
- Ran ML Task 0's matrix demo and both ML Task 1 NumPy demos successfully under Python 3.13.
- Ran the first import/data-loading cell for Task 10's Decision Tree, linear SVM, and RBF SVM notebooks from their documented child folders. Each loaded the expected **1,199 training / 400 test rows and 11 features**.
- Checked the Random Forest folder's parent-helper import and dataset loading with the same shapes. Its complete import cell was not run because Optuna is not installed in the existing test environment; Optuna is included in its requirements file.
- Verified the root README's updated indexes and navigated to the material index in the local browser preview.
- `git diff --check` passed. Full notebook sweeps/retraining and clean installation of every environment were not performed in this documentation update.

## Practical limits

- No hosted demo was deployed in this update. Documentation provides local access and accurately describes the unavailable team tunnel.
- A Smart Home hardware demo video needs a connected board and an enrolled speaker; no simulated hardware result is presented as a live demonstration.
- FareCast scaling and map-provider repairs are separate application work; documented artifact metrics are not a new validation of fare accuracy.
- Existing untracked NLP Week 2 material was left untouched.

## Submission archive follow-up

At the user’s request, `NLP/Week1/Task0/Task0_submission.zip` is excluded from Git and kept locally. Removed its relocation entry and documentation links. The mistakenly generated full-repository ZIP was deleted.

## Iteration 7 — Validation-only model selection

**GOAL**

Keep test scores out of parameter selection in Task 8 while retaining all original assignment grid sizes.

**CHANGES**

- Added `evaluation.py` and rebuilt `task.ipynb` as an executable report with an optional full recomputation flag.
- Split 8,124 rows deterministically into 4,874 train / 1,625 validation / 1,625 test rows. Encoders/scalers fit only on training rows during tuning.
- Selected K, SGD learning rate, and iteration budget on validation F1; ties prefer the smallest parameter. Both encoding configurations are frozen before any test scoring, then refit on the 80% development set.
- Retained the six original CSV/plot artifacts under `legacy/`, with their test-selection limitation documented. New sweeps and figures contain train/validation scores only.
- Added pinned evaluation requirements and prepared a Task 8 CI workflow locally. The GitHub connection rejected workflow publication because it lacks the `workflow` permission; the proposed workflow is ignored and excluded from the PR. All seven checks were run locally.

**TESTING — PASS**

- Ran all 149 K values, 1,000 learning rates, and 100 iteration limits for each of two encodings: **2,498 candidate fits** plus six final refits.
- Seven regression tests passed: deterministic/disjoint stratified partitions; validation-only setting choice and tie handling; unseen-category isolation; cached preprocessing equivalence to full pipelines; saved grid/protocol consistency; rejection of incomplete validation scores; exact raw/CSV correspondence after restoring missing-value markers.
- Executed every notebook cell successfully in the Task 8 directory; displayed all six generated validation figures and the recorded final comparison.
- Preserved all six legacy artifacts byte for byte against their original committed versions.
- No convergence warnings occurred in the six final model fits. Candidate warning counts are recorded in sweep CSVs.

The same fixed test partition was already inspected in the older submission. The new implementation fixes selection isolation; it does not create an external blind benchmark. This is one split, without uncertainty intervals.

## Iteration 8 — Categorical encoding comparison

**GOAL**

Compare an ordinal baseline with one-hot features using identical row partitions, model grids, scaling configuration, seeds, and validation-selection rules.

**CHANGES**

- Replaced feature-wise `LabelEncoder` with train-fitted `OrdinalEncoder` for the baseline and `OneHotEncoder` for the predefined primary representation.
- Added unknown-category handling and a combined [comparison CSV](../Machine-Learning/Tasks/Task8/results/encoding_comparison.csv).
- Documented that standardizing one-hot indicators weights rare categories; the comparison covers the specified pipelines rather than every possible categorical distance choice.

**TESTING — PASS**

- Both representations used the same 60/20/20 row partitions. Encoded dimensions: 22 ordinal / 117 one-hot.
- Unknown validation-only categories did not enter the fitted encoder and transformed to finite arrays without refitting.
- Final test F1, ordinal → one-hot: KNN **1.0000 → 0.9987**, SGD **0.9602 → 0.9987**, LBFGS logistic **0.9610 → 1.0000**.
- Selected settings: K=2 for both; SGD eta=0.028 ordinal / 0.001 one-hot; max_iter=1,000 for both.
- Reported the slight KNN decrease alongside the linear-model improvements. Final test results are not used to select an encoding or retune settings.

## Iteration 9 — License and source attribution

**GOAL**

Define reuse rights for original code while keeping third-party materials and linked team work outside that license.

**CHANGES**

- Added [MIT license](../LICENSE) for original code and associated documentation owned by Abdlrhman Hisham Ismail, following the user's explicit choice.
- Added [third-party notices](../THIRD_PARTY_NOTICES.md) covering datasets, coursework references, images, branding, model artifacts, and linked team repositories.
- Linked the scope/source notices from the root README, course-material index, and Task 8 report.

**TESTING — PASS for scope and attribution documentation**

- Checked the official UCI source pages for Mushroom, Auto MPG, and Wine Quality and recorded their CC BY 4.0 attribution and dataset DOIs.
- Kept known Kaggle training-source links and team repository links. Materials without an independently verified source/license are marked as unresolved provenance, not relicensed as MIT.
- Checked the MIT text and explicit exclusions against the user's “original code only” choice.

## Final publication checks

All 176 intended Markdown links, image references, and heading anchors in 26 documents were checked after the edits. Task 0's submission ZIP is ignored and excluded from the staged publication; the six local Week 2 files are left untracked. Full experiment outputs, dataset hash, and package versions are in Task 8's protocol record. A PR publishes the portfolio changes for review; hosted application demos and Smart Home hardware video remain separate work.

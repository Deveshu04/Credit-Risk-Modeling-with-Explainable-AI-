# Credit Risk Modeling with Explainable AI

An end-to-end probability of default (PD) model on the Home Credit Default Risk dataset: feature engineering across seven tables, tuned LightGBM and XGBoost blended in log-odds, exact SHAP explanations, a Basel-III-aligned master scale with IRB capital by segment, and a live Flask dashboard.

## Results

All 307,511 labelled applicants are used. A stratified 20% holdout (61,503 applicants, seed 42) was fixed before any model existed and is used only for final evaluation, never for any modelling choice.

| Metric | Value |
|---|---|
| Holdout AUC (blend) | **0.7991** |
| Holdout Gini | 0.5981 |
| Holdout KS | 0.4550 |
| Holdout PR-AUC | 0.3041 (base rate 8.07%) |
| Holdout Brier score, calibrated PD | 0.0648 |
| 5-fold cross-validated AUC | 0.7959 (fold std 0.0006) |
| Logistic regression baseline, 5-fold AUC | 0.7778 |
| Untuned LightGBM baseline, 5-fold AUC | 0.7866 |
| Features after pruning | 723 of 836 |

Scorecard validation on the holdout: default rates rise strictly across all 10 grades, no grade fails the Jeffreys back-test at 5%, and the population stability index of the grade mix is 0.00025.

## Live dashboard

The dashboard runs on Render's free tier, which puts the service to sleep after 15 minutes without traffic. The first visit after a quiet spell can take up to a minute while it wakes; after that, pages and live scoring respond immediately.

Four pages:

- **Model**: holdout ROC and precision-recall curves, calibration by PD decile, AUC per fold, and a comparison of every model including the baselines.
- **Explainability**: global SHAP importance, a beeswarm of effects by feature value, and dependence plots for the top 20 features.
- **Portfolio**: the 10-grade master scale with live expected loss and risk-weighted assets as you change LGD, by grade and by segment.
- **Applicant**: score any of 5,000 holdout applicants live, see a SHAP waterfall, and edit the strongest drivers to watch PD, grade and capital respond.

## Pipeline

```
Kaggle competition data (home-credit-default-risk)
        |
        +--> 01-eda           analysis only
        |
        +--> 02-features      features.parquet, split.parquet, feature_meta.json
                  |
                  +--> 03-tuning      selected features, tuned parameters, baselines
                  |        |
                  +--> 04-training    refit models, blend weight, OOF and holdout margins, metrics
                  |        |
                  +--> 05-explainability-scorecard   dashboard artifacts
                                |
                                v
                     Flask dashboard (Docker, gunicorn, Render)
```

Each notebook runs on Kaggle's CPU and reads earlier notebooks' outputs as Kaggle data sources. The executed notebooks, with outputs, are in `notebooks/`.

| Notebook | What it does | Kaggle CPU runtime |
|---|---|---|
| 01 EDA | Tables, target balance, anomalies, segment default rates, history signals | about 4 minutes |
| 02 Features | 836 features from seven tables, fixed holdout split | about 6 minutes |
| 03 Tuning | Baselines, zero-gain pruning, Optuna studies for LightGBM and XGBoost | about 3.3 hours |
| 04 Training | 5-fold CV at learning rate 0.02, blend, refit, single holdout evaluation, protected-attribute comparison | about 4 to 4.5 hours |
| 05 Explainability and scorecard | TreeSHAP, calibration, master scale, IRB capital, segment view, artifact export | under 5 minutes |

## Why five chained notebooks instead of one

Run end to end, the pipeline takes about 7.5 to 8 hours of Kaggle CPU time, most of it in tuning (about 3.3 hours) and training with the comparison model (about 4 to 4.5 hours). One notebook would fit inside Kaggle's 12-hour session limit, but with little margin, and it would tie every stage to every other: a failure in a late cell throws away all the hours before it, and any change anywhere means rerunning everything.

Chaining removes that coupling. Each stage writes its outputs once and later stages read them as Kaggle inputs, so a stage can be rerun on its own. This paid off during the project. When the protected-attribute comparison was added to training, only notebooks 4 and 5 were rerun, and the 3.3-hour tuning run was reused untouched. Tuning and training are separate notebooks for exactly that reason. The split also keeps each notebook short enough to read as one argument, with every code cell between a markdown cell that states its aim and one that states the conclusion drawn from its output.

## Design decisions

**Holdout protocol.** The 80/20 split is stratified on the target with seed 42 and fixed in notebook 2, before any model exists. Feature pruning, tuning, the blend weight, the refit length, calibration and the grade cut-offs are all chosen on training data, mostly on out-of-fold predictions. The holdout is scored only by the final refit models in notebook 4 (the served blend and the protected-attribute comparison), after every choice is fixed.

**Tuning on CPU.** Kaggle's GPU quota was not available, so tuning was designed for four CPU cores. Optuna's TPE sampler searches at learning rate 0.1 to keep trials short, XGBoost is tuned on a stratified half of the training rows because its histogram method is slower on CPU, and the 113 features that no default LightGBM fold model ever split on are dropped first. Final training then reuses the tuned tree settings at learning rate 0.02 on 5 folds. Both studies plateaued well before their budgets ran out (LightGBM found its best region by trial 6), so more trials would likely add little.

**Blending in log-odds.** The blend is `0.70 * LightGBM + 0.30 * XGBoost` on raw margins, with the weight chosen on out-of-fold AUC. Blending margins rather than probabilities keeps SHAP values additive: the blended explanation plus the base value equals the blended prediction exactly.

**Calibration.** Platt scaling and isotonic regression were compared by 5-fold cross-validated Brier score on out-of-fold margins, and Platt won narrowly (0.065383 against 0.065400). Calibrated PDs match observed default rates within 0.7 percentage points in every holdout decile (the largest gap, 0.66 points, is in the ninth decile), and the holdout mean PD is 8.08% against 8.07% observed.

**Score scaling.** PD converts to points as `Score = Offset + Factor * ln(odds)`, with 600 points at good-to-bad odds of 50:1 and 20 points to double the odds, a common industry anchor. Holdout scores run from 432 to 671.

**Master scale.** Ten grades are cut at the out-of-fold score deciles, with a rule to merge neighbours if default rates ever failed to rise (none needed merging). Each grade's PD is its pooled out-of-fold default rate, from 0.80% in grade 1 to 30.88% in grade 10. Capital uses the pooled grade PD, as the IRB approach does for retail pools.

**Floors.** PDs are floored at 0.05%, the Basel III floor for retail exposures, and LGD defaults to 45% with a 30% floor, the Basel III A-IRB floor for other unsecured retail exposures. The dashboard lets you move LGD between 30% and 100%. Revolving loans are treated as other retail for simplicity; under the IRB rules they could qualify as qualifying revolving retail exposures, which use a fixed 4% asset correlation, a 50% LGD floor and a 0.10% PD floor for revolvers. No grade PD here is below 0.8%, so the PD floors never bind.

**Capital.** Risk-weighted assets follow the Basel IRB formula for other retail exposures: asset correlation between 3% and 16% depending on PD, the 99.9% conditional default rate, no maturity adjustment, `RWA = 12.5 * K * EAD` with exposure at default equal to the credit amount. A unit test checks `K` against an independently computed value (PD 1% and LGD 45% give K = 0.03661818, a 45.77% risk weight). At 45% LGD the holdout portfolio's RWA density is 68.6%, with risk weights rising from 41% in grade 1 to 116% in grade 10.

**Validation.** Each grade gets a Jeffreys test, the ECB's back-test for PD (does the observed default count look too high for the grade PD?), and the grade mix is checked for drift between out-of-fold and holdout with the population stability index, where below 0.10 is stable.

**SHAP at serving time.** The dashboard computes exact TreeSHAP with each library's own implementation (`pred_contrib` in LightGBM, `pred_contribs` in XGBoost) instead of the `shap` package. It is exact, takes a few milliseconds per applicant, and keeps the container small.

**Two runs per notebook.** The Kaggle CLI returns only printed output and saved files, not rendered cells. So every notebook prints the numbers its conclusions rely on and saves its figures, a first run produces them, the conclusions are written from them, and a final run reproduces the first. Seeds, deterministic LightGBM and Optuna budgets set by trial count make the two runs match line for line.

## Protected attributes

`CODE_GENDER` turns out to be the second strongest driver by mean absolute SHAP, and `NAME_FAMILY_STATUS` ranks eleventh. Both are protected attributes under fair-lending rules: in the US the Equal Credit Opportunity Act prohibits using sex or marital status in credit decisions, and EU equal-treatment rules restrict sex-based pricing. The served model keeps them because the competition data includes them and the benchmark was set on that basis, but a production model would exclude them and be tested for disparate impact.

To measure the cost, notebook 4 repeats the identical procedure without both attributes and scores the same holdout:

| Holdout metric | With both attributes (served) | Without both | Difference |
|---|---|---|---|
| AUC | 0.7991 | 0.7971 | -0.0020 |
| Gini | 0.5981 | 0.5942 | -0.0039 |
| KS | 0.4550 | 0.4500 | -0.0050 |
| PR-AUC | 0.3041 | 0.3012 | -0.0028 |
| 5-fold cross-validated AUC | 0.7959 | 0.7941 | -0.0018 |

Without them the model still meets the 0.797 benchmark, so the two attributes add only about 0.002 of AUC. The real cost shows up in calibration by group. With gender, mean predicted PD matches each group's observed default rate (6.94% against 6.99% for women, 9.92% against 10.17% for men). Without it, the predicted gap between the groups narrows (7.33% against 9.22%), but women are now over-predicted and men under-predicted, because correlated features carry part of the signal. Removing a protected attribute is necessary but not sufficient for fair lending: a production model would also need disparate-impact testing and an explicit choice between calibration within groups and parity between them. The dashboard's Model page shows both models side by side.

## Reproduce

1. Accept the Home Credit Default Risk competition rules on Kaggle and authenticate the Kaggle CLI.
2. Create a Python 3.12 virtual environment and install `requirements-dev.txt`.
3. In every `notebooks/*/kernel-metadata.json`, replace the `deveshupathak` owner in `id` and `kernel_sources` with your Kaggle username. The metadata keeps each notebook private on Kaggle.
4. Push the committed notebooks to Kaggle in order, 01 to 05, waiting for each run to finish before starting the next, since each one reads the previous outputs:
   ```bash
   python scripts/run_notebook.py push 01-eda
   ```
   ```bash
   python scripts/run_notebook.py wait 01-eda
   ```
   Pushing reruns the notebook on Kaggle. `push --smoke` runs a quick check on a subsample first, and `fetch` downloads a run's printed output and figures.
5. Download notebook 05's `artifacts/` output into `app/artifacts/`, then run the tests:
   ```bash
   python -m pytest
   ```
6. Run the dashboard locally:
   ```bash
   python -m flask --app "app/server.py:create_app()" run --port 7860
   ```

To deploy your own copy, set your repository name in `scripts/upload_artifacts.py` and in `ARTIFACT_REPO` in `render.yaml`, upload the artifacts to a private Hugging Face model repo with that script, create a Render service from `render.yaml`, and set `HF_TOKEN` to a read token in Render's environment settings.

## Repository layout

```
notebooks/      five executed Kaggle notebooks and their kernel metadata
app/            Flask dashboard: server, model wrapper, Basel calculations, templates, static files, Dockerfile
tests/          unit and integration tests (synthetic artifacts; real-artifact checks run when artifacts are present)
scripts/        Kaggle notebook runner and artifact upload
render.yaml     Render deployment blueprint
```

## Data

The data comes from the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) competition on Kaggle and is used under its rules. It is not redistributed in this repository; the notebooks read it directly on Kaggle.

# Notebooks

Notebooks will be numbered in execution order:

1. `01_data_validation.ipynb`
2. `02_exploratory_data_analysis.ipynb`
3. `03_feature_engineering.ipynb`
4. `04_model_experiments.ipynb`
5. `05_model_selection.ipynb`
6. `06_hyperparameter_tuning.ipynb`
7. `07_model_interpretation.ipynb`
8. `08_final_model_evaluation.ipynb`

Reusable logic will be moved into `src/crop_yield/` so production behaviour
does not depend on manually running notebooks.

## Completed

- `01_data_validation.ipynb` validates the source grain and creates the local
  processed modelling table.
- `02_exploratory_data_analysis.ipynb` documents distributions, crop-level
  yield differences, the time trend, numeric associations, and important
  interpretation limitations.
- `03_feature_engineering.ipynb` applies the chronological split, separates the
  target, and fits leakage-safe categorical and numeric transformations on
  training data only.
- `04_model_experiments.ipynb` compares global- and crop-median benchmarks with
  a leakage-safe linear-regression pipeline using validation data only.
- `05_model_selection.ipynb` compares positive-output linear and nonlinear
  candidates, then analyzes the leading random forest by crop and year.
- `06_hyperparameter_tuning.ipynb` tunes random-forest complexity with
  expanding temporal folds, checks seed stability, and evaluates the selected
  configuration on the outer validation years without touching test data.
- `07_model_interpretation.ipynb` measures grouped permutation importance,
  reconciles it with native tree importance, and documents the forest's
  inability to extrapolate the year feature beyond its training range.
- `08_final_model_evaluation.ipynb` performs the locked 2011–2013 test
  evaluation and documents the research-versus-deployment model trade-off.

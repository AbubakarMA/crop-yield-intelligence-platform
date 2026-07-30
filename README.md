# Crop Yield Intelligence Platform

An end-to-end machine-learning portfolio project for predicting agricultural
crop yield from historical crop, location, weather, and pesticide data.

## Project objective

The project asks:

> How accurately can historical crop, location, rainfall, temperature, and
> pesticide information predict agricultural yield?

The final product will include reproducible data preparation, exploratory
analysis, regression modelling, model interpretation, a prediction API, a
user-facing application, automated tests, containerisation, deployment, and
monitoring.

## Intended users

- Agricultural analysts
- Extension organisations
- Agribusinesses
- NGOs and government planning teams
- Researchers

## Planned architecture

```text
Kaggle data -> validation -> analysis -> feature pipeline -> trained model
                                                           |
                                                           v
User interface <- prediction API <- versioned model artefact
                                      |
                                      v
                              logs and monitoring
```

## Project structure

```text
.
├── data/                  # Dataset documentation; local data files are ignored
├── docs/                  # Problem definition and project decisions
├── notebooks/             # Numbered exploration and modelling notebooks
├── src/crop_yield/        # Reusable production Python code
├── tests/                 # Automated tests
├── .env.example           # Safe template for environment variables
├── pyproject.toml         # Package and tool configuration
├── requirements.txt       # Reproducible Python dependencies
└── README.md
```

## Development phases

1. Define the problem and success criteria.
2. Acquire and validate the data.
3. Clean and explore the data.
4. Engineer features and establish a baseline.
5. Train, compare, and interpret regression models.
6. Package the selected model.
7. Build a FastAPI prediction service.
8. Build a Streamlit user application.
9. Test and containerise the system.
10. Deploy and monitor the application.

## Current status

**Phase 8 — Nonlinear model selection**

The source files have been profiled and a reproducible data-quality workflow
produces one validated observation per country, crop, and year. Exploratory
analysis shows that crop type strongly separates yield, overall numeric
associations with yield are weak, 2003 is absent, and rainfall is constant
through time within each country. The analysis treats these relationships as
descriptive rather than causal and documents crop-mix confounding in country
comparisons. A chronological train/validation/test strategy prevents future
years from influencing model development. The preprocessing pipeline one-hot
encodes country and crop, standardizes numeric features, reduces pesticide
right-skew, and tolerates previously unseen categories without learning from
validation or test data. Validation experiments compare global- and crop-median
benchmarks with linear regression. Linear regression improves validation MAE
but still produces physically impossible negative-yield predictions, so it is
retained as a baseline rather than selected for production. Positive-output and
tree-based candidates have now been compared. Random forest is the strongest
initial validation candidate, with substantially lower overall error but
larger remaining errors for cassava, potatoes, and sweet potatoes.

See
[`notebooks/02_exploratory_data_analysis.ipynb`](notebooks/02_exploratory_data_analysis.ipynb)
for the complete Phase 4 analysis and
[`docs/validation_strategy.md`](docs/validation_strategy.md) for the modelling
evaluation design. See
[`notebooks/03_feature_engineering.ipynb`](notebooks/03_feature_engineering.ipynb)
for the leakage-safe feature workflow and
[`notebooks/04_model_experiments.ipynb`](notebooks/04_model_experiments.ipynb)
for the first model comparison and
[`notebooks/05_model_selection.ipynb`](notebooks/05_model_selection.ipynb) for
the nonlinear candidate evaluation.

## Data source

The project uses the
[Crop Yield Prediction Dataset](https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset).
The raw and processed datasets are not stored in this repository. See
[`data/README.md`](data/README.md) for download and validation instructions.

## Author

**Abubakar Mamudu Alutiba**  
Crop and Soil Sciences graduate and data analyst pursuing an MSc in Financial
Engineering.

## Licence

Project code is available under the [MIT License](LICENSE). The Kaggle dataset
retains its own licence and terms.

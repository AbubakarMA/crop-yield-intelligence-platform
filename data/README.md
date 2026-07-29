# Data Directory

Raw and processed datasets are deliberately excluded from Git because the
source dataset has its own terms and generated files can be reproduced.

Local structure:

```text
data/
├── raw/          # Original, unchanged source files
├── interim/      # Intermediate transformation outputs
└── processed/    # Model-ready datasets
```

## Source

Download the
[Crop Yield Prediction Dataset](https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset)
and place its five CSV files inside `data/raw/`:

- `pesticides.csv`
- `rainfall.csv`
- `temp.csv`
- `yield.csv`
- `yield_df.csv`

Do not commit these files.

## Build the validated dataset

Run the Phase 3 notebook from the project root:

```bash
jupyter nbconvert --execute --to notebook --inplace \
  notebooks/01_data_validation.ipynb
```

The notebook writes `data/processed/crop_yield_modeling.csv`. That generated
file is also ignored by Git.

## Intended grain

One processed row represents one country, crop, and year. The source merged
file contains repeated records caused by multiple temperature observations.
The preparation code removes exact semantic duplicates and averages the
remaining temperatures within each country-crop-year group.

The year 2003 is absent from the supplied merged dataset and is documented as
a known coverage gap.

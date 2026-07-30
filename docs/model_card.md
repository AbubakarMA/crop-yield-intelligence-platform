# Model card

## Intended use

The system estimates historical crop yield for country–crop combinations
represented in the training data. It supports agricultural analysis,
demonstration, and portfolio learning. It is not a causal recommendation
engine and must not be used as a long-range climate, investment, or food
security forecast.

## Data

- Source: Kaggle Crop Yield Prediction Dataset
- Observation: one country–crop–year
- Historical coverage: 1990–2013, with 2003 absent
- Target: yield in hectograms per hectare (`hg/ha`)
- Inputs: country, crop, year, annual rainfall, national pesticide use, and
  average temperature

## Validation design

The split is chronological:

- Training: 1990–2007
- Validation and model selection: 2008–2010
- Final test: 2011–2013

After all model and hyperparameter decisions were locked, training and
validation were combined and the sealed test period was evaluated once.

## Final results

| Model | Purpose | Test MAE | Test RMSE | Test R² |
|---|---|---:|---:|---:|
| 300-tree unrestricted random forest | Research model | 11,065.87 hg/ha | 21,938.75 hg/ha | 0.9382 |
| 100-tree depth-18 random forest | Browser deployment | 11,815.30 hg/ha | 22,615.23 hg/ha | 0.9343 |

The deployed forest trades about 6.8% higher MAE for a much smaller and more
portable artifact. Both models produced zero negative test predictions.

## Important limitations

- The forest interpolates learned historical patterns; it cannot extrapolate a
  genuine future-year trend.
- Rainfall is constant through time within each country in the source data, so
  it partly acts as a country identifier.
- Pesticide use is a national total, not a per-hectare exposure.
- Country effects can proxy unmeasured technology, seed, infrastructure, and
  reporting differences.
- Errors are larger for cassava, potatoes, sweet potatoes, plantains, and yams.
- Feature importance describes model reliance, not causation.

## Monitoring

Production monitoring should track request failures, out-of-domain inputs,
prediction distributions, feature drift, and error by crop when new observed
yield becomes available. Any data extending beyond 2013 requires a documented
retraining and temporal backtest before the model version changes.

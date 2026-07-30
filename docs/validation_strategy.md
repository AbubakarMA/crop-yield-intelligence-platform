# Validation Strategy

## Decision

The primary model-development split is chronological:

| Partition | Years | Rows | Share |
|---|---:|---:|---:|
| Training | 1990–2007 | 9,609 | 73.2% |
| Validation | 2008–2010 | 1,751 | 13.3% |
| Test | 2011–2013 | 1,770 | 13.5% |

Training learns model parameters. Validation is used to compare algorithms and
tune settings. Test remains untouched until the modelling approach and settings
have been finalized.

## Why a random row split was rejected

The dataset contains several crop observations for the same country and year.
Those observations reuse country-year rainfall, temperature, and national
pesticide values.

In a deterministic 80/10/10 random row split:

- every validation country-crop pair also appears in training;
- every test country-crop pair also appears in training;
- 99.1% of validation country-year combinations appear in training; and
- 98.9% of test country-year combinations appear in training.

A random split could therefore place one crop from a country-year in training
and another crop from the same country-year in validation or test. This is not
direct target leakage, but it creates an unrealistically similar evaluation
set and can produce an optimistic performance estimate.

## Leakage controls

1. Split observations before fitting any preprocessing step.
2. Fit encoders, imputers, scalers, and transformations on training only.
3. Use validation—not test—to choose features, algorithms, and settings.
4. Evaluate test once after the modelling choices are finalized.
5. Keep preprocessing and the model inside one scikit-learn pipeline.

## Coverage and known challenge

All ten crops are represented in every partition. Validation contains 100
countries and test contains 101. Sudan first appears in the test period, so the
categorical encoder must tolerate unknown countries. This is a useful
real-world robustness check, but performance for unseen countries should be
reported separately.

## Secondary robustness evaluation

The primary split tests prediction for future years. A later country-grouped
evaluation will test the harder question of generalising to countries that
were entirely unseen during training. It will supplement rather than replace
the chronological test.

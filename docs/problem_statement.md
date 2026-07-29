# Problem Statement

## Decision context

Agricultural planners need credible yield estimates to support production
planning, resource allocation, and early investigation of unusual conditions.
This project will develop a regression system that estimates crop yield from
historical agricultural and environmental observations.

## Analytical question

How accurately can historical crop, location, rainfall, temperature, and
pesticide information predict agricultural yield?

## Unit of analysis

One observation represents a crop grown in a geographical area during a given
year, together with its recorded input conditions and yield.

## Target

The target is crop yield, a continuous numerical outcome. This makes the task
a regression problem.

## Initial predictors

Candidate predictors will be confirmed after data validation. They are
expected to include:

- geographical area;
- crop type;
- year;
- average annual rainfall;
- average temperature; and
- pesticide use.

## Evaluation strategy

The model will be compared with a simple baseline and evaluated using:

- mean absolute error (MAE);
- root mean squared error (RMSE);
- coefficient of determination (R-squared);
- performance across crops and geographical areas; and
- operational reliability of the deployed service.

## Boundaries and limitations

- The system is predictive, not causal.
- Historical associations do not prove that an input caused a yield change.
- Predictions outside the crops, places, years, or environmental conditions
  represented in the training data may be unreliable.
- Dataset quality, measurement definitions, and licensing must be verified
  before modelling.
- The application will support human judgement rather than replace agronomic
  expertise.

## Definition of done

The portfolio project is complete when another person can reproduce the data
pipeline, run the tests, train the approved model, start the API and user
interface, and access the deployed application using the repository
documentation.

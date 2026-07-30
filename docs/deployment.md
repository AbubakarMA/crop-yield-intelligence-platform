# Deployment guide

## Live application

The public portfolio application is available at
[crop-yield-intelligence.alutiba.chatgpt.site](https://crop-yield-intelligence.alutiba.chatgpt.site).

It loads a compressed, depth-capped random forest and performs inference
directly in the browser. The portable forest was selected using validation
data, tested once after selection, and then refitted on all 1990–2013
observations for deployment.

## Python service

Generate the local artifacts:

```bash
python -m crop_yield.production
```

Run the API:

```bash
uvicorn crop_yield.api:app --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /health` — service and model readiness
- `GET /metadata` — model version, ranges, metrics, and limitations
- `POST /predict` — one validated historical yield estimate

## Container

The Docker image expects the generated artifacts directory to exist:

```bash
python -m crop_yield.production
docker compose up --build
```

The API is exposed on port 8000 and the Streamlit client on port 8501.

## Monitoring plan

Track request count, latency, failures, rejected inputs, feature and prediction
distributions, and error by crop when actual yield becomes available. Retrain
only after adding newer observations and repeating the chronological backtest.
A retrained model must receive a new semantic version and updated model card.

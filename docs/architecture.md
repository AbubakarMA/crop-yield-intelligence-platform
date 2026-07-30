# Production architecture

```mermaid
flowchart TD
    A[Kaggle CSV] --> B[Schema and grain validation]
    B --> C[Chronological model development]
    C --> D[Locked final test]
    D --> E[Versioned model artifact]
    E --> F[FastAPI prediction contract]
    F --> G[Streamlit local app]
    E --> H[Portable browser model]
    H --> I[Deployed portfolio app]
    F --> J[Health and metadata endpoints]
```

## Separation of responsibilities

- `data.py` establishes a trustworthy country–crop–year table.
- `splitting.py` prevents future years from leaking into training.
- `preprocessing.py` owns category encoding and numeric transformations.
- `models.py` defines reproducible candidates.
- `production.py` performs final evaluation, refitting, and artifact creation.
- `api.py` validates requests and serves predictions through a stable contract.
- `streamlit_app.py` is a thin client of the API.
- The deployed web experience runs the portable depth-capped forest directly
  in the browser and exposes its limitations beside every estimate.

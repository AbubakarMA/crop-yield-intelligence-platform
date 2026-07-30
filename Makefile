.PHONY: install test lint train api app

install:
	python -m pip install -r requirements.txt
	python -m pip install -e .

test:
	pytest -q

lint:
	ruff check src tests

train:
	python -m crop_yield.production

api:
	uvicorn crop_yield.api:app --reload

app:
	streamlit run streamlit_app.py

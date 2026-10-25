# Used Car Price & Fuel Type Predictor

A small full-stack machine learning app. It predicts a used car's selling price
(regression) and its fuel type (classification).

## Project structure
- `data/cars.csv` – the dataset
- `train_models.py` – trains both models and saves them with joblib
- `main.py` – FastAPI backend with the two POST endpoints
- `static/index.html` – web page with a form for each model
- `models/` – saved models

## How to run
1. Install the packages:
   pip install -r requirements.txt
2. Train and save the models:
   python train_models.py
3. Start the server:
   python -m uvicorn main:app --reload
4. Open http://127.0.0.1:8000 in our browser.

## Models
- Regression: [Linear Regression] – predicts selling price. R² = [value]
- Classification: [Random Forest] – predicts fuel type. Accuracy = [value]%

## API endpoints
- POST /predict/regression – returns the predicted price as JSON
- POST /predict/classification – returns the predicted fuel type as JSON

## Dataset
CarDekho used car dataset ([number] rows after cleaning).

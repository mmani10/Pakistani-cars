# Used Car Price Predictor

1. Install:  pip install -r requirements.txt
2. Train:    python train.py      (creates car_price_model.joblib)
3. Run site: python app.py        (open http://127.0.0.1:5000)

Files: train.py (cleaning + model), app.py (Flask backend), templates/index.html (website),
pakwheels_used_car_data_v02.csv (dataset used by both train.py and app.py).

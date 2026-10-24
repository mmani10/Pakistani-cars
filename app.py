"""
Backend for the Car Price Predictor website.
Run:   python app.py      then open  http://127.0.0.1:5000
It connects three things: the saved model, the CSV dataset, and the HTML page.
"""
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from train import load_clean_data, prepare_features, CURRENT_YEAR

app = Flask(__name__)

# ---- Load the model and the dataset once, when the server starts ----
saved = joblib.load("car_price_model.joblib")
model, categories = saved["model"], saved["categories"]
data = load_clean_data()                      # the same CSV used for training
metrics = json.load(open("metrics.json"))


def top_values(series, n=1):
    return series.value_counts().head(n).index.tolist()


def build_options():
    """Everything the website dropdowns need, taken straight from the dataset."""
    cars = {}
    for (make, name), g in data.groupby(["make", "model"]):
        if len(g) < 5:
            continue
        cars.setdefault(make, {})[name] = {
            "count": len(g),
            "year_min": int(g["year"].min()), "year_max": int(g["year"].max()),
            "engines": [int(e) for e in g["engine"].value_counts().head(6).index.sort_values()],
            "body": top_values(g["body"])[0],
            "transmissions": top_values(g["transmission"], 2),
            "fuels": top_values(g["fuel"], 3),
            "typical_engine": int(g["engine"].mode()[0]),
            "typical_mileage": int(g["mileage"].median()),
        }
    makes = sorted(cars, key=lambda m: -sum(v["count"] for v in cars[m].values()))
    return {
        "makes": makes,
        "cars": {m: dict(sorted(cars[m].items(), key=lambda kv: -kv[1]["count"])) for m in makes},
        "cities": [c for c in categories["city"] if c != "Other"],
        "registered": [c for c in categories["registered"] if c != "Other"],
        "metrics": metrics,
    }


OPTIONS = build_options()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/options")
def options():
    return jsonify(OPTIONS)


@app.route("/api/predict", methods=["POST"])
def predict():
    f = request.get_json()
    try:
        year, engine, mileage = int(f["year"]), int(f["engine"]), int(f["mileage"])
        row = pd.DataFrame([{
            "city": f["city"], "assembly": f["assembly"],
            "body": f["body"], "make": f["make"], "model": f["model"],
            "transmission": f["transmission"], "fuel": f["fuel"],
            "registered": f["registered"],
            "year": year, "age": CURRENT_YEAR - year, "engine": engine, "mileage": mileage,
        }])
    except (KeyError, ValueError, TypeError):
        return jsonify({"error": "Please fill in all the fields."}), 400

    X, _ = prepare_features(row, categories)
    price = float(np.expm1(model.predict(X))[0])
    price = round(price, -3)
    margin = metrics["typical_error_pct"] / 100

    # Look up similar real listings from the dataset to back up the prediction
    sim = data[(data["make"] == f["make"]) & (data["model"] == f["model"])
               & (data["year"].between(year - 2, year + 2))]
    similar = {"count": len(sim), "listings": []}
    if len(sim):
        similar["median"] = float(sim["price"].median())
        near = sim.assign(gap=(sim["mileage"] - mileage).abs()).nsmallest(4, "gap")
        similar["listings"] = near[["year", "mileage", "city", "transmission", "price"]].to_dict("records")

    return jsonify({"price": price, "low": round(price * (1 - margin), -3),
                    "high": round(price * (1 + margin), -3), "similar": similar})


if __name__ == "__main__":
    app.run(debug=False, port=5000)

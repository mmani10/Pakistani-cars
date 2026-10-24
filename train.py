"""
Used Car Price Prediction - Model Training
------------------------------------------
Run this file once:   python train.py
It will: load the dataset -> clean it -> compare a few models -> save the best one.
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

DATA_FILE = "pakwheels_used_car_data_v02.csv"
CURRENT_YEAR = 2022          # newest car in the dataset is 2022
CATEGORY_COLUMNS = ["city", "assembly", "body", "make", "model",
                    "transmission", "fuel", "registered"]
NUMBER_COLUMNS = ["year", "age", "engine", "mileage"]


def load_clean_data(path=DATA_FILE):
    """Read the CSV and fix the messy parts (missing values, crazy outliers)."""
    df = pd.read_csv(path)

    # 1. Rows without a price / year / engine are useless for learning
    df = df.dropna(subset=["price", "year", "engine"])

    # 2. Remove unrealistic rows (typing mistakes in the ads)
    df = df[df["price"].between(150_000, 60_000_000)]
    df = df[df["engine"].between(600, 6600)]
    df = df[df["mileage"].between(100, 500_000)]

    # 3. Fill the remaining blanks with sensible values
    df["assembly"] = df["assembly"].fillna("Local")   # blank means locally assembled
    df["fuel"] = df["fuel"].fillna("Petrol")
    df["body"] = df["body"].fillna("Unknown")

    # 4. New helpful column: how old is the car?
    df["age"] = CURRENT_YEAR - df["year"]
    df["year"] = df["year"].astype(int)
    return df.reset_index(drop=True)


def prepare_features(df, categories=None):
    """Turn text columns into 'category' type. Rare values become 'Other'."""
    X = df[CATEGORY_COLUMNS + NUMBER_COLUMNS].copy()
    if categories is None:                      # training time: learn the categories
        categories = {}
        for col in CATEGORY_COLUMNS:
            counts = X[col].value_counts()
            keep = counts[counts >= 30].index.tolist()
            categories[col] = sorted(keep) + ["Other"]
    for col in CATEGORY_COLUMNS:                # prediction time: reuse them
        X[col] = X[col].where(X[col].isin(categories[col]), "Other")
        X[col] = pd.Categorical(X[col], categories=categories[col])
    return X, categories


def score(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{name:<22} MAE = Rs {mae:>10,.0f}   R2 = {r2:.3f}")
    return mae, r2


if __name__ == "__main__":
    df = load_clean_data()
    print(f"Rows used for training: {len(df):,}\n")

    X, categories = prepare_features(df)
    # We predict log(price) because prices are very skewed (a Mehran vs a Land Cruiser).
    y = np.log1p(df["price"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    real_test_price = np.expm1(y_test)

    # ---------- Compare 3 models ----------
    print("Model comparison on unseen test data:")

    # a) Simple baseline: Linear Regression
    X_lin = pd.get_dummies(X, dtype=float)
    Xl_train, Xl_test = X_lin.loc[X_train.index], X_lin.loc[X_test.index]
    lin = LinearRegression().fit(Xl_train, y_train)
    score("Linear Regression", real_test_price, np.expm1(lin.predict(Xl_test)))

    # b) Random Forest
    X_codes = X.apply(lambda c: c.cat.codes if str(c.dtype) == "category" else c)
    forest = RandomForestRegressor(n_estimators=150, min_samples_leaf=2, n_jobs=-1, random_state=1)
    forest.fit(X_codes.loc[X_train.index], y_train)
    score("Random Forest", real_test_price, np.expm1(forest.predict(X_codes.loc[X_test.index])))

    # c) Gradient Boosting (our final choice - usually the most accurate)
    settings = dict(learning_rate=0.08, max_leaf_nodes=127, min_samples_leaf=10,
                    l2_regularization=1.0, categorical_features="from_dtype", random_state=1)
    booster = HistGradientBoostingRegressor(max_iter=2000, early_stopping=True, **settings)
    booster.fit(X_train, y_train)
    pred = np.expm1(booster.predict(X_test))
    mae, r2 = score("Gradient Boosting", real_test_price, pred)
    typical_error_pct = float(np.median(np.abs(pred - real_test_price) / real_test_price))
    print(f"\nTypical error: {typical_error_pct*100:.1f}% of the car's price")

    # ---------- Final model: re-train on ALL data so it learns from every car ----------
    final_model = HistGradientBoostingRegressor(max_iter=booster.n_iter_, early_stopping=False, **settings)
    final_model.fit(X, y)

    joblib.dump({"model": final_model, "categories": categories}, "car_price_model.joblib")
    with open("metrics.json", "w") as f:
        json.dump({"mae": round(mae), "r2": round(r2, 4),
                   "typical_error_pct": round(typical_error_pct * 100, 1),
                   "rows": len(df)}, f)
    print("Saved: car_price_model.joblib and metrics.json")

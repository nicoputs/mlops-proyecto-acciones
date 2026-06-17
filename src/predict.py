import os
import joblib
import pandas as pd

from data import make_prediction_features

MODEL_PATH = os.path.join("models", "model.pkl")
INPUT_PATH = os.path.join("batch_prediction_dataset", "on_demand_dataset.csv")
OUTPUT_PATH = os.path.join("batch_prediction_dataset", "predictions.csv")


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No existe {MODEL_PATH}. Corre primero train.")
    artifact = joblib.load(MODEL_PATH)
    pipeline = artifact["pipeline"]
    features = artifact["features"]

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"No existe {INPUT_PATH}.")
    df = pd.read_csv(INPUT_PATH, index_col="Date", parse_dates=True)

    X, dates = make_prediction_features(df)
    X = X[features]

    pred = pipeline.predict(X)
    proba = pipeline.predict_proba(X)[:, 1]

    out = pd.DataFrame(
        {
            "Date": dates,
            "prediction": pred,
            "direction": ["SUBE" if p == 1 else "BAJA" for p in pred],
            "prob_sube": proba.round(4),
        }
    )
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"Predicciones guardadas en {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

"""
predict.py
----------
Workflow 'on-demand': hace predicciones sobre un dataset nuevo.

Lee:    batch_prediction_dataset/on_demand_dataset.csv  (datos crudos OHLCV)
Genera: batch_prediction_dataset/predictions.csv        (con la prediccion)

Responsable: Nico (Parte 1 - infraestructura).
"""

import os
import joblib
import pandas as pd

from data import make_prediction_features

MODEL_PATH = os.path.join("models", "model.pkl")
INPUT_PATH = os.path.join("batch_prediction_dataset", "on_demand_dataset.csv")
OUTPUT_PATH = os.path.join("batch_prediction_dataset", "predictions.csv")


def main():
    # 1. Cargar el modelo entrenado (pipeline + lista de features)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No existe {MODEL_PATH}. Corre primero train.py."
        )
    artifact = joblib.load(MODEL_PATH)
    pipeline = artifact["pipeline"]
    features = artifact["features"]

    # 2. Leer el dataset de entrada (datos crudos OHLCV)
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"No existe {INPUT_PATH}.")
    df = pd.read_csv(INPUT_PATH, index_col="Date", parse_dates=True)
    print(f"Leidas {len(df)} filas de {INPUT_PATH}")

    # 3. Calcular las MISMAS features que en entrenamiento
    X, dates = make_prediction_features(df)
    X = X[features]  # mismo orden de columnas

    # 4. Predecir direccion (1 = sube, 0 = baja) y probabilidad
    pred = pipeline.predict(X)
    proba = pipeline.predict_proba(X)[:, 1]

    # 5. Guardar resultados en la MISMA carpeta
    out = pd.DataFrame({
        "Date": dates,
        "prediction": pred,                 # 1 sube, 0 baja
        "direction": ["SUBE" if p == 1 else "BAJA" for p in pred],
        "prob_sube": proba.round(4),
    })
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"Predicciones guardadas en {OUTPUT_PATH}")
    print(out.tail())


if __name__ == "__main__":
    main()

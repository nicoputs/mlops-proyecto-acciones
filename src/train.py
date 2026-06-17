"""
train.py
--------
Entrena el modelo, registra el experimento en MLflow y guarda el modelo final.

Este es el script que corre el pipeline de CD (al hacer push a main):
entrena y deja el modelo en models/model.pkl.

Version baseline (Parte 1 - Nico): entrena UN modelo simple para que el
pipeline funcione de punta a punta.

>>> ALEJANDRO (Parte 2): aqui es donde agregas mas modelos, los comparas en
    MLflow con varias metricas, eliges el ganador y dejas ESE como modelo final.
"""

import os
import joblib
import mlflow

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

from data import get_training_data, TICKER

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

# MLflow guarda los experimentos en una carpeta local ./mlruns
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("stock-direction")


def evaluate(model, X, y):
    """Calcula varias metricas sobre un conjunto."""
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, proba),
    }


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Cargar datos y features
    X, y, feature_names = get_training_data()

    # 2. Split RESPETANDO EL TIEMPO (nada de shuffle en series de tiempo):
    #    primer 80% para entrenar, ultimo 20% para probar.
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    print(f"Entrenamiento: {len(X_train)} filas | Prueba: {len(X_test)} filas")

    # 3. Baseline de referencia: predecir siempre la clase mayoritaria.
    #    Sirve para comparar: el modelo debe superar esto.
    majority = y_train.mode()[0]
    baseline_acc = (y_test == majority).mean()
    print(f"Baseline (clase mayoritaria): accuracy = {baseline_acc:.4f}")

    # 4. Entrenar el modelo dentro de un Pipeline (escalado + clasificador)
    with mlflow.start_run(run_name="logistic_regression_baseline"):
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=1.0)),
        ])
        model.fit(X_train, y_train)

        metrics = evaluate(model, X_test, y_test)
        print("Metricas en prueba:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        # Registrar en MLflow
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("ticker", TICKER)
        mlflow.log_param("n_features", len(feature_names))
        mlflow.log_metric("baseline_accuracy", baseline_acc)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

    # 5. Guardar el modelo final (pipeline + lista de features + ticker).
    #    predict.py usara exactamente esto.
    artifact = {
        "pipeline": model,
        "features": feature_names,
        "ticker": TICKER,
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"\nModelo guardado en {MODEL_PATH}")


if __name__ == "__main__":
    main()

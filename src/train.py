import os
import joblib
import mlflow

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from data import get_training_data, TICKER

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("stock-direction")


def evaluate(model, X, y):
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
    X, y, feature_names = get_training_data()

    # split temporal: primer 80% entrena, ultimo 20% prueba (sin shuffle)
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    majority = y_train.mode()[0]
    baseline_acc = (y_test == majority).mean()

    with mlflow.start_run(run_name="random_forest_simple"):
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=100, max_depth=3, random_state=42
                    ),
                ),
            ]
        )
        model.fit(X_train, y_train)

        metrics = evaluate(model, X_test, y_test)
        mlflow.log_param("model", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 3)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("ticker", TICKER)
        mlflow.log_param("n_features", len(feature_names))
        mlflow.log_metric("baseline_accuracy", baseline_acc)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        print("Metricas en prueba:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    joblib.dump(
        {"pipeline": model, "features": feature_names, "ticker": TICKER},
        MODEL_PATH,
    )
    print(f"Modelo guardado en {MODEL_PATH}")


if __name__ == "__main__":
    main()

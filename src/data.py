"""
data.py
-------
Adquisicion de datos y creacion de features para el proyecto.

Responsable: Nico (Parte 1 - adquisicion de datos).

Que hace este archivo:
  1. Descarga el historico de una accion desde Yahoo Finance (yfinance).
  2. Guarda una copia local (snapshot) en datasets/ como respaldo.
  3. Si la API de Yahoo falla, usa el snapshot guardado (asi el pipeline no se cae).
  4. Crea las features (indicadores) y la variable objetivo:
        target = 1 si el precio de cierre SUBE al dia siguiente, 0 si baja.
"""

import os
import pandas as pd
import numpy as np

# Configuracion por defecto del proyecto
TICKER = "AAPL"
START = "2015-01-01"
END = "2024-12-31"

# Ruta del snapshot (respaldo en CSV)
DATASET_DIR = "datasets"
SNAPSHOT_PATH = os.path.join(DATASET_DIR, "historico.csv")


def download_data(ticker=TICKER, start=START, end=END, snapshot_path=SNAPSHOT_PATH):
    """
    Descarga datos de Yahoo Finance. Guarda un snapshot en CSV.
    Si la descarga falla, intenta cargar el snapshot guardado.
    Devuelve un DataFrame con columnas: Open, High, Low, Close, Volume.
    """
    os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)

    try:
        import yfinance as yf
        print(f"Descargando {ticker} de Yahoo Finance ({start} a {end})...")
        df = yf.Ticker(ticker).history(start=start, end=end)

        if df is None or df.empty:
            raise ValueError("Yahoo devolvio un DataFrame vacio.")

        # Nos quedamos solo con las columnas que necesitamos
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index.name = "Date"

        # Guardamos el snapshot de respaldo
        df.to_csv(snapshot_path)
        print(f"Snapshot guardado en {snapshot_path} ({len(df)} filas).")
        return df

    except Exception as e:
        print(f"[AVISO] Fallo la descarga ({e}).")
        if os.path.exists(snapshot_path):
            print(f"Usando snapshot de respaldo: {snapshot_path}")
            df = pd.read_csv(snapshot_path, index_col="Date", parse_dates=True)
            return df
        raise RuntimeError(
            "No se pudo descargar de Yahoo y no hay snapshot de respaldo."
        )


# Lista de features que usa el modelo (mismas en entrenamiento y prediccion)
FEATURE_NAMES = [
    "return",
    "return_lag_1", "return_lag_2", "return_lag_3", "return_lag_5",
    "price_to_ma_5", "price_to_ma_10", "price_to_ma_20",
    "volatility_5", "volatility_10",
    "volume_change", "rsi_14",
]


def _compute_features(df):
    """Agrega las columnas de indicadores al DataFrame (sin target)."""
    df = df.copy().sort_index()

    # Retorno diario
    df["return"] = df["Close"].pct_change()

    # Retornos rezagados (lo que paso 1, 2, 3 y 5 dias atras)
    for lag in [1, 2, 3, 5]:
        df[f"return_lag_{lag}"] = df["return"].shift(lag)

    # Medias moviles y razon precio/media
    for w in [5, 10, 20]:
        df[f"ma_{w}"] = df["Close"].rolling(w).mean()
        df[f"price_to_ma_{w}"] = df["Close"] / df[f"ma_{w}"]

    # Volatilidad (desviacion estandar de los retornos)
    df["volatility_5"] = df["return"].rolling(5).std()
    df["volatility_10"] = df["return"].rolling(10).std()

    # Cambio en el volumen
    df["volume_change"] = df["Volume"].pct_change()

    # RSI de 14 dias
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    return df


def make_features(df):
    """
    Para ENTRENAR. Devuelve (X, y, feature_names) con la variable objetivo.
    """
    df = _compute_features(df)

    # OBJETIVO: 1 si el cierre de MANANA es mayor que el de hoy, si no 0
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    # Quitamos filas con NaN (las primeras por los rolling, la ultima por el shift)
    df = df.dropna(subset=FEATURE_NAMES + ["target"])

    X = df[FEATURE_NAMES]
    y = df["target"]
    return X, y, FEATURE_NAMES


def make_prediction_features(df):
    """
    Para PREDECIR. Calcula las features sobre datos nuevos (sin target).
    Conserva la fecha como columna para identificar cada prediccion.
    """
    df = _compute_features(df)
    df = df.dropna(subset=FEATURE_NAMES)  # quitamos solo las filas sin features
    return df[FEATURE_NAMES], df.index


def get_training_data(ticker=TICKER, start=START, end=END):
    """Funcion de conveniencia: descarga + features en un solo paso."""
    df = download_data(ticker=ticker, start=start, end=end)
    return make_features(df)


if __name__ == "__main__":
    # Prueba rapida al correr este archivo directamente
    X, y, feats = get_training_data()
    print(f"\nFilas: {len(X)} | Features: {len(feats)}")
    print(f"Distribucion del target (0=baja, 1=sube):\n{y.value_counts(normalize=True)}")

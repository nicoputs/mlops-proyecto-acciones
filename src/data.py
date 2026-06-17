import os
import pandas as pd

TICKER = "AAPL"
START = "2015-01-01"
END = "2024-12-31"

DATASET_DIR = "datasets"
SNAPSHOT_PATH = os.path.join(DATASET_DIR, "historico.csv")

FEATURE_NAMES = [
    "return",
    "return_lag_1",
    "return_lag_2",
    "return_lag_3",
    "return_lag_5",
    "price_to_ma_5",
    "price_to_ma_10",
    "price_to_ma_20",
    "volatility_5",
    "volatility_10",
    "volume_change",
    "rsi_14",
]


def download_data(ticker=TICKER, start=START, end=END, snapshot_path=SNAPSHOT_PATH):
    os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)
    try:
        import yfinance as yf

        df = yf.Ticker(ticker).history(start=start, end=end)
        if df is None or df.empty:
            raise ValueError("Yahoo devolvio un DataFrame vacio.")
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index.name = "Date"
        df.to_csv(snapshot_path)
        return df
    except Exception:
        if os.path.exists(snapshot_path):
            return pd.read_csv(snapshot_path, index_col="Date", parse_dates=True)
        raise RuntimeError("No hay conexion con Yahoo ni snapshot de respaldo.")


def _compute_features(df):
    df = df.copy().sort_index()
    df["return"] = df["Close"].pct_change()
    for lag in [1, 2, 3, 5]:
        df[f"return_lag_{lag}"] = df["return"].shift(lag)
    for w in [5, 10, 20]:
        df[f"ma_{w}"] = df["Close"].rolling(w).mean()
        df[f"price_to_ma_{w}"] = df["Close"] / df[f"ma_{w}"]
    df["volatility_5"] = df["return"].rolling(5).std()
    df["volatility_10"] = df["return"].rolling(10).std()
    df["volume_change"] = df["Volume"].pct_change()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["rsi_14"] = 100 - (100 / (1 + rs))
    return df


def make_features(df):
    df = _compute_features(df)
    # objetivo: 1 si el cierre sube al dia siguiente, 0 si baja
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df.dropna(subset=FEATURE_NAMES + ["target"])
    return df[FEATURE_NAMES], df["target"], FEATURE_NAMES


def make_prediction_features(df):
    df = _compute_features(df)
    df = df.dropna(subset=FEATURE_NAMES)
    return df[FEATURE_NAMES], df.index


def get_training_data(ticker=TICKER, start=START, end=END):
    df = download_data(ticker=ticker, start=start, end=end)
    return make_features(df)

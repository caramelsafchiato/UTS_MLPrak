import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src import config
from src.utils import banner, subbanner


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    subbanner("Drop kolom ID dan kebocoran label")
    df = df.drop(columns=config.DROP_COLUMNS)
    print("Kolom dibuang :", config.DROP_COLUMNS)
    print("Kolom tersisa :", df.columns.tolist())
    return df


def encode_type(df: pd.DataFrame) -> pd.DataFrame:
    subbanner(f"Encoding {config.TYPE_COLUMN} {config.TYPE_MAPPING}")
    df = df.copy()
    df[config.TYPE_COLUMN] = df[config.TYPE_COLUMN].map(config.TYPE_MAPPING)
    if df[config.TYPE_COLUMN].isnull().any():
        raise ValueError("Ada nilai Type di luar mapping L/M/H.")
    print(df[config.TYPE_COLUMN].value_counts().sort_index().to_string())
    return df


def separate_xy(df: pd.DataFrame):
    subbanner("Penentuan X (fitur) dan y (label)")
    X, y = df.drop(columns=[config.TARGET]), df[config.TARGET]
    print("Fitur X :", X.columns.tolist())
    print("Label y :", config.TARGET)
    print("Bentuk X:", X.shape, "| Bentuk y:", y.shape)
    return X, y


def scale_features(X: pd.DataFrame):
    subbanner("Normalisasi MinMaxScaler")
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    print("5 baris pertama setelah scaling:")
    print(X_scaled.head().to_string())
    print("\nMin per kolom:", X_scaled.min().round(2).to_dict())
    print("Max per kolom:", X_scaled.max().round(2).to_dict())
    return X_scaled, scaler


def run(df: pd.DataFrame):
    banner("3. TRANSFORMATION")
    df = drop_columns(df)
    df = encode_type(df)
    X, y = separate_xy(df)
    X_scaled, scaler = scale_features(X)
    return X_scaled, y, scaler


if __name__ == "__main__":
    from src import init_dataset, preprocessing
    from src.utils import quiet
    run(quiet(lambda: preprocessing.run(init_dataset.run())))

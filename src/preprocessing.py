import pandas as pd
import matplotlib.pyplot as plt

from src import config
from src.utils import banner, subbanner, save_fig


# ---------- Missing value ----------
def check_missing(df: pd.DataFrame) -> int:
    subbanner("Cek missing value")
    missing = df.isnull().sum()
    print(missing.to_string())
    total = int(missing.sum())
    print("Total missing value:", total)
    return total


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Numerik -> median, kategorikal -> modus (hanya berjalan bila ada missing)."""
    df = df.copy()
    for col in df.columns[df.isnull().any()]:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
    return df


# Duplikat
def check_duplicates(df: pd.DataFrame) -> pd.Series:
    """Mengecek baris data yang memiliki nilai duplikat."""
    subbanner("Cek data duplikat")
    mask = df.duplicated()
    print("Jumlah data duplikat:", int(mask.sum()))
    return mask


def handle_duplicates(df: pd.DataFrame, dup_mask: pd.Series) -> pd.DataFrame:
    if not dup_mask.any():
        print("Tidak ada duplikat, tidak ada tindakan.")
        return df
    df = df[~dup_mask].reset_index(drop=True)
    print("Duplikat dihapus. Ukuran data sekarang:", df.shape)
    return df


# ---------- Outlier (IQR) ----------
def compute_bounds(df: pd.DataFrame) -> dict:
    bounds = {}
    for col in config.NUMERIC_COLUMNS:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        bounds[col] = (q1 - config.IQR_FACTOR * iqr, q3 + config.IQR_FACTOR * iqr)
    return bounds


def outlier_mask(df: pd.DataFrame, col: str, bounds: dict) -> pd.Series:
    low, high = bounds[col]
    return (df[col] < low) | (df[col] > high)


def detect_outliers(df: pd.DataFrame, bounds: dict, judul: str) -> pd.DataFrame:
    subbanner(judul)
    rows = []
    for col in config.NUMERIC_COLUMNS:
        mask = outlier_mask(df, col, bounds)
        rows.append({
            "Kolom": col,
            "Batas bawah": round(bounds[col][0], 2),
            "Batas atas": round(bounds[col][1], 2),
            "Jumlah outlier": int(mask.sum()),
            "Di antaranya gagal": int(df.loc[mask, config.TARGET].sum()),
        })
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    return summary


def clip_outliers(df: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    """Handling sesuai MD: clipping ke batas IQR (baris tidak dihapus)."""
    df = df.copy()
    for col, (low, high) in bounds.items():
        df[col] = df[col].clip(lower=low, upper=high)
    return df


def plot_boxplot(df: pd.DataFrame, title: str, filename: str) -> None:
    df[config.NUMERIC_COLUMNS].plot(kind="box", subplots=True, layout=(1, 5),
                                    figsize=(15, 4), title=title)
    plt.tight_layout()
    save_fig(config.FIGURES_DIR / filename)


# ---------- Orkestrasi ----------
def run(df: pd.DataFrame) -> pd.DataFrame:
    banner("2. PREPROCESSING")
    df = df.copy()

    if check_missing(df) > 0:
        df = handle_missing(df)
        print("Missing value ditangani. Sisa missing:", int(df.isnull().sum().sum()))
    else:
        print("Tidak ada missing value, tidak ada tindakan.")

    df = handle_duplicates(df, check_duplicates(df))

    bounds = compute_bounds(df)
    detect_outliers(df, bounds, "Deteksi outlier (IQR, faktor %.1f)" % config.IQR_FACTOR)
    plot_boxplot(df, "Boxplot Sebelum Handling Outlier", "02_boxplot_sebelum.png")

    df = clip_outliers(df, bounds)
    detect_outliers(df, bounds, "Outlier setelah clipping (batas IQR yang sama)")
    plot_boxplot(df, "Boxplot Setelah Handling Outlier", "03_boxplot_sesudah.png")

    print("\nUkuran data setelah preprocessing:", df.shape)
    return df


if __name__ == "__main__":
    from src import init_dataset
    from src.utils import quiet
    run(quiet(init_dataset.run))

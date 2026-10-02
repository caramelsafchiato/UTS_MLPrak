import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src import config
from src.utils import banner, subbanner, save_fig


def load_dataset(path=config.DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def understand(df: pd.DataFrame) -> None:
    """Data understanding: ukuran, tipe, statistik, distribusi target & Type."""
    subbanner("Ukuran dan 5 baris pertama")
    print("Ukuran data (baris, kolom):", df.shape)
    print(df.head().to_string())

    subbanner("Info kolom dan tipe data")
    df.info()

    subbanner("Statistik deskriptif (kolom numerik)")
    print(df.describe().round(2).T.to_string())

    subbanner(f"Distribusi kelas {config.TARGET} (0 = normal, 1 = gagal)")
    counts = df[config.TARGET].value_counts()
    persen = (df[config.TARGET].value_counts(normalize=True) * 100).round(2)
    print(pd.DataFrame({"jumlah": counts, "persen": persen}).to_string())

    subbanner(f"Distribusi kolom {config.TYPE_COLUMN}")
    print(df[config.TYPE_COLUMN].value_counts().to_string())

    for col in config.LEAKAGE_COLUMNS:
        if col in df.columns:
            subbanner(f"Pemeriksaan kebocoran label: {col} vs {config.TARGET}")
            print(pd.crosstab(df[col], df[config.TARGET]).to_string())


def plot_target_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(5, 4))
    ax = sns.countplot(x=config.TARGET, data=df)
    ax.bar_label(ax.containers[0])
    plt.title("Distribusi Kelas Awal")
    save_fig(config.FIGURES_DIR / "01_distribusi_awal.png")


def run() -> pd.DataFrame:
    banner("1. INIT DATASET & DATA UNDERSTANDING")
    df = load_dataset()
    understand(df)
    plot_target_distribution(df)
    return df


if __name__ == "__main__":
    run()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src import config
from src.utils import banner, subbanner, save_fig


def split(X: pd.DataFrame, y: pd.Series):
    return train_test_split(X, y, test_size=config.TEST_SIZE,
                            random_state=config.RANDOM_STATE, stratify=y)


def report(X_train, X_test, y_train, y_test) -> None:
    subbanner(f"Ukuran data (test_size={config.TEST_SIZE}, "
              f"random_state={config.RANDOM_STATE}, stratify=y)")
    total = len(X_train) + len(X_test)
    print(f"X_train: {X_train.shape} ({len(X_train) / total:.0%})")
    print(f"X_test : {X_test.shape} ({len(X_test) / total:.0%})")
    print(f"y_train: {y_train.shape}")
    print(f"y_test : {y_test.shape}")
    for nama, y in [("y_train", y_train), ("y_test", y_test)]:
        subbanner(f"Distribusi {nama}")
        counts = y.value_counts().sort_index()
        persen = (y.value_counts(normalize=True).sort_index() * 100).round(2)
        print(pd.DataFrame({"jumlah": counts, "persen": persen}).to_string())


def plot_distribution(y_train, y_test) -> None:
    _, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, (judul, data) in zip(axes, [("y_train", y_train), ("y_test", y_test)]):
        sns.countplot(x=data, ax=ax)
        ax.bar_label(ax.containers[0])
        ax.set_title(f"Distribusi {judul}")
    plt.tight_layout()
    save_fig(config.FIGURES_DIR / "04_distribusi_split.png")


def save_baseline(X_train, X_test, y_train, y_test, directory=config.BASELINE_DIR) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(directory / "X_train.csv", index=False)
    X_test.to_csv(directory / "X_test.csv", index=False)
    y_train.to_csv(directory / "y_train.csv", index=False)
    y_test.to_csv(directory / "y_test.csv", index=False)
    print(f"[baseline tersimpan] {directory}")


def load_baseline(directory=config.BASELINE_DIR):
    """Dipakai Alisya & Nadine:
        from src.split_data import load_baseline
        X_train, X_test, y_train, y_test = load_baseline()
    """
    return (pd.read_csv(directory / "X_train.csv"),
            pd.read_csv(directory / "X_test.csv"),
            pd.read_csv(directory / "y_train.csv").squeeze("columns"),
            pd.read_csv(directory / "y_test.csv").squeeze("columns"))


def run(X: pd.DataFrame, y: pd.Series):
    banner("4. SPLIT DATA 80:20")
    X_train, X_test, y_train, y_test = split(X, y)
    report(X_train, X_test, y_train, y_test)
    plot_distribution(y_train, y_test)
    save_baseline(X_train, X_test, y_train, y_test)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    from src import init_dataset, preprocessing, transformation
    from src.utils import quiet
    X, y, _ = quiet(lambda: transformation.run(preprocessing.run(init_dataset.run())))
    run(X, y)

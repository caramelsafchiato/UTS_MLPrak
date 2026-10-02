"""Fungsi bantu kecil: banner output, simpan grafik, dan membisukan output tahap hulu."""
import contextlib
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # simpan ke file, tanpa membuka jendela
import matplotlib.pyplot as plt


def banner(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def subbanner(title: str) -> None:
    print("\n--- " + title + " ---")


def save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[grafik tersimpan] {path.name}")


def quiet(func, *args, **kwargs):
    """Jalankan func tanpa mencetak output. Dipakai agar saat satu tahap dijalankan
    sendiri, output tahap sebelumnya (hulu) tidak ikut tampil di screenshot."""
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)

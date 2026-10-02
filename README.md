# UTS ML Soal C – Baseline (Bagian Cindy)

Cakupan: Dataset → Data Understanding → Preprocessing → Transformation → Split 80:20.
**Tidak** mencakup resampling, Decision Tree, maupun evaluasi (dikerjakan anggota lain).

## Struktur
```
Project/
├── dataset/predictive_maintenance.csv
├── src/
│   ├── init_dataset.py    # Proses 1
│   ├── preprocessing.py   # Proses 2
│   ├── transformation.py  # Proses 3
│   ├── split_data.py      # Proses 4 (+ save/load baseline)
│   ├── config.py          # path, nama kolom, parameter (random_state=42, test_size=0.2)
│   └── utils.py           # banner print, simpan grafik, quiet()
├── main.py                # menjalankan pipeline
├── requirements.txt
└── output/                # dibuat otomatis: baseline/*.csv dan figures/*.png
```

## Dependency antar-proses
Setiap proses menerima hasil proses sebelumnya dan mengembalikan data untuk proses berikutnya:

| Proses | Fungsi `run()` | Input | Output (diteruskan ke) |
|---|---|---|---|
| 1. init_dataset | `run()` | file CSV | `df` mentah, semua 10 kolom → preprocessing |
| 2. preprocessing | `run(df)` | `df` mentah | `df` bersih (missing/duplikat dicek, outlier di-clipping; kolom masih lengkap) → transformation |
| 3. transformation | `run(df)` | `df` bersih | `X_scaled`, `y`, `scaler` (kolom ID & Failure Type dibuang, Type di-encode, MinMax) → split_data |
| 4. split_data | `run(X, y)` | `X_scaled`, `y` | `X_train, X_test, y_train, y_test` → **baseline** + disimpan ke `output/baseline/` |

## Menjalankan
```bash
pip install -r requirements.txt
python main.py                          # semua proses
python main.py --stage init             # hanya output proses 1
python main.py --stage preprocessing    # hanya output proses 2
python main.py --stage transformation   # hanya output proses 3
python main.py --stage split            # hanya output proses 4
```
Atau per modul: `python -m src.preprocessing`, `python -m src.transformation`, dst.
Pada mode satu-tahap, tahap sebelumnya tetap dijalankan di belakang layar (outputnya dibisukan),
sehingga layar hanya berisi output tahap yang dipilih dan siap di-screenshot.

## Memakai baseline (Alisya & Nadine)
```python
from src.split_data import load_baseline
X_train, X_test, y_train, y_test = load_baseline()
# Resampling HANYA pada X_train, y_train. X_test, y_test jangan disentuh.
```

# UTS ML Soal C – Baseline (Bagian Cindy)

Cakupan: Dataset → Data Understanding → Preprocessing → Transformation → Split 80:20
→ eksperimen oversampling dan evaluasi Decision Tree.

## Struktur
```
Project/
├── dataset/predictive_maintenance.csv
├── src/
│   ├── init_dataset.py    # Proses 1
│   ├── preprocessing.py   # Proses 2
│   ├── transformation.py  # Proses 3
│   ├── split_data.py      # Proses 4 (+ save/load baseline)
│   ├── oversampling.py    # BorderlineSMOTE, Decision Tree, dan evaluasi
│   ├── config.py          # path, nama kolom, parameter (random_state=42, test_size=0.2)
│   └── utils.py           # banner print, simpan grafik, quiet()
├── main.py                # menjalankan pipeline dan eksperimen oversampling
└── output/
    ├── baseline/          # data train/test sebelum resampling
    └── figures/           # grafik distribusi, split, dan confusion matrix
```

## Dependency antar-proses
Setiap proses menerima hasil proses sebelumnya dan mengembalikan data untuk proses berikutnya:

| Proses | Fungsi `run()` | Input | Output (diteruskan ke) |
|---|---|---|---|
| 1. init_dataset | `run()` | file CSV | `df` mentah, semua 10 kolom → preprocessing |
| 2. preprocessing | `run(df)` | `df` mentah | `df` bersih (missing/duplikat dicek, outlier di-clipping; kolom masih lengkap) → transformation |
| 3. transformation | `run(df)` | `df` bersih | `X_scaled`, `y`, `scaler` (kolom ID & Failure Type dibuang, Type di-encode, MinMax) → split_data |
| 4. split_data | `run(X, y)` | `X_scaled`, `y` | `X_train, X_test, y_train, y_test` → **baseline** + disimpan ke `output/baseline/` |
| 5. oversampling | `jalankan_eksperimen_oversampling()` | baseline dari `output/baseline/` | BorderlineSMOTE pada data training, Decision Tree, metrik evaluasi, dan grafik |

## Eksperimen oversampling
Saat seluruh pipeline dijalankan, data training diseimbangkan menggunakan
`BorderlineSMOTE(sampling_strategy=0.5, random_state=42)`. Artinya jumlah kelas
minoritas hasil resampling ditargetkan menjadi 50% dari jumlah kelas mayoritas.
Oversampling hanya diterapkan pada `X_train` dan `y_train`; `X_test` dan `y_test`
tetap menggunakan data asli untuk evaluasi.

Model yang dilatih adalah `DecisionTreeClassifier(criterion="entropy",
random_state=42)`. Hasil evaluasi menampilkan akurasi, presisi, recall, F1-score,
dan classification report di terminal. Grafik distribusi sebelum/sesudah
oversampling disimpan di `output/figures/distribusi_oversampling.png`, sedangkan
confusion matrix disimpan di `output/figures/confusion_matrix.png`.

## Menjalankan
```bash
pip install pandas seaborn matplotlib scikit-learn imbalanced-learn
python main.py                          # semua proses
python main.py --stage init             # hanya output proses 1
python main.py --stage preprocessing    # hanya output proses 2
python main.py --stage transformation   # hanya output proses 3
python main.py --stage split            # hanya output proses 4; menyimpan baseline
python -m src.oversampling               # menjalankan eksperimen dari baseline tersimpan
```
Atau per modul: `python -m src.preprocessing`, `python -m src.transformation`, dst.
Pada mode satu-tahap, tahap sebelumnya tetap dijalankan di belakang layar (outputnya dibisukan),
sehingga layar hanya berisi output tahap yang dipilih dan siap di-screenshot.
Eksperimen oversampling otomatis dijalankan setelah baseline pada mode default
(`python main.py`); menjalankan `--stage split` saja tidak menjalankan eksperimen tersebut.

## Memakai baseline (Alisya & Nadine)
```python
from src.split_data import load_baseline
X_train, X_test, y_train, y_test = load_baseline()
# Jika melakukan resampling sendiri, terapkan hanya ke X_train dan y_train.
# Jangan resample X_test atau y_test.
```

from pathlib import Path

# --- Path ---
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "dataset" / "predictive_maintenance.csv"
OUTPUT_DIR = ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
BASELINE_DIR = OUTPUT_DIR / "baseline"

# --- Parameter split (sesuai MD) ---
RANDOM_STATE = 42
TEST_SIZE = 0.2  # 80% training : 20% testing

# --- Kolom ---
TARGET = "Target"
TYPE_COLUMN = "Type"
ID_COLUMNS = ["UDI", "Product ID"]          # hanya identitas
LEAKAGE_COLUMNS = ["Failure Type"]          # bocoran label
DROP_COLUMNS = ID_COLUMNS + LEAKAGE_COLUMNS
NUMERIC_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# --- Transformasi (sesuai MD) ---
TYPE_MAPPING = {"L": 0, "M": 1, "H": 2}

# --- Outlier (sesuai MD: IQR + clipping) ---
IQR_FACTOR = 1.5

from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from src import config

def oversample(X_train, y_train):
    """Melakukan oversampling menggunakan BorderlineSMOTE."""
    smote = BorderlineSMOTE(sampling_strategy=0.5, random_state=config.RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res

def undersample(X_train, y_train):
    """Melakukan undersampling menggunakan RandomUnderSampler."""
    rus = RandomUnderSampler(sampling_strategy=1.0, random_state=config.RANDOM_STATE)
    X_res, y_res = rus.fit_resample(X_train, y_train)
    return X_res, y_res

# SCENARIOS dalam format tuple (Judul Deskriptif, Fungsi Resampling)
SCENARIOS = {
    "over": ("Oversampling (BorderlineSMOTE)", oversample),
    "under": ("Undersampling (RandomUnderSampler)", undersample),
}
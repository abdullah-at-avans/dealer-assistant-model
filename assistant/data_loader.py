import pandas as pd
from .config import DATASET_PATH, DATA_FILES
from .printer import log_info, log_error

def load_datasets():
    datasets = {}

    if len(DATA_FILES) == 0:
        log_error("No data files specified in DATA_FILES.")
        raise ValueError("No data files specified in DATA_FILES.")

    for i, fname in DATA_FILES.items():
        path = DATASET_PATH / fname
        log_info(f'Loading "{i}" From "{path}"`')
        datasets[i] = pd.read_csv(path, low_memory=False)
    return datasets

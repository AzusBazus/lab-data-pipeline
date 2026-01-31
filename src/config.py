import os
from pathlib import Path

# Get the base directory of the project (one level up from 'src')
BASE_DIR = Path(__file__).resolve().parent.parent

# Define paths
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
ARCHIVE_DIR = os.path.join(BASE_DIR, 'data', 'archive')

COLUMN_KEYWORDS = {
    "test_name": [
        "наименование", "тест", "parameter", "test", "name", 
        "исследование", "показатель", "analysis"
    ],
    "result": [
        "результат", "result", "value", "значение", 
        "концентрация", "res"
    ],
    "norm": [
        "норма", "norm", "reference", "ref. range", "range", 
        "референсные", "диапазон", "интервал"
    ],
    "unit": [
        "ед.изм", "unit", "units", "единицы", "dimension"
    ]
}

# Ensure these verify on run
print(f"Project Base: {BASE_DIR}")
print(f"Reading PDFs from: {RAW_DATA_DIR}")
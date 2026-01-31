import os
import re
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

NOISE_PATTERNS = [
    # Top of page: "30.01.2026, 15:17 Print page"
    re.compile(r'\d{2}\.\d{2}\.\d{4},\s+\d{2}:\d{2}.*Print\s+page', re.IGNORECASE),
    
    # Bottom of page: "192.168.68.249:5869... print.php 1/2"
    re.compile(r'.*print\.php.*', re.IGNORECASE),
    re.compile(r'.*print/print.*', re.IGNORECASE),
    
    # Page numbers alone like "1 / 2"
    re.compile(r'^\s*\d+\s*/\s*\d+\s*$', re.IGNORECASE)
]

# Ensure these verify on run
print(f"Project Base: {BASE_DIR}")
print(f"Reading PDFs from: {RAW_DATA_DIR}")
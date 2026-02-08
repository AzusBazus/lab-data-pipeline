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
        "исследование", "показатель", "analysis", "параметры",
        "анализ", 
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

PATIENT_FIELDS = {
    'name': ["ф.и.о.", "фамилия", "patient name", "full name", "фио"],
    'dob': ["дата рождения", "dob", "born", "birth", "рождения", "год рождения"],
    'report_date': ["дата", "date", "created"],
    'height': ["рост", "height", "length"],
    'weight': ["вес", "weight", "mass", "body weight"]
}

UNIT_SUFFIX_MAP = {
    "Seconds": ["sec", "сек", "с.", "seconds", "секунды"],
    "Percentage": ["%", "процент", "percent"],
    "INR": ["inr", "мно", "international normalized ratio"],
    "Ratio": ["ratio", "коэф", "отношение", "index"],
    "G/L": ["g/l", "г/л", "gram/liter"],
    "Count": ["count", "количество", "number", "10^9"]
}

# Regex to capture numeric values attached to time units
# Captures: "4", "4.5", "04"
# Keywords: мин, min, м, m (Minutes) | сек, sec, с, s (Seconds)
TIME_PATTERNS = {
    "minutes": re.compile(r'(\d+(?:[.,]\d+)?)\s*(?:мин|min|м\b|m\b)', re.IGNORECASE),
    "seconds": re.compile(r'(\d+(?:[.,]\d+)?)\s*(?:сек|sec|с\b|s\b)', re.IGNORECASE),
    "hours":   re.compile(r'(\d+(?:[.,]\d+)?)\s*(?:час|hour|ч\b|h\b)', re.IGNORECASE)
}

# Regex for finding dates with various separators (., -, /)
# Captures: 30.01.2026, 30-01-2026, 30/01/2026
DATE_PATTERN = re.compile(r'\b(\d{2})[./-](\d{2})[./-](\d{4})\b')

# Regex for finding years (19xx or 20xx)
YEAR_PATTERN = re.compile(r'\b(19|20)\d{2}\b')

HIDDEN_RESULT_KEYWORDS = ["посеве обнаружено", "рост микрофлоры", "detected"]

# Expanded based on:
# - Antibiotics (Эгамбердиева Мадина.docx)
# - Blood/Biochem/Hormones (Авазмухаммедова, Костенко, Тоштемиров, Хасанова)
# - Coagulation/Express Tests (Мовлонова)

LABEL_KEYWORDS = [
    # --- Existing ---
    "ЧУВСТВИТЕЛЬНОСТЬ", "АНТИБИОТИК", "SENSITIVITY", "ANTIBIOTIC", 
    "ANALYSIS", "RESULT", "РЕЗУЛЬТАТЫ", "ИССЛЕДОВАНИЯ",
    
    # --- General Categories ---
    "ОБЩИЙ", "КРОВИ", "ОАК", "BLOOD", "CBC",  # General Blood [cite: 20]
    "БИОХИМИЯ", "BIOCHEMISTRY",               # Biochemistry [cite: 27]
    "ЛИПИДНЫЙ", "СПЕКТР", "LIPID",            # Lipids [cite: 29]
    "УГЛЕВОДНЫЙ", "ОБМЕН", "CARBOHYDRATE",    # Carbs/Glucose [cite: 36]
    "ГОРМОНАЛЬНЫЕ", "ГОРМОН", "HORMONAL",     # Hormones [cite: 38]
    "МОЧИ", "ОАМ", "URINE",                   # Urine [cite: 40]
    "КОАГУЛОГРАММА", "СВЕРТЫВАНИЯ", "COAGULATION", "HEMOSTASIS", # Coagulation 
    "ВИТАМИНЫ", "VITAMINS",                   # Vitamins [cite: 49]
    "ЭКСПРЕСС", "ТЕСТ", "EXPRESS", "TEST",    # Rapid Tests [cite: 86]
    "МАРКЕР", "MARKER"                        # Common in other lab types
]

LABEL_NOISE_KEYWORDS = [
    # --- Standards & Versions ---
    "версия", "version", "eucast", 
    "комитет", "committee", 
    "год", "года", "year",

    # --- Printer & System Artifacts ---
    "print", "page", "страница", "лист",      # [cite: 15, 25]
    "192.168", "http", "www", ".uz",          # IP/URL [cite: 22, 19]
    
    # --- Lab Metadata (Header/Footer noise) ---
    "heartteam", "laboratories", "clinic",    # Lab Names 
    "узбекистан", "ташкент", "город",         # Address [cite: 18]
    "ферганский", "шоссе", "ул.", "street",   # Address [cite: 19]
    "тел:", "tel:", "факс", "phone",          # Contacts [cite: 19]
    
    # --- Staff / Signatures ---
    "врач", "doctor", "проводил", "анализ"    # [cite: 51]
]

ANTIBIOTIC_STUDY_KEYWORDS = ["антибиотик", "antibiotic", "eucast"]
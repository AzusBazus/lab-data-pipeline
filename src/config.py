import os
from pathlib import Path

# Get the base directory of the project (one level up from 'src')
BASE_DIR = Path(__file__).resolve().parent.parent

# Define paths
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
ARCHIVE_DIR = os.path.join(BASE_DIR, 'data', 'archive')

# Ensure these verify on run
print(f"Project Base: {BASE_DIR}")
print(f"Reading PDFs from: {RAW_DATA_DIR}")
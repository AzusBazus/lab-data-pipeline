from src.config import UNIT_SUFFIX_MAP, TIME_PATTERNS
import re

def expand_composite_rows(raw_rows):
    """Stage 1: Split multi-line rows."""
    expanded_rows = []
    for row in raw_rows:
        if _is_row_expandable(row):
            expanded_rows.extend(_split_and_suffix_row(row))
        else:
            expanded_rows.append(row)
    return expanded_rows

def _is_row_expandable(row):
    """
    Determines if a row contains multiple newline-separated values.
    Returns True only if the value column is present and has split-able content.
    """
    val_str = str(row.get('value', ''))
    
    # Must contain a newline AND have actual content after the split
    if '\n' in val_str:
        parts = [v for v in val_str.split('\n') if v.strip()]
        return len(parts) > 1
        
    return False

def _split_and_suffix_row(row):
    sub_rows = []
    
    # Split Values
    val_str = str(row.get('value', ''))
    values = [v.strip() for v in val_str.split('\n') if v.strip()]
    
    # Split Norms
    norm_str = str(row.get('norm', ''))
    norms = [n.strip() for n in norm_str.split('\n') if n.strip()]
    
    # Handle mismatches
    if len(norms) != len(values):
        aligned_norms = [norm_str] * len(values)
    else:
        aligned_norms = norms

    for i, val in enumerate(values):
        new_row = row.copy()
        
        # CRITICAL: Update both value AND text_value to the split part
        new_row['value'] = val
        new_row['text_value'] = val 
        new_row['norm'] = aligned_norms[i]
        
        # Suffix Generation
        suffix = _detect_unit_keyword(new_row['norm'])
        if suffix:
            new_row['test_name'] = f"{row['test_name']} ({suffix})"
        else:
            new_row['test_name'] = f"{row['test_name']} ({i+1})"
            
        sub_rows.append(new_row)
        
    return sub_rows

def infer_missing_units(rows):
    """Stage 2: Fill in missing units based on Value or Norm context."""
    for row in rows:
        # If unit already exists, skip (or you could normalize it here too)
        if row.get('unit'):
            continue

        # Strategy A: Check Value for clues (e.g., "5 min")
        # (This is where you'd add regex for "5 min" later)
        
        # Strategy B: Check Norm for clues
        norm_text = row.get('norm')
        found_unit = _detect_unit_keyword(norm_text)
        
        if found_unit:
            row['unit'] = found_unit

    return rows

def _detect_unit_keyword(text):
    """
    Shared Helper: Scans text against config to find a canonical unit.
    Used by both Row Expansion (for renaming) and Unit Inference (for filling).
    """
    if not text: return None
    text_lower = text.lower()
    
    for canonical_unit, keywords in UNIT_SUFFIX_MAP.items():
        for keyword in keywords:
            if keyword in text_lower:
                return canonical_unit
    return None

def normalize_result_values(rows):
    """
    Stage 3: Universal Value Cleaner.
    - Converts Time strings -> Seconds (float)
    - Converts Numeric strings -> Float
    - Handles "Greater/Less than" signs (> 5.0 -> 5.0)
    - Leaves non-numeric strings (like "negative") as None for the 'value' column
    """
    for row in rows:
        val_str = str(row.get('value', '')).strip()
        if not val_str: 
            row['value'] = None
            continue

        # --- A. Time Logic ---
        total_seconds = 0
        is_time_data = False
        
        # (Your existing time logic, slightly compacted)
        for unit, pattern in TIME_PATTERNS.items():
            match = pattern.search(val_str)
            if match:
                is_time_data = True
                val = float(match.group(1).replace(',', '.'))
                if unit == 'hours': total_seconds += val * 3600
                elif unit == 'minutes': total_seconds += val * 60
                elif unit == 'seconds': total_seconds += val
        
        if is_time_data:
            row['value'] = round(total_seconds, 2)
            row['unit'] = "Seconds"
            continue # Done with this row

        # --- B. Numeric Logic ---
        # 1. Clean symbols (<, >) and spaces
        # " > 5.5 " -> "5.5"
        clean_str = val_str.replace(',', '.').replace('>', '').replace('<', '').replace(' ', '')
        
        # 2. Try to convert to float
        try:
            # This regex allows negative numbers, decimals, but rejects "abc"
            if re.match(r'^-?\d+(?:\.\d+)?$', clean_str):
                row['value'] = float(clean_str)
            else:
                # It's a string like "negative" or "absent"
                # text_value keeps the info, value becomes None (so it doesn't break graphs)
                row['value'] = None
        except ValueError:
            row['value'] = None

    return rows
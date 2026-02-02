from src.config import UNIT_SUFFIX_MAP, TIME_PATTERNS

def expand_composite_rows(raw_rows):
    """
    Main entry point to fix rows where multiple results are crammed into one cell.
    """
    expanded_rows = []

    for row in raw_rows:
        # 1. Check Logic
        if _is_row_expandable(row):
            # 2. Split Logic
            new_sub_rows = _split_and_suffix_row(row)
            expanded_rows.extend(new_sub_rows)
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
    """
    Splits a composite row into multiple rows and appends unit suffixes to names.
    """
    sub_rows = []
    
    # Split Values
    val_str = str(row.get('value', ''))
    values = [v.strip() for v in val_str.split('\n') if v.strip()]
    
    # Split Norms (Try to align with values)
    norm_str = str(row.get('norm', ''))
    norms = [n.strip() for n in norm_str.split('\n') if n.strip()]
    
    # Safety: If norm counts don't match, replicate the full norm string for context
    # or align strictly if lengths match exactly.
    if len(norms) != len(values):
        aligned_norms = [norm_str] * len(values)
    else:
        aligned_norms = norms

    # Create new rows
    for i, val in enumerate(values):
        new_row = row.copy()
        new_row['value'] = val
        new_row['norm'] = aligned_norms[i]
        
        # Suffix Generation
        # We look at the specific NORM for this specific sub-result to guess the unit
        suffix = _detect_unit_keyword(new_row['norm'])
        
        if suffix:
            new_row['test_name'] = f"{row['test_name']} ({suffix})"
        else:
            # Fallback for unknown types (e.g. "Prothrombin Time (Part 1)")
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

def normalize_time_values(rows):
    """
    Stage 3: formatting "5 min 30 sec" strings into total seconds.
    Also handles comma replacement for plain numbers (14,7 -> 14.7).
    """
    for row in rows:
        val_str = str(row.get('value', '')).strip()
        if not val_str: continue

        # 1. Check for Time Patterns (Min/Sec strings)
        # We calculate total seconds: (Hours * 3600) + (Minutes * 60) + Seconds
        total_seconds = 0
        match_found = False

        # Check Hours
        h_match = TIME_PATTERNS['hours'].search(val_str)
        if h_match:
            total_seconds += float(h_match.group(1).replace(',', '.')) * 3600
            match_found = True

        # Check Minutes
        m_match = TIME_PATTERNS['minutes'].search(val_str)
        if m_match:
            total_seconds += float(m_match.group(1).replace(',', '.')) * 60
            match_found = True

        # Check Seconds
        s_match = TIME_PATTERNS['seconds'].search(val_str)
        if s_match:
            total_seconds += float(s_match.group(1).replace(',', '.'))
            match_found = True

        # If we found time data, update the row
        if match_found:
            row['value'] = round(total_seconds, 2) # Clean float
            row['unit'] = "Seconds" # Standardized unit
            continue # Skip to next row

        # 2. General Number Cleaning (if not a time string)
        # Fix "14,7" -> 14.7 for SQL compatibility
        if ',' in val_str and val_str.replace(',', '').replace('.', '').isdigit():
             try:
                 row['value'] = float(val_str.replace(',', '.'))
             except ValueError:
                 pass # Keep original string if conversion fails

    return rows
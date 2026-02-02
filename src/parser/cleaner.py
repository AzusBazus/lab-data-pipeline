from src.config import UNIT_SUFFIX_MAP

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
        suffix = _infer_suffix_from_text(new_row['norm'])
        
        if suffix:
            new_row['test_name'] = f"{row['test_name']} ({suffix})"
        else:
            # Fallback for unknown types (e.g. "Prothrombin Time (Part 1)")
            new_row['test_name'] = f"{row['test_name']} ({i+1})"
            
        sub_rows.append(new_row)
        
    return sub_rows

def _infer_suffix_from_text(text):
    """
    Scans text against the config map to find a canonical unit suffix.
    """
    if not text: return None
    text_lower = text.lower()
    
    # Iterate through our config map
    # Key = "Seconds", Keywords = ["sec", "сек", ...]
    for canonical_suffix, keywords in UNIT_SUFFIX_MAP.items():
        for keyword in keywords:
            if keyword in text_lower:
                return canonical_suffix
                
    return None
def flatten_hierarchical_table(table_data, inherited_parent=None):
    """
    Pre-processes a table to resolve hierarchical "Index/Category" columns.
    Handles 'Ghost Columns' where data shifts to Col 2 due to formatting.
    """
    if not table_data:
        return [], inherited_parent

    processed_data = []
    current_parent = inherited_parent
    
    # 1. Header Detection (Only skip if we explicitly see a header)
    start_idx = 0
    first_cell = str(table_data[0][0] or "").strip()
    
    # Check for "No" or "№" to identify header row
    if "no" in first_cell.lower() or "№" in first_cell:
        processed_data.append(table_data[0]) 
        start_idx = 1
    
    for row in table_data[start_idx:]:
        new_row = list(row)
        if len(new_row) < 2:
            processed_data.append(new_row)
            continue

        # --- GHOST COLUMN FIX ---
        # Get raw text
        col0 = str(new_row[0] or "").strip()
        col1 = str(new_row[1] or "").strip()
        
        # If Col 1 is empty, check Col 2 (Common PDF parsing quirk)
        # We assume the name is in Col 1 OR Col 2
        name_text = col1
        if not name_text and len(new_row) > 2:
            name_text = str(new_row[2] or "").strip()

        # LOGIC GATE 1: Reset (It's a Number)
        if col0.isdigit():
            current_parent = None
            # If we found the name in Col 2 (Ghost), move it to Col 1
            if not col1 and name_text:
                new_row[1] = name_text
                new_row[2] = "" # Clean up source

        # LOGIC GATE 2: New Parent (Text in Index Column)
        elif col0:
            current_parent = col0.replace(":", "").strip()
            
            if name_text:
                # Parent + Child
                new_row[1] = f"{current_parent} {name_text}"
                if not col1: new_row[2] = "" # Clean up ghost
            else:
                # Parent Only -> Move to Name Column
                new_row[1] = current_parent

        # LOGIC GATE 3: Continuation (Empty Index)
        elif not col0 and current_parent:
            if name_text:
                new_row[1] = f"{current_parent} {name_text}"
                if not col1: new_row[2] = "" # Clean up ghost

        processed_data.append(new_row)

    return processed_data, current_parent


def is_table_hierarchical(table_data):
    """
    Determines if a table is hierarchical based on the first row.
    """
    if not table_data:
        return False
    
    first_cell = str(table_data[0][0] or "").strip()
    return "no" in first_cell.lower() or "№" in first_cell
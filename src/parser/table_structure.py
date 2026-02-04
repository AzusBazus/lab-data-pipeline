import pandas as pd

def flatten_hierarchical_table(table_data, inherited_parent=None):
    """
    Pre-processes a table to resolve hierarchical "Index/Category" columns.
    Assumes TableNormalizer has already fixed column shifts/ghosts.
    """

    print("\n--- Original Table ---")
    print(pd.DataFrame(table_data))
    
    if not table_data:
        return [], inherited_parent

    processed_data = []
    current_parent = inherited_parent
    
    # 1. Header Detection
    start_idx = 0
    if table_data:
        first_cell = str(table_data[0][0] or "").strip()
        if "no" in first_cell.lower() or "№" in first_cell:
            processed_data.append(table_data[0]) 
            start_idx = 1
    
    for row in table_data[start_idx:]:
        new_row = list(row)
        if len(new_row) < 2:
            processed_data.append(new_row)
            continue

        # --- DATA PREP ---
        col0 = str(new_row[0] or "").strip()
        col1 = str(new_row[1] or "").strip()
        
        # CLEANUP: Handle explicit "nan" strings 
        if col0.lower() == 'nan': col0 = "" # <--- ADD THIS
        if col1.lower() == 'nan': col1 = ""

        # --- DIRECT MAPPING (The Fix) ---
        # We TRUST that TableNormalizer put the name in Col 1.
        # We DO NOT look at Col 2.
        name_text = col1

        # LOGIC GATE 1: Reset (It's a Number)
        if col0.isdigit():
            current_parent = None
            # If name_text is present, it's already in Col 1. Perfect.

        # LOGIC GATE 2: New Parent (Text in Index Column)
        elif col0:
            current_parent = col0.replace(":", "").strip()
            print("New Parent: " + current_parent)
            
            if name_text:
                # Parent + Child (e.g. "Epithelium Flat")
                new_row[1] = f"{current_parent} {name_text}"
                print("Parent: " + current_parent)
                print("Child: " + name_text)
                print("Parent + Child: " + new_row[1])
            else:
                # Parent Only (e.g. "Leukocytes") -> Move Parent to Name Col
                new_row[1] = current_parent

        # LOGIC GATE 3: Continuation (Empty Index)
        elif not col0 and current_parent:
            if name_text:
                new_row[1] = f"{current_parent} {name_text}"
            # If no name_text, it's likely a row of just values (which shouldn't happen here)
            # or a blank line. We leave it alone.

        processed_data.append(new_row)

    print("\n--- Flattened Table ---")
    print(pd.DataFrame(processed_data))
    
    return processed_data, current_parent


def is_table_hierarchical(table_data):
    """
    Determines if a table is hierarchical based on the first row.
    """
    if not table_data:
        return False
    
    first_cell = str(table_data[0][0] or "").strip()
    return "no" in first_cell.lower() or "№" in first_cell
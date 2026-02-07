import pandas as pd
import numpy as np
from src.config import COLUMN_KEYWORDS

class TableHandler:
    
    @staticmethod
    def clean_and_normalize(data: list) -> list:
        """
        Main entry point. Takes raw list-of-lists, fixes structure, returns list-of-lists.
        """
        if not data: return []

        # 1. Convert to DataFrame
        df = pd.DataFrame(data)
        
        # 2. Convert "None", empty strings, and whitespace to proper NaNs
        df = df.replace([None, 'None', '', r'^\s*$'], np.nan, regex=True)
        
        # 3. Trim Trailing Ghost Columns (Right-to-Left based on Header)
        df = TableHandler._trim_trailing_ghosts(df)
        
        # 4. Merge Complementary Columns (The "Zipper" Logic)
        df = TableHandler._merge_complementary_columns(df)
        
        # 5. Convert back to list of lists (replacing NaNs with None for compatibility)
        return df.where(pd.notnull(df), None).values.tolist()

    @staticmethod
    def _trim_trailing_ghosts(df: pd.DataFrame) -> pd.DataFrame:
        """
        Locates the header row, finds where valid text ends, and slices the DF.
        """
        # Find header index (simplified logic for DF)
        header_idx = None
        target_headers = COLUMN_KEYWORDS['result']
        
        for idx, row in df.iterrows():
            # Check if row contains any result keywords
            if row.astype(str).str.contains('|'.join(target_headers), case=False, na=False).any():
                header_idx = idx
                break
        
        if header_idx is None: return df

        # Get header row
        header_row = df.iloc[header_idx]
        last_valid_col = len(header_row) - 1
        
        # Scan Right-to-Left
        for c in range(len(header_row) - 1, -1, -1):
            if pd.notna(header_row[c]):
                last_valid_col = c
                break
                
        # Slice columns
        return df.iloc[:, :last_valid_col + 1]

    @staticmethod
    def _merge_complementary_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Scans adjacent columns. If Col A and Col B are complementary 
        (never have values in the same row), merge them.
        """
        # Iterate through columns backwards so we don't mess up indices while dropping
        # But for merging forward (Col 1 into Col 2), usually left-to-right is safer 
        # provided we restart check or manage indices carefully. 
        # Let's do a safe iterative approach.
        
        cols_to_drop = set()
        
        # We look at Col i and Col i+1
        for i in range(len(df.columns) - 1):
            if i in cols_to_drop: continue
            
            col_a = df.iloc[:, i]
            col_b = df.iloc[:, i+1]
            
            # 1. Check Intersection: Do they ever BOTH have data in the same row?
            # (We treat NaNs as empty)
            overlap = (col_a.notna() & col_b.notna()).any()
            
            if not overlap:
                # 2. Check Density: Don't merge two empty columns
                if col_a.isna().all() and col_b.isna().all():
                    continue

                # 3. Merge: Fill NaNs in A with values from B
                # combine_first does exactly this: A.combine_first(B)
                df.iloc[:, i] = col_a.combine_first(col_b)
                
                # Mark B for deletion
                cols_to_drop.add(i+1)
                
                print(f"🧩 Merged Column {i} and {i+1} (Complementary Pattern Found)")

        # Drop the absorbed columns
        if cols_to_drop:
            df.drop(df.columns[list(cols_to_drop)], axis=1, inplace=True)
            # Reset column numbers (0, 1, 2...)
            df.columns = range(df.shape[1])
            
        return df

    @staticmethod
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

    @staticmethod
    def is_table_hierarchical(table_data):
        """
        Determines if a table is hierarchical based on the first row.
        """
        if not table_data:
            return False
        
        first_cell = str(table_data[0][0] or "").strip()
        return "no" in first_cell.lower() or "№" in first_cell

    @staticmethod
    def demultiplex(df):
        """
        Detects 'Double-Wide' tables (Name, Value, Name, Value) and splits them 
        into two standard tables (Name, Value).
        
        Also injects standardized headers so the Interpreter maps them instantly.
        """
        if df is None or df.empty:
            return []
            
        # 1. Check if we need to split
        if TableHandler._is_demultiplexable(df):
            print("✂️  Detected Multi-Column Table. Demultiplexing...")
            
            # Create Standard Header Row
            # We use Russian keywords that we KNOW your config catches
            # ['Наименование', 'Результат']
            standard_header = pd.DataFrame([['Наименование', 'Результат']], columns=[0, 1])
            
            # --- SPLIT LEFT (Cols 0, 1) ---
            # We skip the original header (row 0) because it's usually merged/messy
            df_left_data = df.iloc[1:, [0, 1]].copy()
            df_left_data.columns = [0, 1] # Reset columns to match header
            
            # Concatenate: [Standard Header] + [Data]
            df_left = pd.concat([standard_header, df_left_data], ignore_index=True)
            
            # --- SPLIT RIGHT (Cols 2, 3) ---
            df_right_data = df.iloc[1:, [2, 3]].copy()
            df_right_data.columns = [0, 1]
            
            df_right = pd.concat([standard_header, df_right_data], ignore_index=True)
            
            return [df_left, df_right]
            
        # If not split, return original as single item list
        return [df]

    @staticmethod
    def _is_demultiplexable(df):
        """
        Heuristic: Is this a 4-column table where Cols 0 & 2 are text (Names) 
        and Cols 1 & 3 are short values (Results)?
        """
        # 1. dimensions Check
        if df.shape[1] < 4:
            return False

        # 2. Length Heuristic
        # We calculate avg length of NON-EMPTY strings in each column
        col_lengths = []
        for i in range(4):
            # Convert to string, strip whitespace
            s = df.iloc[:, i].astype(str).str.strip()
            # Ignore empty, 'nan', 'None'
            mask = (s != '') & (s.str.lower() != 'nan') & (s.str.lower() != 'none')
            valid_lens = s[mask].str.len()
            
            if len(valid_lens) == 0:
                col_lengths.append(0)
            else:
                col_lengths.append(valid_lens.mean())

        # Logic: 
        # Col 0 (Name) should be significantly longer than Col 1 (Value)
        # Col 2 (Name) should be significantly longer than Col 3 (Value)
        # We use a 1.5x multiplier as a safe margin.
        left_side_valid = col_lengths[0] > (col_lengths[1] * 1.5)
        right_side_valid = col_lengths[2] > (col_lengths[3] * 1.5)
        
        # Also check that Name columns actually have content (avg len > 3)
        names_have_content = col_lengths[0] > 3 and col_lengths[2] > 3
        
        return left_side_valid and right_side_valid and names_have_content
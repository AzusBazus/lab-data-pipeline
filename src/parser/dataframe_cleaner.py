import pandas as pd
import numpy as np
from src.config import COLUMN_KEYWORDS

class TableNormalizer:
    
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
        df = TableNormalizer._trim_trailing_ghosts(df)
        
        # 4. Merge Complementary Columns (The "Zipper" Logic)
        df = TableNormalizer._merge_complementary_columns(df)
        
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
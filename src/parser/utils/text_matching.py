from rapidfuzz import fuzz

def is_fuzzy_match(value, target, threshold=85):
    """
    Checks if 'value' is similar to 'target'.
    """
    if not value or not target:
        return False
    
    # 1. Exact Match (Fastest)
    if value.lower().strip() == target.lower().strip():
        return True
        
    # 2. Fuzzy Match (Slower, but handles typos)
    # partial_ratio handles substrings well (e.g. "Result:" vs "Result")
    score = fuzz.partial_ratio(value.lower(), target.lower())
    return score >= threshold

def find_best_match(value, candidates, threshold=85):
    """
    Finds the best matching string from a list of candidates.
    Returns the candidate name if found, else None.
    """
    if not value: return None
    
    # process.extractOne would be the library function, 
    # but we can do a simple loop for control
    best_score = 0
    best_match = None
    
    for candidate in candidates:
        score = fuzz.partial_ratio(value.lower(), candidate.lower())
        if score > best_score:
            best_score = score
            best_match = candidate
            
    if best_score >= threshold:
        return best_match
    return None
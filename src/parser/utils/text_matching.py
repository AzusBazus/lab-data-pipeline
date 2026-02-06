from rapidfuzz import fuzz

def is_fuzzy_match(text, keyword, threshold=90):
    if not text or not keyword:
        return False
    
    text = text.lower()
    keyword = keyword.lower()
    
    # 1. Exact Substring Check (Fastest)
    if keyword in text:
        return True
        
    # 2. Length Safety Check
    # If one string is significantly shorter than the other, partial_ratio is dangerous.
    # e.g. "a" vs "apple" -> partial_ratio is 100.
    if len(text) < 3 or len(keyword) < 3:
        return text == keyword # Force exact match for short strings

    # 3. Fuzzy Check
    # partial_ratio is good for "Patient Name:" vs "Name"
    # ratio is good for "Date" vs "Data"
    score = fuzz.partial_ratio(text, keyword)
    
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
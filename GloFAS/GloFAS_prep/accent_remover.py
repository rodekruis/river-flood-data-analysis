import unicodedata
def remove_accents(input_str):
    """Remove accents from a given string."""
    if isinstance(input_str, str):  # Only process strings
        nfkd_form = unicodedata.normalize('NFKD', input_str)
        return "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    return input_str  # Return as is if not a string
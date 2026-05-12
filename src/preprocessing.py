import re

def clean_text(text):
    import re
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()
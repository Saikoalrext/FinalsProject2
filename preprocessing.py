import re

STOPWORDS= {
    "is","are","the","in","on","at","and","or","to","of","a","an","from","it", "this", "that", "with", "for", "as", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "would", "could"
}

def is_valid_sentence(s):
    s_lower= s.lower()

    if len(s.split())< 6:
        return False
    if any(c.isdigit() for c in s[:10]) and ("citation" in s_lower or "manuscript" in s_lower):
        return False
    invalid= ["doi", "copyright", "correspondence", "citation", "manuscript", "revised", "reviewed", "received", "published", "creative commons", "license", "open-access", "email:", "eter", "vol.", "issue", "department", "accessed"]
    return not any(m in s_lower for m in invalid)
def clean_text(text):
    text= re.sub(r'\n+', ' ', text)
    text= re.sub(r'([.!?])\s*', r'\1 ', text)
    return text.strip()

def tokenize(text):
    return [t.lower() for t in re.findall(r'[a-zA-Z]+', text)]

def tokenize_clean(text):
    return [t for t in tokenize(text) if t not in STOPWORDS and len(t)> 2]

def split_sentences(text):
    text= re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
    sentences= re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip())> 20]
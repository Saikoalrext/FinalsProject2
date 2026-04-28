import re

STOPWORDS= {
    "is","are","the","in","on","at","and","or","to","of","a","an","from","it", "this", "that", "with", "for", "as", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "would", "could"
}

def is_valid_sentence(s):
    s_lower= s.lower()

    if len(s.split())< 6:
        return False
    if re.match(r'^\d+\s+[A-Z]', s) and any(x in s_lower for x in ['citation', 'manuscript']):
        return False
    if s.count('  ')> 2:
        return False
    invalid= ["doi", "copyright", "correspondence", "published", "revised", "reviewed", "received", "published", "creative commons", "license", "open-access", "email:", "vol.", "issue", "department", "accessed", "vol"]
    return not any(m in s_lower for m in invalid)

def clean_text(text):
    text= re.sub(r'\n+', ' ', text)
    text= re.sub(r'([.!?])\s*', r'\1 ', text)
    text= re.sub(r'(?<=[a-zA-Z])\n\s*\d+\s+(?=[A-Z])', ' ', text)
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

def sentence_quality_score(s):
    score= 0
    s_lower= s.lower()

    if re.search(r'^(fig|figure)\b', s_lower):
        score-= 20
    if s.count('(')>= 2 and s.count(')')>= 2:
        score -= 5
    if re.match(r'^\s*\(', s):
        score-= 10

    if any(w in s_lower for w in ['however', 'therefore', 'moreover', 'consequently']):
        score+= 3
    if len(s.split())> 15:
        score+= 2

    return score
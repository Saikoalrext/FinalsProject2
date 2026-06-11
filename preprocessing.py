import re
from nltk.stem import PorterStemmer

ps= PorterStemmer()

STOPWORDS= {
    "is","are","the","in","on","at","and","or","to","of","a","an","from","it", "this", "that", "with", "for", "as", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "would", "could", "by", "but"
}

ABBREVIATIONS= {
    'dr', 'mr', 'mrs', 'ms', 'prof', 'sr', 'jr', 'st', 'ave', 'blvd', 'rd', 'fig', 'figs', 'et', 'al', 'eg', 'ie', 'vs', 'vol', 'vols', 'inc', 'ltd', 'jr', 'sr', 'no', 'nos', 'pp', 'pg', 'pgs', 'ch', 'chs', 'sec', 'secs', 'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 'sir', 'lord', 'e.g', 'i.e', 'et al', 'ph.d', 'b.a', 'm.a', 'm.d', 'd.d.s', 'u.s', 'u.k', 'u.n', 'n.a', 's.a', 'ltd', 'inc', 'co', 'corp', 'a.m', 'p.m', 'a.d', 'b.c', 'c.e', 'b.c.e'
}

DECIMAL_PATTERN= re.compile(r'\d+\.\d+')

# LIST_ITEM_PATTERN= re.compile(r'(?::\s+|\n\s*|^\s*)([a-zA-Z]?\d+\.)\s+')

def protect_special_cases(text):
    PLACEHOLDER= '\uFFFF'

    # def repl(m):
    #     return m.group(1)+ m.group(2).replace('.', PLACEHOLDER)+ m.group(3)
    
    # text= re.sub(r'(:)(\s*\d+\.)(\s+)', repl, text)
    # text= re.sub(r'(:)(\s*[a-z]\.)(\s+)', repl, text, flags=re.IGNORECASE)

    def protect_decimal(m):
        return m.group(0).replace('.', PLACEHOLDER)

    text= DECIMAL_PATTERN.sub(protect_decimal, text)

    for abbr in ABBREVIATIONS:
        pattern= rf'\b({abbr})\.\s+'
        text= re.sub(pattern, lambda m: m.group(1)+ PLACEHOLDER+ ' ', text, flags=re.IGNORECASE)
    
    def protect_list_item(m):
        return m.group(0).replace('.', PLACEHOLDER)
    
    LIST_ITEM_PATTERN= re.compile(r'(?::\s+|,\s+|\n\s*|^\s*)([a-zA-Z]?\d+\.)\s+')

    prev=  None
    while prev!= text:
        prev= text
        text= LIST_ITEM_PATTERN.sub(
            lambda m: m.group(0).replace('.', PLACEHOLDER, 1), text
        )
    
    # text= LIST_ITEM_PATTERN.sub(protect_list_item, text)

    return text, PLACEHOLDER

def restore_placeholders(sentences, placeholder):
    return [s.replace(placeholder, '.').strip() for s in sentences]

def is_valid_sentence(s):
    s_lower= s.lower()

    if len(s.split())< 4:
        return False
    if re.search(r'\b\d+\s*[-–—]\s*\d+\b', s):
        return False
    if re.search(r'\b\d+\s*\(\d+\)', s): 
        return False
    if re.match(r'^\d+\s+[A-Z]', s) and any(x in s_lower for x in ['citation', 'manuscript']):
        return False
    invalid= ["doi", "copyright", "correspondence", "published", "revised", "reviewed", "received", "published", "creative commons", "license", "open-access", "email:", "vol.", "issue", "department", "accessed", "vol", "keywords"]
    return not any(m in s_lower for m in invalid)

def clean_text(text):
    text= re.sub(r'\n+', ' ', text)
    text= re.sub(r'([.!?])\s*', r'\1 ', text)
    text= re.sub(r'(?<=[a-zA-Z])\n\s*\d+\s+(?=[A-Z])', ' ', text)
    return text.strip()

def tokenize(text):
    return [t.lower() for t in re.findall(r'[a-zA-Z]+', text)]

def tokenize_clean(text):
    # return [t for t in tokenize(text) if t not in STOPWORDS and len(t)> 2]
    return [
        ps.stem(t)
        for t in tokenize(text) if t not in STOPWORDS and len(t)> 2
    ]

def split_sentences(text):
    if not text or not text.strip():
        return []
    
    text, placeholder= protect_special_cases(text)

    text= re.sub(r'\s+', ' ', text)
    text= re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
    text= text.replace('...', placeholder* 3)

    raw_sentences= re.split(r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$', text)
    raw_sentences= [s.replace(placeholder* 3, '...') for s in raw_sentences]

    sentences= restore_placeholders(raw_sentences, placeholder)
        
    # text= re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
    # sentences= re.split(r'(?<=[.!?])\s+', text)
    # return [s.strip() for s in sentences if len(s.strip())> 20]
    return [s for s in sentences if len(s.strip()) > 20]

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
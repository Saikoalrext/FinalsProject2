import math
from collections import Counter

def build_stats(tokenized_docs):
    N= len(tokenized_docs)

    df= {}
    lengths= []

    for doc in tokenized_docs:
        lengths.append(len(doc))
        for term in set(doc):
            df[term]= df.get(term, 0)+ 1

    avgdl= sum(lengths)/ N if N else 0

    return df, lengths, avgdl, N

def bm25_score(query_tokens, doc_tokens, df, N, avgdl, k1= 1.2, b= 0.2):
    tf= Counter(doc_tokens)
    score= 0
    dl= len(doc_tokens)

    for term in query_tokens:
        if term not in tf or term not in df:
            continue

        # idf= math.log((N- df[term]+ 0.5)/(df[term]+ 0.5))+ 1

        idf= math.log((N+ 1)/ (df[term]+ 1))+ 1

        f= tf[term]

        denom= f+ k1* (1- b+ b* (dl/avgdl))
        
        score+= idf* (f* (k1+ 1))/ denom
    
    return score
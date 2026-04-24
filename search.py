from collections import Counter
import math
from preprocessing import tokenize_clean
from similarity import cosine

def build_query_vector(query, df, N):
    tokens= tokenize_clean(query)
    tf= Counter(tokens)
    vec= {}

    for term, count in tf.items():
        if term in df:
            idf= math.log((N+ 1)/ (df[term]+ 1))+ 1
            vec[term]= count* idf

    return vec

def search(query, vectors, df, N, top_k=3):
    q_vec= build_query_vector(query, df, N)

    scores= []
    for i, doc_vec in enumerate(vectors):
        score = cosine(q_vec, doc_vec)
        scores.append((i, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]
from collections import Counter
import math
import re
from preprocessing import tokenize_clean, sentence_quality_score, clean_text, is_valid_sentence, split_sentences
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

def search(query, vectors, df, N, docs, top_k=3):
    q_vec= build_query_vector(query, df, N)

    scores= []

    for i, doc_vec in enumerate(vectors):
        base_scores= cosine(q_vec, doc_vec)
        quality= sentence_quality_score(docs[i])* 0.01
        scores.append((i, max(0, base_scores+ quality)))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]
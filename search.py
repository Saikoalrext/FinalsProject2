from collections import Counter
import math
from preprocessing import tokenize_clean, sentence_quality_score, clean_text, is_valid_sentence, split_sentences
from similarity import cosine
from bm25 import build_stats, bm25_score

def build_query_vector(query, df, N):
    tokens= tokenize_clean(query)
    tf= Counter(tokens)
    vec= {}

    for term, count in tf.items():
        if term in df:
            idf= math.log((N+ 1)/ (df[term]+ 1))+ 1
            vec[term]= count* idf

    return vec

# def search(query, vectors, df, N, docs, top_k=3):
#     q_vec= build_query_vector(query, df, N)

#     scores= []

#     for i, doc_vec in enumerate(vectors):
#         base_scores= cosine(q_vec, doc_vec)
#         quality= sentence_quality_score(docs[i])* 0.01
#         scores.append((i, max(0, base_scores+ quality)))

#     scores.sort(key=lambda x: x[1], reverse=True)

#     threshold= 0.05
#     valid_scores= [(i, s) for i, s in scores if s> threshold]
#     if not valid_scores:
#         return []
    
#     return valid_scores[:top_k]

def search(query, vectors, df, N, docs, top_k= 5):
    query_tokens= tokenize_clean(query)

    tokenized_docs= [tokenize_clean(d) for d in docs]

    bm_df, lengths, avgdl, N= build_stats(tokenized_docs)

    q_vec= build_query_vector(query, df, N)

    if not any(t in bm_df for t in query_tokens):
        return []

    scores= []

    for i, docs_tokens in enumerate(tokenized_docs):
        base= bm25_score(query_tokens, docs_tokens, df, N, avgdl, k1= 1.2, b=0.2)
        cos= cosine(q_vec, vectors[i])
        # quality= (sentence_quality_score(docs[i])* 0.01)

        score= 0.7*base+ 0.3*cos

        if len(query_tokens)== 1:
            score+= (0.3* docs_tokens.count(query_tokens[0]))

        scores.append((i, score))
    
    if all(score== 0 for _, score in scores):
        return[]
    
    scores.sort(key=lambda x: x[1], reverse=True)

    threshold= 0.01

    valid= [(i,s) for i, s in scores if s> threshold]

    return valid[:top_k]
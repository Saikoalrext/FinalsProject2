import math
from preprocessing import split_sentences, tokenize_clean
from tfidf import compute_tfidf
from similarity import cosine

def build_similarity_matrix(vectors):
    n= len(vectors)
    sim= [[0.0]*n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i!= j:
                sim[i][j]= cosine(vectors[i], vectors[j])

    return sim

def textrank(sim, iterations= 20, d= 0.85, position_weights= None):
    n= len(sim)
    scores= [1.0]* n

    if position_weights:
        for i in range(n):
            scores[i]*= position_weights[i]
    
    for i in range(n):
        pos= i/ max(n-1, 1)
        if pos< 0.2:
            scores[i]*= 1.3
        elif pos> 0.8:
            scores[i]*= 0.7
        elif 0.3< pos< 0.7:
            scores[i]*= 1.1

    for _ in range(iterations):
        new_scores= [0.0]* n
        for i in range(n):
            total= 0
            for j in range(n):
                if sim[j][i]> 0:
                    norm= sum(sim[j])
                    if norm!= 0:
                        total+= (sim[j][i]/norm)* scores[j]
            new_scores[i]= (1-d)+ d*total
        scores= new_scores

    return scores

def compute_sentence_importance(tokenized_doc, df, N):
    if not tokenized_doc:
        return 0
    
    score= 0
    for term in set(tokenized_doc):
        if term in df:
            idf= math.log((N+ 1)/ (df[term]+ 1))+ 1
            tf= tokenized_doc.count(term)
            score+= tf* idf
        
    return score/ len(tokenized_doc)

def query_boost(sentence_tokens, query_tokens):
    if not query_tokens:
        return 1.0
    sent_set= set(sentence_tokens)
    query_set= set(query_tokens)
    overlap= len(sent_set& query_set)
    return 1+ (1.5** overlap)

def summarize(text, top_k=3, vectors= None, df= None, N= None, query= None):
    sentences= split_sentences(text)
    print(f"Debug: Found {len(sentences)} sentences to summarize")

    if len(sentences)<= top_k:
        return sentences
    
    tokenized= [tokenize_clean(s) for s in sentences]

    valid= [(i, s, t) for i, (s, t) in enumerate(zip(sentences, tokenized)) if len(t)> 0]

    if len(valid)< top_k:
        return [s for _, s, _ in valid]

    indices, sentences_filtered, tokenized_filtered= zip(*valid)
    indices= list(indices)


    query_tokens= tokenize_clean(query) if query else []

    query_matching_indices= []
    for i, tokens in enumerate(tokenized_filtered):
        if set(tokens)& set(query_tokens):
            query_matching_indices.append(i)

    print(f"Deug: {len(query_matching_indices)} sentences contain query terms")

    vectors, df, N= compute_tfidf(tokenized_filtered)
    sim= build_similarity_matrix(vectors)
    scores= textrank(sim)

    final_scores= []
    for i in range(len(indices)):
        tr= scores[i]

        imp= 1.0
        if df and N:
            imp= 1+ compute_sentence_importance(tokenized_filtered[i], df, N)

        boost= query_boost(tokenized_filtered[i], query_tokens)

        final_scores.append(tr* imp* boost)

    ranked= list(enumerate(final_scores))
    ranked.sort(key=lambda x: x[1], reverse= True)

    selected= []
    ranked_indices= [i for i, _ in ranked]

    if query_matching_indices:
        best_match= max(query_matching_indices, key=lambda idx:final_scores[idx])
        selected.append(indices[best_match])
        ranked_indices= [i for i in ranked_indices if i!= best_match]
    
    for i in ranked_indices:
        if len(selected)>= top_k:
            break
        if indices[i] not in selected:
            selected.append(indices[i])

    selected.sort()
    return [sentences[i] for i in selected]
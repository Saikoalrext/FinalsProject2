import math
from collections import Counter

def compute_df(tokenized_docs):
    df= {}
    for doc in tokenized_docs:
        for term in set(doc):
            df[term]= df.get(term, 0)+ 1
    return df

def compute_tfidf(tokenized_docs):
    df= compute_df(tokenized_docs)
    N= len(tokenized_docs)

    vectors= []
    for doc in tokenized_docs:
        tf= Counter(doc)
        vec= {}

        for term, count in tf.items():
            idf= math.log((N + 1)/ (df[term] +1))+ 1
            vec[term]= count* idf

        vectors.append(vec)

    return vectors, df, N
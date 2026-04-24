from pdf_reader import extract_pdf_text
from preprocessing import tokenize_clean, clean_text, split_sentences, is_valid_sentence
from tfidf import compute_tfidf
from search import search
from textrank import summarize

pdf_path= "paper1.pdf"
text= extract_pdf_text(pdf_path)
text= clean_text(text)

if not text.strip():
    print("Empty or failed PDF extraction")
    exit()

docs= [s for s in split_sentences(text) if is_valid_sentence(s)]

if not docs:
    print("No valid sentences found after filtering")
    exit()

tokenized_docs= [tokenize_clean(d) for d in docs]
vectors, df, N= compute_tfidf(tokenized_docs)

query= input("Enter search query: ")
results= search(query, vectors, df, N)

print("Search Results:")
for i, score in results:
    print(f"Doc {i}: {score:.4f} -> {docs[i][:100]}...")

if not results or results[0][1]== 0:
    print("No relevant results found")
    exit()

top_idx= results[0][0]

window= int(input("How many sentence to summarize: "))
start= max(0, top_idx- window)
end= min(len(docs), top_idx+ window+ 1)
context= " ".join(docs[start:end])

summary= summarize(context, top_k= window, query= query)

print("\nSummary:")
for s in summary:
    print("-", s)

print("Total sentences:", len(summary))
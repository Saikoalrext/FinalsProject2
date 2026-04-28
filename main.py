import os
import sys

from pdf_reader import extract_pdf_text
from preprocessing import tokenize_clean, clean_text, split_sentences, is_valid_sentence, sentence_quality_score
from tfidf import compute_tfidf
from search import search
from textrank import summarize

def select_pdf():
    pdfs= []
    for root, dirs, files in os.walk('.'):
        dirs[:]= [d for d in dirs if not d.startswith('.') and d!= 'venv']
        for f in files:
            if f.lower().endswith('.pdf'):
                pdfs.append(os.path.join(root,f))
    pdfs= [p[2:] if p.startswith('./') else p for p in pdfs]
    if not pdfs:
        path= input("No PDFs found. enter path to PDF: ").strip()
        return path if os.path.exists(path) else None
    
    print(f"\nFound {len(pdfs)} PDF file(s):")
    for i, p in enumerate(pdfs, 1):
        print(f"  {i}. {p}")
    
    choice= input("\nEnter number or full path: ").strip()

    if choice.isdigit():
        idx= int(choice)- 1
        if 0<= idx< len(pdfs):
            return pdfs[idx]
        
    if os.path.exists(choice):
        return choice
    
    if not choice.lower().endswith('.pdf'):
        choice+= '.pdf'
        if os.path.exists(choice):
            return choice
        
    return None

pdf_path= select_pdf()

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
results= search(query, vectors, df, N, docs)

print("Search Results:")
for i, score in results:
    print(f"Doc {i}: {score:.4f} -> {docs[i][:100]}...")

if not results or results[0][1]== 0:
    print("No relevant results found")
    exit()

top_idx= results[0][0]

context_size= int(input("How many sentences to consider for summary: "))

context_indices= set()
for idx, score in results:
    if score> 0:
        for j in range(max(0, idx- 1), min(len(docs), idx+ 2)):
            context_indices.add(j)
        if len(context_indices)>= context_size* 2:
            break

context_indices= sorted(context_indices)
context= " ".join([docs[i] for i in context_indices])

print(f"\nBuilt context from {len(context_indices)} sentences \n")

# window= int(input("How many sentences around top result to consider: "))
# start= max(0, top_idx- window)
# end= min(len(docs), top_idx+ window+ 1)
# context= " ".join(docs[start:end])

summary= summarize(context, top_k= context_size, query= query)

print("\nSummary:")
for s in summary:
    print("-", s)

print("Total sentences:", len(summary))
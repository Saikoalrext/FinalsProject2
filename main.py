import os

from pdf_reader import extract_pdf_text
from preprocessing import tokenize_clean, clean_text, split_sentences, is_valid_sentence, sentence_quality_score
from tfidf import compute_tfidf
from search import search
from textrank import summarize

def select_pdf():
    pdfs= []
    pdfholder= ["hehe"]
    for root, dirs, files in os.walk('.'):
        dirs[:]= [d for d in dirs if not d.startswith('.') and d!= 'venv']
        for f in files:
            if f.lower().endswith('.pdf'):
                pdfs.append(os.path.join(root,f))
    pdfs= [p[2:] if p.startswith('./') else p for p in pdfs]
    if not pdfs:
        path= input("\nNo PDFs found. enter path to PDF: ").strip()
        return path if os.path.exists(path) else None
    
    while True:
        print(f"\n****************** PDF Files ******************\n\nFound {len(pdfs)} PDF file(s):")
        for i, p in enumerate(pdfs, 1):
            pdfholder.append(p)
            if len(p)< 34:
                print(f"  {i}. {p}")
            elif len(p)>= 34:
                print(f"  {i}. {p[:34]}... .pdf")
        print(f"\n***********************************************")
        choice1= input("\n1. Pick\n2. Full name\n0. Exit\n\nInput: ")
        if choice1== "0":
            return None
        elif not choice1.isdigit():
            print("Input a number.")
            continue
        elif choice1== "1":
            choice= input("\nEnter number or full path: ").strip()
            if not choice:
                print("No selection made")
                return None

            if choice.isdigit():
                idx= int(choice)- 1
                if 0<= idx< len(pdfs):
                    return pdfs[idx]
                else:
                    print(f"Invalid selection: {choice}")
                    return None
                
            if os.path.exists(choice):
                return choice
            
            if not choice.lower().endswith('.pdf'):
                choice+= '.pdf'
                if os.path.exists(choice):
                    return choice
                
            print(f"File not found: {choice}")
            continue
        elif choice1== "2":
            choice2= input("\nEnter number: ")
            if not choice2.isdigit():
                print("Input 1 or 2")
                continue
            c2holder= int(choice2)
            print(f"  - {pdfholder[c2holder]}")

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
if not any(tokenized_docs):
    print("All sentences filtered out during tokenization")
    exit(1)
vectors, df, N= compute_tfidf(tokenized_docs)

# query= input("\nEnter search query (\"exit program\" to quit): ").strip()

def safe_search(query, vectors, df, N, docs, tokenized_docs):
    if not query or not query.strip():
        return []
    if query.lower().strip()== "exit program":
        exit()
    return search(query, vectors, df, N, docs, tokenized_docs)

def get_results_with_retry(docs, vectors, df, N, tokenized_docs):
    while True:
        query= input("\nEnter search query (\"exit program\" to quit): ").strip()
        results= safe_search(query, vectors, df, N, docs, tokenized_docs)
        if results and results[0][1]> 0:
            print("\nSearch Results:")
            for i, score in results:
                print(f"Doc {i}: {score:.4f} -> {docs[i][:100]}...")
            return results, query
        print("No relevant results found.")

results, query= get_results_with_retry(docs, vectors, df, N, tokenized_docs)

top_idx= results[0][0]

context_size= 99
# context_size= max(1, int(input("How many sentences to consider for summary: ")))

context_indices= set()
for idx, score in results:
    if score> 0:
        for j in range(max(0, idx- 1), min(len(docs), idx+ 2)):
            context_indices.add(j)
        if len(context_indices)>= context_size* 2:
            break

context_indices= sorted(context_indices)
context_sentences= []
context= " ".join([docs[i] for i in context_indices])

# print(f"\nBuilt context from {len(context_indices)} sentences \n")

# window= int(input("How many sentences around top result to consider: "))
# start= max(0, top_idx- window)
# end= min(len(docs), top_idx+ window+ 1)
# context= " ".join(docs[start:end])

summary= summarize(context, top_k= context_size, query= query)

# print("\nSummary:")
# for s, scores in enumerate(summary, 1):
#     print(f"{idx}. [{scores:.4f}] {s}")
# # for s, t in summary:
# #     print("-", "[", t, "]", s)
#     print(" ")
print("\nContext:")
summary_map= {}
placeholder= ["hehe"]
numberholder= [0]

for idx, (s, score, i) in enumerate(summary, 1):
    print(f"{idx}. [{score:.4f}] {s[:100]}...")
    print(" ")
    doc_idx= context_indices[i]
    summary_map[idx]= (i, doc_idx)
    placeholder.append(docs[doc_idx])
    numberholder.append(score)

while True:
    choice1= input(f"\n*********************\n\n1. Summarize\n2. Show full sentence\n3. Show whole paragraph\n00. Preview\n0. Exit\n\n*********************\n\nInput: ").strip()
    if choice1== "0":
        break
    elif choice1== "00":
        print("\nContext:")
        for idx, (s, score, i) in enumerate(summary, 1):
            print(f"{idx}. [{score:.4f}] {s[:100]}...")
            print(" ")
        continue
    elif choice1== "1":
        print(" ")
        print(f"{'*'*60}")
        for i in range(len(summary)):
            if numberholder[i]>= 5.0:
                print(f"    {placeholder[i]}")
        print(f"{'*'*60}")
        continue
    elif choice1== "2":
        choice2= input(f"\nWhich sentence (1-{len(summary)}): ").strip()
        if not choice2.isdigit():
            print("Input a number.")
            continue 
        holder= int(choice2)
        print(f"- {placeholder[holder]}")
    elif choice1== "3":
        choice= input(f"\nWhich sentence (1-{len(summary)}): ").strip()
        if not choice.isdigit():
            print("Input a number.")
            continue

        num= int(choice)
        if num< 1 or num> len(summary):
            print(f"Input 1-{len(summary)}")
            continue

        _,  doc_idx= summary_map[num]

        start= max(0, doc_idx- 5)
        end= min(len(docs), doc_idx+ 6)

        print(f"\n{'*'*60}")
        for i in range(start, end):
            prefix= ">>> " if i== doc_idx else "    "
            print(f"{prefix}{docs[i]}")
        print(f"{'*'*60}")

# while True:
#     choice= input(f"\nShow the whole paragraph (1-{len(summary)}), (99) to preview, (0) to exit: ").strip()
#     if choice== "0":
#         break
#     elif choice== "99":
#         print("\nContext:")
#         for idx, (s, score, i) in enumerate(summary, 1):
#             print(f"{idx}. [{score:.4f}] {s[:100]}...")
#             print(" ")
#         continue
#     elif not choice.isdigit():
#         print("Input a number.")
#         continue

#     num= int(choice)
#     if num< 1 or num> len(summary):
#         print(f"Input 1-{len(summary)}")
#         continue

#     _,  doc_idx= summary_map[num]

#     start= max(0, doc_idx- 5)
#     end= min(len(docs), doc_idx+ 6)

#     print(f"\n{'='*60}")
#     for i in range(start, end):
#         prefix= ">>> " if i== doc_idx else "    "
#         print(f"{prefix}{docs[i]}")
#     print(f"{'='*60}")
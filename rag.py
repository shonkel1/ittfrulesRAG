"""
Simple RAG for ITTF Rules.
No FAISS - uses numpy (avoids segfault on Mac).
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from groq import Groq

# ======================
# CONFIG
# ======================
CHUNKS_PATH = "ittf_chunks.json"
INDEX_DIR = "rag_index"
EMBEDDING_MODEL = "all-mpnet-base-v2"  # Change to "ittf-finetuned-model" after training
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ======================
# LOAD CHUNKS
# ======================
print("Loading chunks...")
with open(CHUNKS_PATH, "r") as f:
    chunks = json.load(f)

chunk_texts = [c["text"] for c in chunks]
chunk_ids = [c["chunk_id"] for c in chunks]
print(f"Loaded {len(chunks)} chunks")

# ======================
# BUILD OR LOAD INDEX
# ======================
data_path = os.path.join(INDEX_DIR, "data.pkl")

if os.path.exists(data_path):
    print("Loading existing index...")
    with open(data_path, "rb") as f:
        data = pickle.load(f)
        embeddings = data["embeddings"]
        bm25 = data["bm25"]
    print("Index loaded!")
else:
    print(f"Building index with {EMBEDDING_MODEL}...")

    # Load embedding model
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    # Generate embeddings for all chunks
    print("Generating embeddings...")
    embeddings = embedding_model.encode(
        chunk_texts,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    # Build BM25 index
    print("Building BM25 index...")
    tokenized = [text.lower().split() for text in chunk_texts]
    bm25 = BM25Okapi(tokenized)

    # Save index
    os.makedirs(INDEX_DIR, exist_ok=True)
    with open(data_path, "wb") as f:
        pickle.dump({"embeddings": embeddings, "bm25": bm25}, f)
    print(f"Index saved to {INDEX_DIR}/")

# ======================
# LOAD MODELS
# ======================
print(f"Loading embedding model: {EMBEDDING_MODEL}")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

print(f"Loading reranker: {RERANKER_MODEL}")
reranker = CrossEncoder(RERANKER_MODEL)

# ======================
# QUERY LOOP
# ======================
print("\n" + "="*50)
print("ITTF Rules RAG - Type 'quit' to exit")
print("="*50 + "\n")

while True:
    query = input("Question: ").strip()
    if query.lower() in ["quit", "exit", "q"]:
        break
    if not query:
        continue

    # ----- STEP 1: Embed query -----
    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )

    # ----- STEP 2: Dense search (numpy dot product) -----
    similarities = np.dot(embeddings, query_embedding.T).flatten()
    top_dense_indices = np.argsort(similarities)[::-1][:40]
    dense_results = [(int(i), float(similarities[i])) for i in top_dense_indices]

    # ----- STEP 3: Sparse search (BM25) -----
    query_tokens = query.lower().split()
    bm25_scores = bm25.get_scores(query_tokens)
    top_bm25_indices = np.argsort(bm25_scores)[::-1][:40]
    sparse_results = [(int(i), float(bm25_scores[i])) for i in top_bm25_indices]

    # ----- STEP 4: Combine with RRF -----
    k = 60  # RRF constant
    combined_scores = {}

    for rank, (idx, _) in enumerate(dense_results):
        combined_scores[idx] = combined_scores.get(idx, 0) + 0.7 / (k + rank + 1)

    for rank, (idx, _) in enumerate(sparse_results):
        combined_scores[idx] = combined_scores.get(idx, 0) + 0.3 / (k + rank + 1)

    # Sort by combined score
    hybrid_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:20]

    # ----- STEP 5: Rerank -----
    pairs = [(query, chunk_texts[idx]) for idx, _ in hybrid_results]
    rerank_scores = reranker.predict(pairs)

    reranked = list(zip([idx for idx, _ in hybrid_results], rerank_scores.tolist()))
    reranked.sort(key=lambda x: x[1], reverse=True)
    top_5 = reranked[:5]

    # ----- STEP 6: Show retrieved chunks -----
    print("\nRetrieved chunks:")
    retrieved_chunks = []
    for i, (idx, score) in enumerate(top_5, 1):
        chunk = chunks[idx]
        retrieved_chunks.append(chunk)
        print(f"  {i}. [{chunk['chunk_id']}] (score: {score:.3f})")
        print(f"     {chunk['text'][:100]}...")

    # ----- STEP 7: Generate answer with Groq -----
    if GROQ_API_KEY:
        context = "\n\n".join([
            f"[Rule {c['chunk_id']}]: {c['text']}"
            for c in retrieved_chunks
        ])

        prompt = f"""Answer based ONLY on the context below.

Context:
{context}

Question: {query}

Answer:"""

        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.3
        )

        print("\n" + "-"*50)
        print("ANSWER:")
        print(response.choices[0].message.content)
        print("-"*50)
    else:
        print("\n(Set GROQ_API_KEY to enable answer generation)")

    print()

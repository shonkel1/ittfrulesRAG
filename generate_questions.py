import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
import os

# --------------------
# CONFIG
# --------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")  
INPUT_JSON = "ittf_chunks.json"
OUTPUT_JSON = "ittf_questions.json"
MIN_WORDS = 50
SIMILARITY_THRESHOLD = 0.75  # Chunks with similarity >= this are grouped
MAX_CHUNKS_PER_QUESTION = 5
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Local, fast model

# --------------------
# LOAD AND FILTER CHUNKS
# --------------------
print("Loading chunks...")
with open(INPUT_JSON) as f:
    all_chunks = json.load(f)

def word_count(text):
    return len(text.split())

# Filter by minimum word count
chunks = [c for c in all_chunks if word_count(c.get("text", "")) >= MIN_WORDS]
print(f"Filtered: {len(all_chunks)} → {len(chunks)} chunks (>= {MIN_WORDS} words)")

# --------------------
# GENERATE EMBEDDINGS
# --------------------
print(f"Generating embeddings with {EMBEDDING_MODEL}...")
model = SentenceTransformer(EMBEDDING_MODEL)
texts = [c["text"] for c in chunks]
embeddings = model.encode(texts, show_progress_bar=True)

# --------------------
# FIND SEMANTIC GROUPS
# --------------------
print("Finding semantically similar chunk groups...")
similarity_matrix = cosine_similarity(embeddings)

# Group chunks by similarity
used = set()
groups = []

for i in range(len(chunks)):
    if i in used:
        continue

    # Find all chunks similar to this one
    similar_indices = [i]
    for j in range(i + 1, len(chunks)):
        if j not in used and similarity_matrix[i][j] >= SIMILARITY_THRESHOLD:
            similar_indices.append(j)
            if len(similar_indices) >= MAX_CHUNKS_PER_QUESTION:
                break

    # Only create groups with 2+ chunks
    if len(similar_indices) >= 2:
        groups.append(similar_indices)
        used.update(similar_indices)

print(f"Found {len(groups)} multi-chunk groups")

# --------------------
# GENERATE QUESTIONS
# --------------------
print("Generating questions with Groq...")
client = Groq(api_key=GROQ_API_KEY)

questions = []

for group_idx, indices in enumerate(groups):
    group_chunks = [chunks[i] for i in indices]
    combined_text = "\n\n".join([
        f"[{c['chunk_id']}]: {c['text']}"
        for c in group_chunks
    ])

    chunk_ids = [c["chunk_id"] for c in group_chunks]

    prompt = f"""Based on the following ITTF rules/regulations, generate 1-2 questions that would require information from MULTIPLE of these chunks to answer fully. The questions should be natural questions someone might ask about table tennis rules.

Rules:
{combined_text}

Requirements:
- Questions should require synthesizing information from at least 2 of the provided chunks
- Questions should be specific and answerable from the text
- Avoid yes/no questions
- Format: Return a JSON array of objects with "question" and "reasoning" fields

Example output:
[{{"question": "What are the responsibilities and restrictions for athletes regarding whereabouts information?", "reasoning": "Requires combining 5.5.5.1 (establishing testing pool) and 5.5.5.8 (consequences for failure)"}}]"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)
        generated = result if isinstance(result, list) else result.get("questions", [result])

        for q in generated:
            questions.append({
                "question": q["question"],
                "reasoning": q.get("reasoning", ""),
                "answer_chunk_ids": chunk_ids,
                "answer_texts": [c["text"] for c in group_chunks],
                "group_similarity": float(np.mean([
                    similarity_matrix[indices[i]][indices[j]]
                    for i in range(len(indices))
                    for j in range(i + 1, len(indices))
                ]))
            })

        if (group_idx + 1) % 10 == 0:
            print(f"  Processed {group_idx + 1}/{len(groups)} groups...")

    except Exception as e:
        print(f"  Error on group {group_idx}: {e}")
        continue

# --------------------
# SAVE OUTPUT
# --------------------
print(f"\nSaving {len(questions)} questions to {OUTPUT_JSON}...")
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(questions, f, indent=2, ensure_ascii=False)

print(f"Done! Generated {len(questions)} multi-chunk questions")

# Print sample
if questions:
    print("\n--- Sample Question ---")
    sample = questions[0]
    print(f"Q: {sample['question']}")
    print(f"Chunks: {sample['answer_chunk_ids']}")
    print(f"Similarity: {sample['group_similarity']:.3f}")

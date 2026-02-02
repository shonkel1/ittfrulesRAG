# ITTF Rules RAG System

A Retrieval-Augmented Generation (RAG) system for querying International Table Tennis Federation (ITTF) rules and regulations.

## What it does

Ask questions about ITTF rules in natural language and get accurate answers with source citations.

```
Question: What happens if an athlete's B-sample doesn't confirm the A-sample?

Answer: According to Rule 5.7.4, if the B Sample analysis does not confirm
the A Sample finding, the Athlete shall not be subject to any further
Provisional Suspension...
```

## How it works

1. **Hybrid Search** - Combines semantic search (embeddings) + keyword search (BM25)
2. **Reranking** - Cross-encoder reranks top candidates for better accuracy
3. **Answer Generation** - LLM generates human-readable answers from retrieved chunks

```
Question → Embed → Search 1102 chunks → Combine results → Rerank → Top 5 → LLM → Answer
```

## Setup

### 1. Install dependencies

```bash
pip install numpy sentence-transformers rank-bm25 groq
```

### 2. Set your Groq API key

```bash
export GROQ_API_KEY="your-api-key-here"
```

Get a free key at [console.groq.com](https://console.groq.com)

### 3. Run

```bash
python rag.py
```

First run builds the index (~30 seconds). Subsequent runs load from cache.

## Files

| File | Description |
|------|-------------|
| `rag.py` | Main RAG system - run this to ask questions |
| `extract.py` | Extracts chunks from ITTF PDF |
| `generate_questions.py` | Generates Q&A pairs for evaluation |
| `ittf_chunks.json` | 1,102 extracted rule chunks |
| `ittf_questions.json` | 215 generated questions with answers |

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    User Question                     │
└─────────────────────┬───────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────┐
│  Embedding Model (all-mpnet-base-v2)                │
│  Converts question to 768-dim vector                │
└─────────────────────┬───────────────────────────────┘
                      ▼
┌──────────────────────────────────────────────────────┐
│  Hybrid Search                                        │
│  ├── Dense: numpy dot product with chunk embeddings  │
│  └── Sparse: BM25 keyword matching                   │
│  Combined using Reciprocal Rank Fusion (RRF)         │
└─────────────────────┬────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────┐
│  Reranker (cross-encoder/ms-marco-MiniLM-L-6-v2)    │
│  Scores (question, chunk) pairs together            │
│  Returns top 5 most relevant                        │
└─────────────────────┬───────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────┐
│  LLM (Llama 3.1 via Groq)                           │
│  Generates answer from retrieved chunks             │
└─────────────────────────────────────────────────────┘
```

## Data

- **Source**: ITTF Statutes 2025 PDF
- **Chunks**: 1,102 rule segments with hierarchical IDs (e.g., "5.7.4.1")
- **Questions**: 215 multi-chunk questions for evaluation

## License

MIT

# simple RAG x5

A small, self-contained **RAG** (Retrieval-Augmented Generation) lab for experimenting with **reranking strategies** and **LLM-as-judge** evaluation. Built in Python with [Ollama](https://ollama.com) — no cloud API keys required.

Use any plain-text document, or try the included samples `input.txt` (short) and `input10.txt` (longer, multi-section).

---

## What's new compared to *simple RAG start*?

This project is an evolution of **simple RAG start** — the same beginner-friendly RAG demo, extended with reranking and automatic evaluation. If you used the earlier version, here is what changed and what stayed the same.

### What stayed the same

| Element | Unchanged |
|---------|-----------|
| **Stack** | Python, Streamlit, Ollama, NumPy |
| **Models** | Default `llama3.2` for answers, `nomic-embed-text` for embeddings — now **selectable in the UI** |
| **Sample documents** | `input.txt` and `input10.txt` ship with the repo |
| **Chunking** | Section-aware splitting (see [Chunking](#chunking) below) — improved since *simple RAG start* |
| **Core class** | `SimpleRAG` in `rag.py` — still the main entry point |
| **Setup** | Same `venv`, `pip install`, and `ollama pull` steps |

### What changed

| Before (*simple RAG start*) | Now (*simple RAG x5*) |
|-----------------------------|------------------------|
| Project name and UI title | Renamed to **simple RAG x5** |
| Single retrieval step: embedding similarity → top-k chunks | Two-step retrieval: embedding similarity → **pool of N candidates** → **rerank** → top-k |
| One fixed search method | Three reranking strategies: **embedding (baseline)**, **keyword overlap**, **LLM relevance score** |
| One question box, one answer | Three UI tabs: **single strategy**, **compare all**, **LLM judge (A vs B)** |
| `rag.query()` returned `(answer, results)` | `rag.query()` returns a `QueryResult` with **before/after** chunk lists and the answer |
| No evaluation tooling | `compare_strategies()` and `judge_strategies()` for side-by-side and automatic comparison |
| Retrieved chunks shown once | **Before reranking** vs **after reranking** columns so you can see what changed |
| Fixed LLM for all tasks | **Answer LLM** and **Judge LLM** pickers — up to 5 local chat models (e.g. `gpt-oss`, `llama3.2`, `mistral`) |
| Max 5 context chunks | Up to **10** context chunks (`top-k`) |
| Naive fixed-size chunking | **Section-aware chunking** on headers and numbered steps (fixes mid-sentence splits) |

### Pipeline comparison

**Previous version:**

```
Document → chunks → embeddings → cosine similarity → top-k → LLM answer
```

**This version:**

```
Document → chunks → embeddings → cosine similarity → top-N candidates
                                                          ↓
                                              rerank (strategy of your choice)
                                                          ↓
                                                     top-k → LLM answer
```

The **embedding-only** strategy reproduces the old behaviour (same ordering as pure cosine similarity), but now runs inside the wider retrieve-then-rerank pipeline. That makes before/after comparisons fair: every strategy starts from the same candidate pool.

### API migration (if you have old scripts)

```python
# Before (simple RAG start)
answer, results = rag.query("Which node was selected?", top_k=3)
for r in results:
    print(r.score, r.chunk.text)

# Now (simple RAG x5) — embedding strategy is the closest equivalent
from rag import RerankStrategy

result = rag.query("Which node was selected?", top_k=3, strategy=RerankStrategy.EMBEDDING)
print(result.answer)
for r in result.reranked_results:
    print(r.score, r.chunk.text)
```

New capabilities not available in the previous version:

```python
# Compare all strategies on one question
comparison = rag.compare_strategies("What are the pre-conditions?")

# LLM judge: which strategy gives the better answer?
verdict = rag.judge_strategies(
    "Which node was selected?",
    strategy_a=RerankStrategy.EMBEDDING,
    strategy_b=RerankStrategy.KEYWORD,
)
print(verdict.winner, verdict.reasoning)
```

### Why the change?

The original project showed that RAG works, but retrieval quality can be weak when embedding similarity alone picks the wrong chunks. **simple RAG x5** keeps the same small codebase and local setup, but adds a focused lab for experimenting with reranking and measuring improvement — without databases, cloud APIs, or model training.

---

## What does this project do?

You start from a basic RAG pipeline (embed → retrieve → generate). Retrieval alone often returns chunks that are *semantically close* but not the *most useful* ones. This project lets you:

1. **Retrieve** a larger pool of candidate chunks (embedding similarity)
2. **Rerank** them with a chosen strategy
3. **Compare** before vs after, or run all strategies side by side
4. **Judge** automatically with an LLM (A vs B)

```
┌──────────────┐     ┌─────────────┐     ┌──────────────────┐
│  Document    │ ──▶ │  Embeddings │ ──▶ │ Retrieve top-N   │
│  (chunks)    │     │  (baseline) │     │ candidates       │
└──────────────┘     └─────────────┘     └────────┬─────────┘
                                                   │
                     ┌─────────────────────────────┼─────────────────────────────┐
                     │                             ▼                             │
                     │              ┌──────────────────────────────┐             │
                     │              │  Rerank (pick one strategy)  │             │
                     │              ├──────────────────────────────┤             │
                     │              │ • Embedding only (baseline)  │             │
                     │              │ • Keyword overlap            │             │
                     │              │ • LLM relevance score        │             │
                     │              └──────────────┬───────────────┘             │
                     │                             │ top-k chunks                  │
                     │                             ▼                               │
                     │              ┌──────────────────────────────┐             │
                     │              │  Answer LLM — generate answer  │             │
                     │              └──────────────────────────────┘             │
                     └───────────────────────────────────────────────────────────┘

Optional: a separate **Judge LLM** compares Answer A vs Answer B and explains the winner.
```

---

## Reranking strategies

| Strategy | How it works | When it helps |
|----------|--------------|---------------|
| **Embedding only** | Keep embedding similarity order (baseline) | Semantic questions, general meaning |
| **Keyword overlap** | Rerank by fraction of query terms found in each chunk | Questions with specific terms, section numbers, names |
| **LLM relevance score** | Ask the LLM to score each candidate 0–10 | Hard questions where heuristics disagree |

All strategies share the same initial retrieval pool (`RETRIEVE_K = 12` candidates), then keep the best `top-k` (default 3, max 10) after reranking.

---

## Embedding model: `nomic-embed-text`

**Where it comes from:** [nomic-embed-text](https://ollama.com/library/nomic-embed-text) is an open embedding model published by [Nomic AI](https://www.nomic.ai/). In this project you install it through Ollama:

```bash
ollama pull nomic-embed-text
```

**Where it runs:** Locally on your machine via the **Ollama** daemon — the same service that runs your chat models. When the app indexes a document or searches for relevant chunks, `rag.py` calls `ollama.embed(model="nomic-embed-text", ...)` for each chunk and for your question. Ollama loads the model into memory (CPU or GPU, depending on your setup), returns a vector of numbers, and the app compares vectors with cosine similarity.

**What it is used for:** Retrieval only — finding which text chunks best match a question. It does **not** write answers; chat models (e.g. `llama3.2`, `gpt-oss`) handle generation and judging.

You can change the embedding model by editing `EMBED_MODEL` in `rag.py`, as long as Ollama supports it and exposes an embed API.

---

## Model selection

Open **Model settings** at the top of the app:

| Picker | Used for |
|--------|----------|
| **Answer LLM** | Generating answers and **LLM relevance score** reranking |
| **Judge LLM** | A vs B evaluation in the **LLM judge** tab |

Both dropdowns list up to **5 chat models** detected locally via `ollama list` (embedding models such as `nomic-embed-text` are excluded). Preferred models appear first when installed: `gpt-oss`, `llama3.2`, `mistral`, `gemma2`, etc.

Pull additional models to get more choices:

```bash
ollama pull llama3.2
ollama pull gpt-oss
ollama pull mistral
ollama pull nomic-embed-text   # required for retrieval (embeddings)
```

The default answer model in code is `llama3.2`; override per query in scripts with the `llm_model` argument.

---

## Chunking

Documents are split **by structure**, not blind character count:

1. **Section headers** — lines like `6.37.2 Pre-conditions` or `## Introduction` start a new block
2. **Numbered steps** — within long sections, split at `1.`, `2.`, `3.` …
3. **Merge** — small blocks are combined up to ~1000 characters
4. **Step groups stay intact** — bullets and follow-up sentences stay with their step

This avoids mid-sentence fragments that hurt embedding retrieval.

Constants in `rag.py`:

```python
CHUNK_SIZE = 1000   # target size when merging blocks
CHUNK_OVERLAP = 0   # no character overlap (sections provide natural boundaries)
```

After changing chunk settings, **reload the Streamlit app** so documents are re-indexed.

---

## Key concepts

| Term | Description |
|------|-------------|
| **RAG** | Retrieval-Augmented Generation — find relevant text, then generate an answer from it |
| **Reranking** | Re-ordering retrieved candidates with a second scoring step |
| **LLM as judge** | Using an LLM to compare two outputs (A vs B) and pick the better one |
| **Embedding** | Numeric representation of text meaning, used for initial retrieval |
| **top-k** | How many chunks are sent to the LLM after reranking (1–10 in the UI) |
| **Ollama** | Runs AI models locally on your machine |

---

## What you need

### Python 3.10+

```bash
python3 --version
```

### Ollama

Install from [https://ollama.com](https://ollama.com), then pull the models:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
# optional — more Answer / Judge LLM choices in the UI
ollama pull gpt-oss
ollama pull mistral
```

---

## Installation

```bash
cd "/path/to/simple RAG x5"
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Running the app

```bash
streamlit run app.py
```

Open `http://localhost:8501`.

1. Choose a **source document** (`input.txt`, `input10.txt`, or add your own — see [Customization](#customization)) and configure **Model settings** (Answer LLM, Judge LLM).
2. Use one of three tabs:

### 1. Single strategy — before / after

Pick one reranking strategy. See the baseline embedding top-k **before** reranking and the final chunks **after** reranking, plus the generated answer.

### 2. Compare all strategies

Run embedding, keyword, and LLM reranking on the same question. Compare answers and retrieved chunks side by side.

### 3. LLM judge (A vs B)

Pick two strategies. Each produces an answer from its own reranked chunks. The **Judge LLM** declares the winner and explains why.

Adjust **Context chunks (top-k)** up to **10** when you need more passage coverage (more noise if set too high).

---

## Example questions

### With `input.txt` (short sample)

| Question | Why it's useful for reranking |
|----------|-------------------------------|
| *Which computing node was selected and why?* | Embedding alone may miss the right section; **LLM rerank** often finds it |
| *What are the pre-conditions?* | Section headings help keyword matching |
| *What happens in step 5 of the service flow?* | LLM rerank can prefer procedural detail over generic text |

### With `input10.txt` (longer, multi-section sample)

| Question | Why it's useful |
|----------|-----------------|
| *What are the pre-conditions for edge resource exposure?* | Many similar sections — reranking matters |
| *What is section 6.2 about?* | Disambiguates among many topics |

---

## Project structure

```
simple RAG x5/
├── input.txt          # Short sample document
├── input10.txt        # Longer multi-section sample document
├── rag.py             # RAG engine: retrieval, rerankers, LLM judge
├── app.py             # Streamlit UI
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Using the engine directly

```python
from pathlib import Path
from rag import SimpleRAG, RerankStrategy

rag = SimpleRAG(Path("input10.txt"))
rag.load_and_index()

# Single strategy with before/after
result = rag.query(
    "What are the pre-conditions?",
    top_k=3,
    strategy=RerankStrategy.KEYWORD,
)
print(result.answer)
print("Before:", [r.chunk.index for r in result.initial_results])
print("After:", [r.chunk.index for r in result.reranked_results])

# Compare all strategies
comparison = rag.compare_strategies("Which node was selected?")
for strategy, res in comparison.items():
    print(strategy.value, "->", res.answer[:80])

# LLM judge (separate judge model optional)
verdict = rag.judge_strategies(
    "Which node was selected?",
    strategy_a=RerankStrategy.EMBEDDING,
    strategy_b=RerankStrategy.LLM,
    llm_model="llama3.2",      # answers
    judge_model="gpt-oss",     # evaluator
)
print(verdict.winner, verdict.reasoning)
```

---

## Customization

Edit constants at the top of `rag.py`:

```python
LLM_MODEL = "llama3.2"       # default Answer LLM (UI overrides this)
EMBED_MODEL = "nomic-embed-text"
DEFAULT_TOP_K = 3            # Chunks sent to the LLM
MAX_TOP_K = 10               # UI slider maximum
RETRIEVE_K = 12              # Candidate pool before reranking
CHUNK_SIZE = 1000            # Section-aware merge target
CHUNK_OVERLAP = 0
```

To use your own document, add a `.txt` file and register it in `AVAILABLE_DOCUMENTS` in `app.py`:

```python
AVAILABLE_DOCUMENTS = {
    "input.txt": PROJECT_DIR / "input.txt",
    "input10.txt": PROJECT_DIR / "input10.txt",
    "my-doc.txt": PROJECT_DIR / "my-doc.txt",
}
```

To add a new reranking strategy, extend `RerankStrategy` and add a branch in `SimpleRAG.rerank()`.

---

## Learning goals

After working through this project you should be able to:

1. Start from a working baseline RAG pipeline
2. Switch between reranking strategies (keyword, embedding, LLM score)
3. Observe before/after differences in retrieved chunks and answers
4. Use an LLM judge to compare which strategy works best for a given question
5. See when simple heuristics are enough — and when an LLM helps

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Missing models | `ollama pull llama3.2` and `ollama pull nomic-embed-text` |
| No chat models in dropdown | Pull a chat model: `ollama pull gpt-oss` or `ollama pull llama3.2` |
| Ollama connection refused | Start Ollama (`ollama list` should work) |
| Before reranking looks wrong | Reload the app after code changes — chunking is cached per document |
| Reranking shows no change | Try `input10.txt`, a more specific question, or **LLM relevance score** |
| Slow compare/judge tabs | LLM reranking and judging call the model multiple times — expected |
| Wrong answer | Check retrieved chunks in the UI; increase `top-k` or tune `RETRIEVE_K` |

---

## Quick reference

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.2
ollama pull nomic-embed-text
ollama pull gpt-oss          # optional
streamlit run app.py
```

---

## License

Use and modify this project freely.

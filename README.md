# simple RAG x5

A small, self-contained **RAG** (Retrieval-Augmented Generation) lab for experimenting with **reranking strategies** and **LLM-as-judge** evaluation. Built in Python with [Ollama](https://ollama.com) — no cloud API keys required.

Use any plain-text document, or try the included samples `input.txt` (short) and `input10.txt` (longer, multi-section).

This project covers two ideas that fit together:

1. **Reranking** — start from a basic RAG pipeline, try simple ways to re-order retrieved chunks (keyword overlap, embeddings, LLM score), and compare quality **before and after**.
2. **LLM as judge** — give two answers (A vs B) to the same question and let an LLM say which is better and why — a small, reusable evaluation pattern you can apply to other AI systems.

The scope is intentionally narrow: no model training, no vector database, no cloud APIs. You should see whether retrieval improves with modest changes, and which strategy works best for your question.

---

## What's new compared to *simple RAG start*?

This project is an evolution of **simple RAG start** — the same small RAG demo, extended with reranking and automatic evaluation. If you used the earlier version, here is what changed and what stayed the same.

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

The **embedding-only** strategy keeps pure cosine-similarity ordering, but now runs inside the wider retrieve-then-rerank pipeline. That makes before/after comparisons fair: every strategy starts from the same candidate pool.

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
| **LLM** (Large Language Model) | An AI that reads and writes text. Examples: Llama, GPT. Used here to write answers and to judge them. |
| **Cosine similarity** | A score (0 to 1) measuring how similar two embeddings are. Higher = more relevant chunk. |
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

1. Choose a **source document** (`input.txt`, `input10.txt`, or add your own — see [Customization](#customization)).
2. Open **Model settings** and pick an **Answer LLM** (and optionally a different **Judge LLM**).
3. Wait for the green message that chunks are indexed.
4. Type a question and pick a tab:

### 1. Single strategy — before / after

Pick one reranking strategy. The left column shows embedding retrieval **before** reranking; the right column shows the same pool **after** reranking. Read the answer below.

This is the quickest way to see whether reranking changed which chunks were selected.

### 2. Compare all strategies

Run embedding, keyword, and LLM reranking on the **same question**. Compare answers and retrieved chunks side by side.

Use this when you want a full picture in one click.

### 3. LLM judge (A vs B)

Pick two strategies. Each produces an answer from its own reranked chunks. The **Judge LLM** declares the winner and explains why.

Use this when you want an automatic verdict instead of reading every answer yourself.

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

`rag.query()` returns a **`QueryResult`** object:

| Field | Contents |
|-------|----------|
| `result.answer` | Text generated by the Answer LLM |
| `result.reranked_results` | Chunks kept after reranking (these feed the answer) |
| `result.initial_results` | Top chunks from embedding search alone (before reranking) |
| `result.strategy` | Which reranking strategy was used |

Each item in `reranked_results` / `initial_results` has `.chunk.text`, `.score`, and `.rank`.

```python
from pathlib import Path
from rag import SimpleRAG, RerankStrategy

rag = SimpleRAG(Path("input10.txt"))
rag.load_and_index()

result = rag.query(
    "What are the pre-conditions?",
    top_k=3,
    strategy=RerankStrategy.KEYWORD,
)
print(result.answer)
for r in result.reranked_results:
    print(r.rank, r.score, r.chunk.text[:80])

# Compare all three strategies on one question
comparison = rag.compare_strategies("Which node was selected?")
for strategy, res in comparison.items():
    print(strategy.value, "->", res.answer[:80])

# LLM judge — optional separate models for answers vs evaluation
verdict = rag.judge_strategies(
    "Which node was selected?",
    strategy_a=RerankStrategy.EMBEDDING,
    strategy_b=RerankStrategy.LLM,
    llm_model="llama3.2",
    judge_model="gpt-oss",
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

## LLM as judge

The **LLM judge** tab implements a simple A/B evaluation:

1. Two reranking strategies each produce an answer to the same question.
2. A separate LLM reads both answers (without seeing the raw chunks).
3. It returns a **winner** and a short **reasoning** (factual accuracy, completeness, clarity).

No training or scoring rubric is required — just a plain prompt. You can reuse the same pattern in `rag.py` via `judge_strategies()` for scripts, tests, or other projects.

Try pairing **embedding only** vs **LLM relevance score** on a question where baseline retrieval picks the wrong section — the judge often agrees with what you see in the before/after columns.

---

## What you'll do

| Step | Action |
|------|--------|
| 1 | Run the app on `input.txt` or `input10.txt` |
| 2 | Ask a question with **embedding only** — note the retrieved chunks |
| 3 | Repeat with **keyword overlap** or **LLM relevance score** — compare before/after |
| 4 | Open **Compare all strategies** on the same question |
| 5 | Open **LLM judge** — pit two strategies against each other |

Suggested first question on `input.txt`: *Which computing node was selected and why?*  
Embedding alone often ranks the wrong chunk first; reranking (especially LLM score) usually fixes it.

---

## What you should take away

- **Clear before/after** — the UI shows exactly which chunks changed, not just a new answer.
- **Simple heuristics can work** — keyword overlap costs nothing extra and sometimes beats raw embeddings.
- **LLMs help when heuristics disagree** — LLM reranking and judging add latency but handle harder questions.
- **A reusable evaluator** — the judge component is independent of the document; swap it into other RAG or agent pipelines.

**Expected result:** a visible improvement in retrieval quality for at least some questions, with little code and no retraining.

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

## Advanced

This section describes how chunks are scored at each stage. All three reranking strategies start from the **same candidate pool** produced by `retrieve()`.

### Step 1 — Initial retrieval (shared by every strategy)

When you ask a question:

1. The question is embedded with `nomic-embed-text` (same model used at index time).
2. Cosine similarity is computed between the question vector and every chunk vector in the index.
3. The top `retrieve_k` chunks are kept (default pool size: `max(12, top_k × 2)`).

**Score:** cosine similarity in the range **0–1** (higher = closer in embedding space).

```python
score = dot(normalize(query_vec), normalize(chunk_vec))
```

The **before reranking** column in the UI shows the top `top_k` chunks from this step, still using these cosine scores.

Implementation: `SimpleRAG.retrieve()` and `_cosine_similarity()` in `rag.py`.

---

### Step 2 — Reranking (strategy-specific)

Reranking only re-orders the candidate pool. It does **not** search the full document again. The top `top_k` chunks after reranking are sent to the Answer LLM.

The **score** shown in the UI after reranking depends on which strategy you picked — the number is not always comparable across strategies.

#### Embedding only (baseline)

**What it does:** Sorts the candidate pool by the **original cosine scores** from step 1. No new scoring.

**Score shown:** cosine similarity (0–1), unchanged from retrieval.

**When order changes vs step 1:** It does not — only the slice length changes (`retrieve_k` candidates narrowed to `top_k`). Use this as the baseline to see what reranking *adds*.

---

#### Keyword overlap

**What it does:** Replaces each chunk’s cosine score with a simple word-count check: *how many words from the question also appear in this chunk?*

**Step by step:**

1. **Split into words** — Take the question and the chunk, turn them lowercase, and split on word boundaries (same idea as `Which` → `which`, `node` → `node`; punctuation is ignored).
2. **List unique question words** — Duplicates count once. *"Which node was selected?"* → `which`, `node`, `was`, `selected` (4 words).
3. **Count matches** — For each chunk, count how many of those question words appear anywhere in the chunk text.
4. **Compute score** — `matches ÷ question words`. Result is between **0** (no overlap) and **1** (every question word appears in the chunk).

**Worked example:**

| | Words |
|---|-------|
| Question | `which`, `node`, `was`, `selected` → **4 words** |
| Chunk A contains | `node`, `selected` → **2 matches** → score **2 ÷ 4 = 0.5** |
| Chunk B contains | `node` only → **1 match** → score **1 ÷ 4 = 0.25** |

Chunk A ranks above Chunk B because more of the question’s vocabulary appears in it.

**Score shown in the UI:** **0–1** (a proportion, not cosine similarity).

**Limits:** Each word is treated independently — no synonyms, no stemming (`running` ≠ `run`), no phrase matching (`"Node A"` is just `node` and `a`). Fast and predictable; useful when the question names specific terms or section labels.

Implementation: `keyword_overlap_score()` in `rag.py`.

---

#### LLM relevance score

**What it does:** Sends the question and **all candidates** (numbered `[1]`, `[2]`, …) to the Answer LLM in one prompt. The model returns a relevance score per chunk.

**Prompt asks for:** JSON array of `{"chunk_id": <1-based index>, "score": <0–10>}` for every chunk.

**Score shown:** **0–10** (model-assigned relevance).

**Parsing:** The code extracts JSON from the response. If parsing fails, it falls back to line-by-line pattern matching; if that also fails, it assigns descending placeholder scores so the pipeline still returns a result.

**Cost:** One LLM call per query (per strategy use). Slower than embedding or keyword, but can judge context and intent beyond token overlap.

Implementation: `llm_rerank()` and `_parse_llm_scores()` in `rag.py`.

---

### Comparing scores in the UI

| Strategy | Score range | Same as “before reranking”? |
|----------|-------------|------------------------------|
| Embedding only | 0–1 (cosine) | Yes |
| Keyword overlap | 0–1 (term fraction) | No — new metric |
| LLM relevance | 0–10 (model judgment) | No — new metric |

Do not compare raw numbers across strategies (e.g. `0.67` cosine vs `0.67` keyword vs `7.0` LLM). Compare **which chunks were selected** and **whether the answer improved**.

---

### LLM judge (separate from reranking)

The judge does **not** score chunks. It compares two **finished answers** (from two reranking strategies) and returns:

- `winner` — label of strategy A, strategy B, or tie
- `reasoning` — short explanation

Criteria in the prompt: factual accuracy, completeness, grounding, clarity. Uses the **Judge LLM** (can differ from the Answer LLM).

Implementation: `llm_judge()` and `judge_strategies()` in `rag.py`.

---

## License

Use and modify this project freely.

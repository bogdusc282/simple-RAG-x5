"""RAG pipeline with pluggable reranking strategies and LLM-as-judge evaluation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import ollama

LLM_MODEL = "llama3.2"
EMBED_MODEL = "nomic-embed-text"
DEFAULT_TOP_K = 3
MAX_TOP_K = 10
RETRIEVE_K = 12
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 0
SECTION_HEADER_RE = re.compile(r"^(\d+(?:\.\d+)+)\s+(.+)$")
NUMBERED_STEP_RE = re.compile(r"^\d+\.\s+")


class RerankStrategy(str, Enum):
    """Available reranking strategies."""

    EMBEDDING = "embedding"
    KEYWORD = "keyword"
    LLM = "llm"


RERANK_STRATEGY_LABELS = {
    RerankStrategy.EMBEDDING: "Embedding only (baseline)",
    RerankStrategy.KEYWORD: "Keyword overlap",
    RerankStrategy.LLM: "LLM relevance score",
}


@dataclass
class Chunk:
    text: str
    index: int


@dataclass
class SearchResult:
    chunk: Chunk
    score: float
    rank: int = 0


@dataclass
class QueryResult:
    question: str
    strategy: RerankStrategy
    initial_results: list[SearchResult]
    reranked_results: list[SearchResult]
    answer: str


@dataclass
class JudgeResult:
    question: str
    strategy_a: RerankStrategy
    strategy_b: RerankStrategy
    answer_a: str
    answer_b: str
    winner: str
    reasoning: str
    result_a: QueryResult
    result_b: QueryResult


class SimpleRAG:
    def __init__(
        self,
        document_path: str | Path,
        llm_model: str = LLM_MODEL,
        embed_model: str = EMBED_MODEL,
    ):
        self.document_path = Path(document_path)
        self.llm_model = llm_model
        self.embed_model = embed_model
        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray | None = None

    def load_and_index(self) -> int:
        text = self.document_path.read_text(encoding="utf-8")
        self.chunks = self._chunk_text(text)
        if not self.chunks:
            self.embeddings = None
            return 0

        vectors = [self._embed(chunk.text) for chunk in self.chunks]
        self.embeddings = np.array(vectors, dtype=np.float32)
        return len(self.chunks)

    def retrieve(
        self,
        question: str,
        retrieve_k: int = RETRIEVE_K,
    ) -> list[SearchResult]:
        if not self.chunks or self.embeddings is None:
            raise ValueError("No document indexed. Call load_and_index() first.")

        query_vec = np.array(self._embed(question), dtype=np.float32)
        scores = self._cosine_similarity(query_vec, self.embeddings)
        top_indices = np.argsort(scores)[::-1][:retrieve_k]

        return [
            SearchResult(chunk=self.chunks[i], score=float(scores[i]), rank=rank + 1)
            for rank, i in enumerate(top_indices)
        ]

    def rerank(
        self,
        question: str,
        candidates: list[SearchResult],
        strategy: RerankStrategy,
        top_k: int = DEFAULT_TOP_K,
        llm_model: str | None = None,
    ) -> list[SearchResult]:
        model = llm_model or self.llm_model
        if strategy == RerankStrategy.EMBEDDING:
            ranked = sorted(candidates, key=lambda r: r.score, reverse=True)
        elif strategy == RerankStrategy.KEYWORD:
            ranked = [
                SearchResult(
                    chunk=r.chunk,
                    score=keyword_overlap_score(question, r.chunk.text),
                )
                for r in candidates
            ]
            ranked.sort(key=lambda r: r.score, reverse=True)
        elif strategy == RerankStrategy.LLM:
            ranked = llm_rerank(question, candidates, model)
        else:
            raise ValueError(f"Unknown rerank strategy: {strategy}")

        return [
            SearchResult(chunk=r.chunk, score=r.score, rank=rank + 1)
            for rank, r in enumerate(ranked[:top_k])
        ]

    def query(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        strategy: RerankStrategy = RerankStrategy.EMBEDDING,
        retrieve_k: int | None = None,
        llm_model: str | None = None,
    ) -> QueryResult:
        model = llm_model or self.llm_model
        pool_size = max(retrieve_k or RETRIEVE_K, top_k * 2, top_k)
        initial = self.retrieve(question, retrieve_k=pool_size)
        reranked = self.rerank(question, initial, strategy, top_k=top_k, llm_model=model)
        answer = self._generate_answer(question, reranked, llm_model=model)
        return QueryResult(
            question=question,
            strategy=strategy,
            initial_results=initial[:top_k],
            reranked_results=reranked,
            answer=answer,
        )

    def compare_strategies(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        strategies: list[RerankStrategy] | None = None,
        llm_model: str | None = None,
    ) -> dict[RerankStrategy, QueryResult]:
        model = llm_model or self.llm_model
        strategies = strategies or list(RerankStrategy)
        pool_size = max(RETRIEVE_K, top_k * 2, top_k)
        initial = self.retrieve(question, retrieve_k=pool_size)

        results: dict[RerankStrategy, QueryResult] = {}
        for strategy in strategies:
            reranked = self.rerank(question, initial, strategy, top_k=top_k, llm_model=model)
            answer = self._generate_answer(question, reranked, llm_model=model)
            results[strategy] = QueryResult(
                question=question,
                strategy=strategy,
                initial_results=initial[:top_k],
                reranked_results=reranked,
                answer=answer,
            )
        return results

    def judge_strategies(
        self,
        question: str,
        strategy_a: RerankStrategy,
        strategy_b: RerankStrategy,
        top_k: int = DEFAULT_TOP_K,
        llm_model: str | None = None,
        judge_model: str | None = None,
    ) -> JudgeResult:
        model = llm_model or self.llm_model
        judge = judge_model or model
        result_a = self.query(question, top_k=top_k, strategy=strategy_a, llm_model=model)
        result_b = self.query(question, top_k=top_k, strategy=strategy_b, llm_model=model)
        winner, reasoning = llm_judge(
            question=question,
            answer_a=result_a.answer,
            answer_b=result_b.answer,
            label_a=RERANK_STRATEGY_LABELS[strategy_a],
            label_b=RERANK_STRATEGY_LABELS[strategy_b],
            llm_model=judge,
        )
        return JudgeResult(
            question=question,
            strategy_a=strategy_a,
            strategy_b=strategy_b,
            answer_a=result_a.answer,
            answer_b=result_b.answer,
            winner=winner,
            reasoning=reasoning,
            result_a=result_a,
            result_b=result_b,
        )

    def _generate_answer(
        self,
        question: str,
        results: list[SearchResult],
        llm_model: str | None = None,
    ) -> str:
        model = llm_model or self.llm_model
        context = "\n\n---\n\n".join(r.chunk.text for r in results)
        response = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Answer the user's question using only "
                        "the provided context. If the context does not contain enough "
                        "information, say so clearly. Be concise."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                },
            ],
        )
        return response["message"]["content"]

    def _chunk_text(self, text: str) -> list[Chunk]:
        blocks = self._split_into_blocks(text)
        pieces = self._blocks_to_pieces(blocks)
        return [
            Chunk(text=piece, index=index)
            for index, piece in enumerate(pieces)
        ]

    def _split_into_blocks(self, text: str) -> list[str]:
        """Split on section headers and numbered steps, keeping structure."""
        text = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
        blocks: list[str] = []
        current: list[str] = []
        section_header: str | None = None

        for raw_line in text.split("\n"):
            line = re.sub(r"\t+", " ", raw_line.strip())
            if not line:
                continue

            is_section = bool(SECTION_HEADER_RE.match(line))
            is_step = bool(NUMBERED_STEP_RE.match(line))

            if is_section:
                if current:
                    blocks.append("\n".join(current))
                current = [line]
                section_header = line
                continue

            if is_step and current:
                candidate = "\n".join([*current, line])
                if len(candidate) > CHUNK_SIZE and len(current) > 1:
                    blocks.append("\n".join(current))
                    current = [section_header, line] if section_header else [line]
                    continue

            current.append(line)

        if current:
            blocks.append("\n".join(current))

        return [block for block in blocks if block.strip()]

    def _blocks_to_pieces(self, blocks: list[str]) -> list[str]:
        """Merge small blocks and split oversized ones at paragraph boundaries."""
        pieces: list[str] = []
        buffer = ""

        for block in blocks:
            if not buffer:
                buffer = block
            elif len(buffer) + len(block) + 1 <= CHUNK_SIZE:
                buffer = f"{buffer}\n{block}"
            else:
                pieces.extend(self._split_oversized(buffer))
                buffer = block

        if buffer:
            pieces.extend(self._split_oversized(buffer))

        return pieces

    def _split_oversized(self, text: str) -> list[str]:
        if len(text) <= CHUNK_SIZE:
            return [text]

        lines = [line for line in text.split("\n") if line.strip()]
        if not lines:
            return []

        section_header = lines[0] if SECTION_HEADER_RE.match(lines[0]) else None
        step_starts = [
            index
            for index, line in enumerate(lines)
            if NUMBERED_STEP_RE.match(line)
        ]

        if step_starts:
            groups: list[str] = []
            for index, start in enumerate(step_starts):
                end = step_starts[index + 1] if index + 1 < len(step_starts) else len(lines)
                group_lines = lines[start:end]
                if section_header and not SECTION_HEADER_RE.match(group_lines[0]):
                    group_lines = [section_header, *group_lines]
                groups.append("\n".join(group_lines))
            return self._merge_piece_groups(groups)

        paragraphs = self._split_by_paragraphs(lines)
        if len(paragraphs) == 1 and len(paragraphs[0]) > CHUNK_SIZE:
            return self._split_by_characters(paragraphs[0])
        return self._merge_piece_groups(paragraphs)

    def _split_by_paragraphs(self, lines: list[str]) -> list[str]:
        paragraphs: list[str] = []
        current = [lines[0]]
        for line in lines[1:]:
            if SECTION_HEADER_RE.match(line):
                paragraphs.append("\n".join(current))
                current = [line]
            else:
                current.append(line)
        paragraphs.append("\n".join(current))
        return paragraphs

    def _merge_piece_groups(self, groups: list[str]) -> list[str]:
        chunks: list[str] = []
        buffer = ""

        for group in groups:
            if not buffer:
                buffer = group
            elif len(buffer) + len(group) + 1 <= CHUNK_SIZE:
                buffer = f"{buffer}\n{group}"
            else:
                chunks.append(buffer)
                buffer = group

        if buffer:
            chunks.append(buffer)

        if len(chunks) >= 2 and len(chunks[-1]) < 250:
            chunks[-2] = f"{chunks[-2]}\n{chunks[-1]}"
            chunks.pop()

        return chunks

    @staticmethod
    def _split_by_characters(text: str) -> list[str]:
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            if end < len(text):
                boundary = text.rfind(" ", start, end)
                if boundary > start:
                    end = boundary
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(chunk_text)
            if end >= len(text):
                break
            start = max(end - CHUNK_OVERLAP, start + 1)
        return chunks

    def _embed(self, text: str) -> list[float]:
        response = ollama.embed(model=self.embed_model, input=text)
        return response["embeddings"][0]

    @staticmethod
    def _cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        query_norm = query / (np.linalg.norm(query) + 1e-10)
        matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
        return matrix_norm @ query_norm


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def keyword_overlap_score(query: str, text: str) -> float:
    """Fraction of query terms that appear in the chunk (simple heuristic)."""
    query_tokens = tokenize(query)
    if not query_tokens:
        return 0.0
    text_tokens = tokenize(text)
    return len(query_tokens & text_tokens) / len(query_tokens)


def llm_rerank(
    question: str,
    candidates: list[SearchResult],
    llm_model: str,
) -> list[SearchResult]:
    """Ask the LLM to score each candidate chunk for relevance."""
    if not candidates:
        return []

    numbered = "\n\n".join(
        f"[{i}] {result.chunk.text}" for i, result in enumerate(candidates, start=1)
    )
    response = ollama.chat(
        model=llm_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You score how relevant document chunks are to a user question. "
                    "Return ONLY valid JSON: an array of objects with keys "
                    '"chunk_id" (integer, 1-based) and "score" (0-10 float). '
                    "Include every chunk exactly once."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\nChunks:\n{numbered}\n\n"
                    "Score each chunk from 0 (irrelevant) to 10 (perfect match)."
                ),
            },
        ],
    )
    content = response["message"]["content"]
    scores_by_id = _parse_llm_scores(content, len(candidates))

    scored = [
        SearchResult(chunk=r.chunk, score=scores_by_id.get(i, 0.0))
        for i, r in enumerate(candidates, start=1)
    ]
    scored.sort(key=lambda r: r.score, reverse=True)
    return scored


def _parse_llm_scores(content: str, expected_count: int) -> dict[int, float]:
    """Extract chunk scores from LLM JSON output, with a simple fallback."""
    match = re.search(r"\[.*\]", content, re.DOTALL)
    if match:
        try:
            items = json.loads(match.group())
            scores: dict[int, float] = {}
            for item in items:
                chunk_id = int(item["chunk_id"])
                scores[chunk_id] = float(item["score"])
            return scores
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass

    scores = {}
    for line in content.splitlines():
        id_match = re.search(r"chunk[_\s]*id[\"']?\s*[:=]\s*(\d+)", line, re.I)
        score_match = re.search(r"score[\"']?\s*[:=]\s*([\d.]+)", line, re.I)
        if id_match and score_match:
            scores[int(id_match.group(1))] = float(score_match.group(1))

    if not scores:
        return {i: float(expected_count - i) for i in range(1, expected_count + 1)}
    return scores


def llm_judge(
    question: str,
    answer_a: str,
    answer_b: str,
    label_a: str,
    label_b: str,
    llm_model: str,
) -> tuple[str, str]:
    """Use an LLM to pick the better answer (A vs B) and explain why."""
    response = ollama.chat(
        model=llm_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an impartial evaluator for RAG systems. "
                    "Compare two answers to the same question and decide which is better. "
                    "Judge on: factual accuracy, completeness, grounding in likely context, "
                    "and clarity. Return ONLY valid JSON with keys "
                    '"winner" ("A" or "B" or "tie"), and "reasoning" (short paragraph).'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Answer A ({label_a}):\n{answer_a}\n\n"
                    f"Answer B ({label_b}):\n{answer_b}\n\n"
                    "Which answer is better?"
                ),
            },
        ],
    )
    content = response["message"]["content"]
    return _parse_judge_response(content, label_a, label_b)


def _parse_judge_response(content: str, label_a: str, label_b: str) -> tuple[str, str]:
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            winner = str(data.get("winner", "tie")).upper()
            reasoning = str(data.get("reasoning", "")).strip()
            if winner == "A":
                return label_a, reasoning
            if winner == "B":
                return label_b, reasoning
            return "Tie", reasoning
        except (json.JSONDecodeError, TypeError):
            pass

    upper = content.upper()
    if "WINNER: A" in upper or '"WINNER": "A"' in upper:
        return label_a, content.strip()
    if "WINNER: B" in upper or '"WINNER": "B"' in upper:
        return label_b, content.strip()
    return "Tie", content.strip()

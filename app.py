"""Streamlit UI for simple RAG x5 — reranking experiments and LLM-as-judge."""

from pathlib import Path

import ollama
import streamlit as st

from rag import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBED_MODEL,
    LLM_MODEL,
    MAX_TOP_K,
    RERANK_STRATEGY_LABELS,
    RerankStrategy,
    SimpleRAG,
)

PROJECT_DIR = Path(__file__).parent
AVAILABLE_DOCUMENTS = {
    "input.txt": PROJECT_DIR / "input.txt",
    "input10.txt": PROJECT_DIR / "input10.txt",
}
MAX_LLM_CHOICES = 5
PREFERRED_CHAT_MODELS = (
    "gpt-oss",
    "llama3.2",
    "llama3.1",
    "llama3",
    "mistral",
    "gemma2",
    "qwen2.5",
    "phi3",
    "deepseek-r1",
)
EMBED_NAME_HINTS = ("embed", "nomic-embed", "bge-", "mxbai-embed")


@st.cache_resource
def get_rag(document_path: str, chunk_size: int, chunk_overlap: int) -> SimpleRAG:
    _ = (chunk_size, chunk_overlap)
    rag = SimpleRAG(document_path)
    rag.load_and_index()
    return rag


def list_ollama_models() -> list[str]:
    try:
        models = ollama.list()
        return [m["model"] for m in models.get("models", [])]
    except Exception:
        return []


def is_chat_model(model_name: str) -> bool:
    lower = model_name.lower()
    return not any(hint in lower for hint in EMBED_NAME_HINTS)


def model_sort_key(model_name: str) -> tuple[int, str]:
    base = model_name.split(":")[0].lower()
    for index, preferred in enumerate(PREFERRED_CHAT_MODELS):
        pref = preferred.lower()
        if base == pref or base.startswith(f"{pref}-") or pref in base:
            return index, model_name.lower()
    return len(PREFERRED_CHAT_MODELS), model_name.lower()


def pick_chat_models(available: list[str], limit: int = MAX_LLM_CHOICES) -> list[str]:
    chat_models = sorted(
        [name for name in available if is_chat_model(name)],
        key=model_sort_key,
    )
    return chat_models[:limit]


def model_is_available(model_name: str, available: list[str]) -> bool:
    return any(
        name == model_name or name.startswith(f"{model_name}:") for name in available
    )


def llm_selectbox(
    key: str,
    label: str,
    options: list[str],
    default: str,
) -> str:
    if default in options:
        index = options.index(default)
    else:
        index = 0
    return st.selectbox(label, options=options, index=index, key=key)


def render_chunks(results, title: str) -> None:
    st.markdown(f"**{title}**")
    if not results:
        st.caption("No chunks retrieved.")
        return
    for result in results:
        st.markdown(f"Rank {result.rank} · score {result.score:.3f}")
        st.text(result.chunk.text[:300] + ("..." if len(result.chunk.text) > 300 else ""))


def strategy_selectbox(key: str, label: str, default: RerankStrategy) -> RerankStrategy:
    options = list(RerankStrategy)
    labels = [RERANK_STRATEGY_LABELS[s] for s in options]
    default_index = options.index(default)
    choice = st.selectbox(label, options=labels, index=default_index, key=key)
    return options[labels.index(choice)]


def main() -> None:
    st.set_page_config(page_title="simple RAG x5", page_icon="📄", layout="wide")
    st.title("simple RAG x5")
    st.caption(
        "Experiment with reranking strategies to improve retrieval quality. "
        "Compare before/after, run all strategies side by side, or use an LLM judge."
    )

    available = list_ollama_models()
    chat_models = pick_chat_models(available)

    if not model_is_available(EMBED_MODEL, available):
        st.error(f"Missing embedding model. Pull it first:\n\n`ollama pull {EMBED_MODEL}`")
        st.stop()

    if not chat_models:
        st.error(
            "No chat models found locally. Pull at least one, for example:\n\n"
            "`ollama pull llama3.2`"
        )
        st.stop()

    selected_document = st.selectbox(
        "Source document",
        options=list(AVAILABLE_DOCUMENTS.keys()),
    )
    document_path = AVAILABLE_DOCUMENTS[selected_document]

    with st.expander("Model settings", expanded=True):
        col_chat, col_judge = st.columns(2)
        with col_chat:
            chat_model = llm_selectbox(
                "chat_model",
                "Answer LLM (chat & LLM reranking)",
                chat_models,
                LLM_MODEL if model_is_available(LLM_MODEL, available) else chat_models[0],
            )
        with col_judge:
            judge_model = llm_selectbox(
                "judge_model",
                "Judge LLM (A vs B evaluation)",
                chat_models,
                chat_models[0],
            )
        if len(chat_models) < MAX_LLM_CHOICES:
            st.caption(
                f"Showing **{len(chat_models)}** local chat model(s). "
                f"Pull up to {MAX_LLM_CHOICES} for more options "
                f"(e.g. `ollama pull mistral`, `ollama pull gemma2`)."
            )
        else:
            st.caption(
                f"Showing up to **{MAX_LLM_CHOICES}** local chat models "
                f"(embedding models excluded). Embeddings: `{EMBED_MODEL}`"
            )

    st.caption(f"Document: `{document_path.name}` · Answer LLM: `{chat_model}` · Judge LLM: `{judge_model}` · Embeddings: `{EMBED_MODEL}`")

    with st.spinner("Loading and indexing document..."):
        try:
            rag = get_rag(str(document_path), CHUNK_SIZE, CHUNK_OVERLAP)
        except Exception as exc:
            st.error(f"Failed to index document: {exc}")
            st.stop()

    st.success(f"Indexed **{len(rag.chunks)}** chunks from `{document_path.name}`")

    tab_single, tab_compare, tab_judge = st.tabs(
        ["Single strategy", "Compare all strategies", "LLM judge (A vs B)"]
    )

    with tab_single:
        st.subheader("Before / after reranking")
        question = st.text_input(
            "Question",
            placeholder="e.g. Which computing node was selected for AI training?",
            key="single_question",
        )
        col1, col2 = st.columns(2)
        with col1:
            strategy = strategy_selectbox("single_strategy", "Reranking strategy", RerankStrategy.EMBEDDING)
        with col2:
            top_k = st.slider(
                "Context chunks (top-k)",
                min_value=1,
                max_value=MAX_TOP_K,
                value=3,
                key="single_top_k",
            )

        if st.button("Run", type="primary", disabled=not question.strip(), key="single_run"):
            with st.spinner(f"Retrieving, reranking, and generating with `{chat_model}`..."):
                try:
                    result = rag.query(
                        question.strip(),
                        top_k=top_k,
                        strategy=strategy,
                        llm_model=chat_model,
                    )
                except Exception as exc:
                    st.error(f"Query failed: {exc}")
                    st.stop()

            st.subheader("Answer")
            st.write(result.answer)

            col_before, col_after = st.columns(2)
            with col_before:
                render_chunks(result.initial_results, "Before reranking (embedding top-k)")
            with col_after:
                label = RERANK_STRATEGY_LABELS[strategy]
                render_chunks(result.reranked_results, f"After reranking ({label})")

            if strategy != RerankStrategy.EMBEDDING:
                before_ids = [r.chunk.index for r in result.initial_results]
                after_ids = [r.chunk.index for r in result.reranked_results]
                if before_ids != after_ids:
                    st.info("Reranking changed which chunks were selected — compare the two columns above.")
                else:
                    st.warning("Reranking did not change the top chunks for this question. Try another question or strategy.")

    with tab_compare:
        st.subheader("Compare all reranking strategies")
        question = st.text_input(
            "Question",
            placeholder="e.g. What are the pre-conditions for edge resource exposure?",
            key="compare_question",
        )
        top_k = st.slider(
            "Context chunks (top-k)",
            min_value=1,
            max_value=MAX_TOP_K,
            value=3,
            key="compare_top_k",
        )

        if st.button("Compare", type="primary", disabled=not question.strip(), key="compare_run"):
            with st.spinner(f"Running all strategies with `{chat_model}` (this may take a minute)..."):
                try:
                    results = rag.compare_strategies(
                        question.strip(),
                        top_k=top_k,
                        llm_model=chat_model,
                    )
                except Exception as exc:
                    st.error(f"Comparison failed: {exc}")
                    st.stop()

            baseline = results[RerankStrategy.EMBEDDING]
            render_chunks(baseline.initial_results, "Baseline retrieval (embedding, no rerank)")

            for strategy in RerankStrategy:
                result = results[strategy]
                with st.expander(RERANK_STRATEGY_LABELS[strategy], expanded=True):
                    st.write(result.answer)
                    render_chunks(result.reranked_results, "Retrieved chunks")

    with tab_judge:
        st.subheader("LLM as judge — which strategy produces the better answer?")
        st.markdown(
            f"Two reranking strategies answer with **`{chat_model}`**. "
            f"**`{judge_model}`** picks the winner and explains why."
        )
        question = st.text_input(
            "Question",
            placeholder="e.g. What happens in step 5 of the service flow?",
            key="judge_question",
        )
        col_a, col_b, col_k = st.columns(3)
        with col_a:
            strategy_a = strategy_selectbox("judge_a", "Strategy A", RerankStrategy.EMBEDDING)
        with col_b:
            strategy_b = strategy_selectbox("judge_b", "Strategy B", RerankStrategy.KEYWORD)
        with col_k:
            top_k = st.slider(
                "Context chunks (top-k)",
                min_value=1,
                max_value=MAX_TOP_K,
                value=3,
                key="judge_top_k",
            )

        if strategy_a == strategy_b:
            st.warning("Pick two different strategies for a meaningful comparison.")

        if st.button("Judge", type="primary", disabled=not question.strip(), key="judge_run"):
            with st.spinner(
                f"Generating answers with `{chat_model}`, judging with `{judge_model}`..."
            ):
                try:
                    verdict = rag.judge_strategies(
                        question.strip(),
                        strategy_a=strategy_a,
                        strategy_b=strategy_b,
                        top_k=top_k,
                        llm_model=chat_model,
                        judge_model=judge_model,
                    )
                except Exception as exc:
                    st.error(f"Judge failed: {exc}")
                    st.stop()

            st.success(f"**Winner:** {verdict.winner}")
            st.markdown(f"**Reasoning:** {verdict.reasoning}")

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**A — {RERANK_STRATEGY_LABELS[strategy_a]}**")
                st.write(verdict.answer_a)
                render_chunks(verdict.result_a.reranked_results, "Chunks used")
            with col_b:
                st.markdown(f"**B — {RERANK_STRATEGY_LABELS[strategy_b]}**")
                st.write(verdict.answer_b)
                render_chunks(verdict.result_b.reranked_results, "Chunks used")


if __name__ == "__main__":
    main()

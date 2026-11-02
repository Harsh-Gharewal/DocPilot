"""DocPilot Enterprise RAG Platform - Streamlit Dashboard.

Provides an interactive user interface for querying documents with hybrid search,
exploring citations, comparing retrieval modalities side-by-side, managing knowledge
base ingestion, and inspecting benchmark evaluation reports.
"""

import json
import os
import sys
import time
from pathlib import Path

# Ensure DocPilot root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ensure offline huggingface operation
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import streamlit as st

from app.config import settings
from app.generation.models import AnswerResponse
from app.generation.rag import ask
from app.ingestion.indexer import index_manager
from app.ingestion.loaders import load_document

# Page Configuration
st.set_page_config(
    page_title="DocPilot | Enterprise RAG Platform",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .citation-card {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 0.75rem 1rem;
        margin-bottom: 0.75rem;
        border-radius: 0 6px 6px 0;
    }
    .badge-high {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-med {
        background-color: #FEF08A;
        color: #713F12;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-low {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_confidence_badge(label: str, score: float) -> str:
    """Generate HTML badge for confidence levels."""
    pct = round(score * 100, 1)
    if label == "HIGH":
        return f'<span class="badge-high">🛡️ HIGH CONFIDENCE ({pct}%)</span>'
    elif label == "MEDIUM":
        return f'<span class="badge-med">⚠️ MEDIUM CONFIDENCE ({pct}%)</span>'
    else:
        return f'<span class="badge-low">⛔ LOW CONFIDENCE ({pct}%)</span>'


# Sidebar Controls
with st.sidebar:
    st.image("https://raw.githubusercontent.com/google/material-design-icons/master/png/action/description/materialicons/48dp/2x/baseline_description_black_48dp.png", width=48)
    st.title("DocPilot Engine")
    st.caption("Hybrid RAG with Dense + BM25 Fusion & Reranking")

    st.markdown("---")
    st.subheader("⚙️ Pipeline Settings")

    strategy = st.selectbox(
        "Chunking Strategy",
        options=["structure_aware", "fixed_size", "semantic"],
        index=0,
        help="Chunking method used to segment documents.",
    )

    mode = st.selectbox(
        "Retrieval Modality",
        options=["hybrid_rerank", "hybrid", "dense_only", "sparse_only"],
        index=0,
        help="Search algorithm: dense vector, BM25 keyword, or RRF hybrid fusion.",
    )

    top_k = st.slider("Top K Retrieved Chunks", min_value=1, max_value=10, value=5)

    st.markdown("---")
    st.subheader("📊 System Status")
    try:
        stats = index_manager.get_strategy_stats(strategy)
        st.write(f"**Strategy Chunks:** {stats.chunk_count}")
        st.write(f"**Avg Chunk Size:** {stats.avg_char_count:.1f} chars")
    except Exception:
        st.write("Indices loading...")

    st.write(f"**LLM Provider:** `{settings.llm_provider}`")
    st.write(f"**Embeddings:** `{settings.embedding_provider}`")


# Main Tabs
tab_query, tab_compare, tab_eval, tab_docs = st.tabs(
    ["🔍 Query & Ask", "⚖️ Side-by-Side Compare", "📈 Evaluation Benchmark", "📁 Document Management"]
)


# ==============================================================================
# TAB 1: Query & Ask
# ==============================================================================
with tab_query:
    st.markdown('<div class="main-header">DocPilot Knowledge Navigator</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Ask natural language questions against internal documents with grounded citations and confidence scores.</div>',
        unsafe_allow_html=True,
    )

    # Preset Example Questions
    examples = [
        "What is the salary limit?",
        "What is the maximum reimbursement for medical claims?",
        "How is the machine vibration threshold monitored?",
        "What is the pet insurance policy?",
    ]
    selected_ex = st.selectbox("Quick prompt examples:", ["-- Select an example --"] + examples)

    default_q = "" if selected_ex == "-- Select an example --" else selected_ex
    query_text = st.text_input("Enter your question:", value=default_q, placeholder="e.g. What is the sick leave policy?")

    col_btn, col_clear = st.columns([1, 8])
    with col_btn:
        ask_submitted = st.button("Submit Query", type="primary")

    if ask_submitted and query_text.strip():
        with st.spinner("Retrieving evidence, reranking, and synthesizing answer..."):
            t0 = time.perf_counter()
            resp: AnswerResponse = ask(
                question=query_text.strip(),
                strategy=strategy,
                mode=mode,
                top_k=top_k,
            )
            elapsed_sec = time.perf_counter() - t0

        st.markdown("### Answer")
        st.write(resp.answer)

        # Confidence Badge & Metrics
        st.markdown("---")
        badge_html = render_confidence_badge(resp.confidence.label, resp.confidence.composite_score)
        st.markdown(f"**Confidence Assessment:** {badge_html}", unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Retrieval Confidence", f"{resp.confidence.retrieval_confidence:.1%}")
        m2.metric("Citation Coverage", f"{resp.confidence.citation_coverage:.1%}")
        m3.metric("Completeness", f"{resp.confidence.completeness:.1%}")
        m4.metric("Total Latency", f"{resp.latency_by_stage.get('total_ms', elapsed_sec*1000):.1f} ms")

        # Citations Section
        if resp.citations:
            st.markdown("### 📚 Source Citations")
            for c in resp.citations:
                with st.expander(f"Citation [{c.citation_number}] - {Path(c.source_doc).name} (Section: {c.section_heading or 'General'})"):
                    st.markdown(f'<div class="citation-card">"{c.quote}"</div>', unsafe_allow_html=True)
                    st.caption(f"**Source Doc:** `{c.source_doc}` | **Chunk ID:** `{c.chunk_id}` | **Similarity/RRF:** `{c.similarity_score:.4f}`")

        # Retrieval Trace & Latency Breakdown
        with st.expander("🔬 Inspection & Latency Breakdown"):
            st.json({
                "latency_by_stage": resp.latency_by_stage,
                "token_usage": resp.token_usage,
                "candidate_chunks_count": len(resp.retrieval_trace.final_chunks) if resp.retrieval_trace else 0,
            })


# ==============================================================================
# TAB 2: Side-by-Side Modality Comparison
# ==============================================================================
with tab_compare:
    st.subheader("⚖️ Side-by-Side Modality Comparison")
    st.caption("Compare how Hybrid Search (Dense + BM25 + Rerank) performs versus baseline Dense-Only vector search.")

    comp_query = st.text_input("Comparison Query:", value="What is the salary limit?", key="comp_q")
    if st.button("Run Side-by-Side Comparison", type="secondary"):
        with st.spinner("Running retrieval modalities in parallel..."):
            col_hybrid, col_dense = st.columns(2)

            with col_hybrid:
                st.markdown("#### 🌟 Mode: `hybrid_rerank`")
                resp_hybrid = ask(question=comp_query, strategy=strategy, mode="hybrid_rerank", top_k=top_k)
                st.write(resp_hybrid.answer)
                st.markdown(render_confidence_badge(resp_hybrid.confidence.label, resp_hybrid.confidence.composite_score), unsafe_allow_html=True)
                st.caption(f"Latency: {resp_hybrid.latency_by_stage.get('total_ms', 0):.1f} ms")

                st.markdown("**Retrieved Chunks:**")
                if resp_hybrid.retrieval_trace:
                    for idx, ch in enumerate(resp_hybrid.retrieval_trace.final_chunks[:3], start=1):
                        st.markdown(f"- **[{idx}]** `{Path(ch.source_file).name}` ({ch.section_heading or 'General'}) - Score: `{ch.rerank_score or ch.fusion_score:.4f}`")

            with col_dense:
                st.markdown("#### 🔍 Mode: `dense_only`")
                resp_dense = ask(question=comp_query, strategy=strategy, mode="dense_only", top_k=top_k)
                st.write(resp_dense.answer)
                st.markdown(render_confidence_badge(resp_dense.confidence.label, resp_dense.confidence.composite_score), unsafe_allow_html=True)
                st.caption(f"Latency: {resp_dense.latency_by_stage.get('total_ms', 0):.1f} ms")

                st.markdown("**Retrieved Chunks:**")
                if resp_dense.retrieval_trace:
                    for idx, ch in enumerate(resp_dense.retrieval_trace.final_chunks[:3], start=1):
                        st.markdown(f"- **[{idx}]** `{Path(ch.source_file).name}` ({ch.section_heading or 'General'}) - Score: `{ch.dense_score:.4f}`")


# ==============================================================================
# TAB 3: Evaluation Benchmark Report
# ==============================================================================
with tab_eval:
    st.subheader("📈 Golden Dataset Benchmark & Strategy Comparison")
    st.caption("50-item golden evaluation test suite comparing chunking strategies and retrieval modes.")

    chart_path = ROOT_DIR / "eval" / "results" / "strategy_comparison.png"
    if chart_path.exists():
        st.image(str(chart_path), caption="Headline RAG Metrics by Chunking Strategy", use_column_width=True)

    report_path = ROOT_DIR / "eval" / "results" / "chunking_comparison.md"
    if report_path.exists():
        report_text = report_path.read_text(encoding="utf-8")
        st.markdown(report_text)
    else:
        st.info("Benchmark report not found. Run `python -m eval.compare_strategies` to generate.")


# ==============================================================================
# TAB 4: Document Ingestion & Management
# ==============================================================================
with tab_docs:
    st.subheader("📁 Knowledge Base Documents")
    st.caption("Inspect indexed files or upload new documents into the RAG pipeline.")

    # Upload Section
    uploaded_files = st.file_uploader(
        "Upload new document (.pdf, .md, .txt, .html):",
        type=["pdf", "md", "txt", "html"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        if st.button("Process & Index Uploads", type="primary"):
            from app.ingestion.chunkers import get_chunker
            from app.ingestion.dedup import Deduplicator

            upload_dir = settings.raw_data_dir / "uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)

            total_new_chunks = 0
            for uf in uploaded_files:
                save_dest = upload_dir / uf.name
                save_dest.write_bytes(uf.getbuffer())
                doc = load_document(save_dest)
                if doc:
                    for s in ["fixed_size", "structure_aware", "semantic"]:
                        chunker = get_chunker(s)
                        chunks = chunker.chunk(doc)
                        index_manager.index_chunks(chunks, strategy=s)
                        total_new_chunks += len(chunks)

            st.success(f"Successfully processed {len(uploaded_files)} file(s) and indexed {total_new_chunks} chunks!")
            st.rerun()

    # Document Inventory
    st.markdown("---")
    st.markdown("### Indexed Document Collection")
    proc_dir = settings.processed_data_dir
    if proc_dir.exists():
        json_files = list(proc_dir.glob("*.json"))
        if json_files:
            rows = []
            for jf in json_files:
                try:
                    data = json.loads(jf.read_text(encoding="utf-8"))
                    meta = data.get("metadata", {})
                    rows.append({
                        "Document ID": data.get("doc_id", jf.stem),
                        "Filename": meta.get("filename", jf.name),
                        "Type": meta.get("file_extension", ""),
                        "Characters": len(data.get("text", "")),
                        "Sections": meta.get("section_count", 0),
                    })
                except Exception:
                    pass
            st.dataframe(rows, use_container_width=True)
        else:
            st.info("No processed documents found in storage.")

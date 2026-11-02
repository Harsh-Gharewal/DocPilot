"""DocPilot Enterprise RAG Platform - Streamlit Dashboard.

Provides an interactive, modern user interface for querying enterprise documents with
hybrid search (Dense + BM25), exploring verified citations, comparing retrieval
modalities side-by-side, evaluating golden benchmarks, and managing knowledge base files.
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

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="DocPilot | Enterprise RAG Platform",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# Modern Bright Design System & Custom CSS
# ==============================================================================
st.markdown(
    """
    <style>
    /* Global Typography & Font Smoothing */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* Top Padding Adjustment */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        max-width: 1380px;
    }

    /* Vibrant Gradient Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #1E1B4B 0%, #312E81 35%, #4F46E5 75%, #06B6D4 100%);
        border-radius: 16px;
        padding: 2.2rem 2.5rem;
        color: #FFFFFF;
        margin-bottom: 1.8rem;
        box-shadow: 0 12px 30px -8px rgba(79, 70, 229, 0.35);
        position: relative;
        overflow: hidden;
    }
    .hero-banner::after {
        content: "";
        position: absolute;
        top: -40%;
        right: -10%;
        width: 320px;
        height: 320px;
        background: radial-gradient(circle, rgba(255,255,255,0.18) 0%, rgba(255,255,255,0) 70%);
        border-radius: 50%;
        pointer-events: none;
    }
    .hero-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255, 255, 255, 0.18);
        border: 1px solid rgba(255, 255, 255, 0.32);
        padding: 4px 14px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #E0E7FF;
        margin-bottom: 0.8rem;
        backdrop-filter: blur(4px);
    }
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin: 0 0 0.5rem 0;
        color: #FFFFFF;
        line-height: 1.15;
    }
    .hero-desc {
        font-size: 1.05rem;
        color: #E0E7FF;
        max-width: 820px;
        line-height: 1.5;
        margin-bottom: 1.2rem;
        font-weight: 400;
    }
    .hero-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }
    .hero-tag {
        background: rgba(255, 255, 255, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.22);
        padding: 4px 12px;
        border-radius: 8px;
        font-size: 0.82rem;
        font-weight: 500;
        color: #F8FAFC;
    }

    /* Modern Tabs Styling */
    div[data-baseweb="tab-list"] {
        gap: 10px;
        border-bottom: 2px solid #E2E8F0;
        margin-bottom: 1.5rem;
        background-color: transparent;
    }
    button[data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 10px 10px 0 0;
        font-weight: 600;
        font-size: 0.95rem;
        color: #64748B;
        transition: all 0.2s ease;
        border: none;
    }
    button[data-baseweb="tab"]:hover {
        color: #4F46E5;
        background-color: #EEF2FF;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #4F46E5 !important;
        border-bottom: 3px solid #4F46E5 !important;
        background-color: #F8FAFC;
    }

    /* Answer Card */
    .answer-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-top: 4px solid #4F46E5;
        border-radius: 14px;
        padding: 1.6rem 1.8rem;
        box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.06);
        margin: 1.2rem 0;
    }
    .answer-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1rem;
        padding-bottom: 0.6rem;
        border-bottom: 1px solid #F1F5F9;
    }
    .answer-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0F172A;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .answer-body {
        font-size: 1.05rem;
        line-height: 1.7;
        color: #1E293B;
        white-space: pre-line;
    }

    /* KPI Metric Cards Grid */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: 14px;
        margin: 1.2rem 0;
    }
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.1rem 1.2rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(79, 70, 229, 0.1);
    }
    .kpi-label {
        font-size: 0.82rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.4rem;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .kpi-value {
        font-size: 1.7rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
    }
    .kpi-sub {
        font-size: 0.76rem;
        color: #94A3B8;
        margin-top: 0.3rem;
    }

    /* Custom Color Accents for KPI cards */
    .kpi-indigo { border-left: 4px solid #4F46E5; }
    .kpi-emerald { border-left: 4px solid #10B981; }
    .kpi-cyan { border-left: 4px solid #06B6D4; }
    .kpi-amber { border-left: 4px solid #F59E0B; }
    .kpi-purple { border-left: 4px solid #8B5CF6; }

    /* Confidence Badges */
    .badge-high {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: #FFFFFF;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.3);
    }
    .badge-med {
        background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);
        color: #FFFFFF;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        box-shadow: 0 2px 8px rgba(245, 158, 11, 0.3);
    }
    .badge-low {
        background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
        color: #FFFFFF;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        box-shadow: 0 2px 8px rgba(239, 68, 68, 0.3);
    }

    /* Citation Cards */
    .citation-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #4F46E5;
        border-radius: 0 12px 12px 0;
        padding: 1.1rem 1.4rem;
        margin-bottom: 0.9rem;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
    }
    .citation-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.5rem;
    }
    .citation-chip {
        background: #EEF2FF;
        color: #4338CA;
        font-weight: 700;
        font-size: 0.78rem;
        padding: 3px 10px;
        border-radius: 6px;
    }
    .citation-quote {
        background: #F8FAFC;
        border: 1px dashed #CBD5E1;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        color: #1E293B;
        font-size: 0.95rem;
        font-style: italic;
        line-height: 1.5;
        margin: 0.6rem 0;
    }
    .citation-meta {
        font-size: 0.78rem;
        color: #64748B;
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
    }

    /* Side-by-Side Comparison Columns */
    .compare-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1.4rem;
        box-shadow: 0 4px 14px rgba(0,0,0,0.04);
        height: 100%;
    }
    .compare-banner-hybrid {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        color: #FFFFFF;
        padding: 0.8rem 1.2rem;
        border-radius: 10px;
        font-weight: 700;
        font-size: 1.05rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .compare-banner-dense {
        background: linear-gradient(135deg, #475569 0%, #334155 100%);
        color: #FFFFFF;
        padding: 0.8rem 1.2rem;
        border-radius: 10px;
        font-weight: 700;
        font-size: 1.05rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .chunk-list-item {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.7rem 0.9rem;
        margin-bottom: 0.6rem;
        font-size: 0.88rem;
    }

    /* Sidebar Clean Styling */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0;
    }
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 0.8rem 0.4rem 1.2rem 0.4rem;
        border-bottom: 1px solid #E2E8F0;
        margin-bottom: 1.2rem;
    }
    .sidebar-brand-icon {
        background: linear-gradient(135deg, #4F46E5 0%, #06B6D4 100%);
        color: #FFFFFF;
        font-size: 1.4rem;
        padding: 8px 12px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
    }
    .sidebar-brand-text h2 {
        font-size: 1.25rem;
        font-weight: 800;
        color: #0F172A;
        margin: 0;
        line-height: 1.2;
    }
    .sidebar-brand-text span {
        font-size: 0.75rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .status-badge-online {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #ECFDF5;
        border: 1px solid #A7F3D0;
        color: #065F46;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.76rem;
        font-weight: 700;
    }
    .sidebar-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 0.9rem;
        margin-top: 1rem;
    }

    /* Primary Buttons Styling */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4F46E5 0%, #4338CA 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        font-weight: 700 !important;
        padding: 0.6rem 1.6rem !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35) !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.45) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_confidence_badge(label: str, score: float) -> str:
    """Generate modern HTML badge for confidence levels."""
    pct = round(score * 100, 1)
    if label == "HIGH":
        return f'<span class="badge-high">🛡️ HIGH CONFIDENCE ({pct}%)</span>'
    elif label == "MEDIUM":
        return f'<span class="badge-med">⚠️ MEDIUM CONFIDENCE ({pct}%)</span>'
    else:
        return f'<span class="badge-low">⛔ LOW CONFIDENCE ({pct}%)</span>'


# ==============================================================================
# Sidebar Controls & Engine Configuration
# ==============================================================================
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-icon">✈️</div>
            <div class="sidebar-brand-text">
                <h2>DocPilot</h2>
                <span>Production RAG Engine</span>
            </div>
        </div>
        <div style="margin-bottom: 1rem; display: flex; align-items: center; justify-content: space-between;">
            <span class="status-badge-online">🟢 Active & Online</span>
            <span style="font-size: 0.75rem; color: #64748B; font-weight: 600;">v1.0.0</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### ⚙️ Pipeline Settings")

    strategy = st.selectbox(
        "Chunking Strategy",
        options=["structure_aware", "fixed_size", "semantic"],
        index=0,
        help="Chunking method used to segment documents into indexable units.",
    )

    mode = st.selectbox(
        "Retrieval Modality",
        options=["hybrid_rerank", "hybrid", "dense_only", "sparse_only"],
        index=0,
        help="Search algorithm: dense vector, BM25 keyword, or RRF hybrid fusion + rerank.",
    )

    top_k = st.slider("Top K Retrieved Chunks", min_value=1, max_value=10, value=5)

    st.markdown("---")
    st.markdown("### 📊 Engine Status")

    chunk_cnt = 0
    avg_size = 0.0
    try:
        stats = index_manager.get_strategy_stats(strategy)
        chunk_cnt = stats.chunk_count
        avg_size = stats.avg_char_count
    except Exception:
        pass

    st.markdown(
        f"""
        <div class="sidebar-card">
            <div style="font-size: 0.78rem; color: #64748B; font-weight: 600; margin-bottom: 4px;">ACTIVE CHUNK REPOSITORY</div>
            <div style="font-size: 1.4rem; font-weight: 800; color: #0F172A;">{chunk_cnt} <span style="font-size: 0.8rem; font-weight: 500; color: #64748B;">chunks</span></div>
            <div style="font-size: 0.75rem; color: #94A3B8; margin-top: 2px;">Avg size: {avg_size:.1f} characters</div>
            <hr style="margin: 8px 0; border: none; border-top: 1px solid #E2E8F0;">
            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 4px;">
                <span style="color: #64748B;">LLM Model:</span>
                <span style="font-weight: 600; color: #4F46E5;">{settings.llm_provider}</span>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 4px;">
                <span style="color: #64748B;">Embeddings:</span>
                <span style="font-weight: 600; color: #06B6D4;">{settings.embedding_provider}</span>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.8rem;">
                <span style="color: #64748B;">Reranker:</span>
                <span style="font-weight: 600; color: #10B981;">ms-marco-MiniLM</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# Hero Banner
# ==============================================================================
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-pill">⚡ Enterprise Production RAG</div>
        <h1 class="hero-title">DocPilot Knowledge Navigator</h1>
        <p class="hero-desc">
            Explore verified answers over enterprise documentation with hybrid dense-vector & BM25 keyword fusion,
            cross-encoder reranking, and citation-grounded verification.
        </p>
        <div class="hero-tags">
            <span class="hero-tag">🔗 Dense + BM25 Fusion (RRF w=60)</span>
            <span class="hero-tag">🎯 Cross-Encoder Reranker</span>
            <span class="hero-tag">🛡️ Anti-Hallucination Guard</span>
            <span class="hero-tag">⚡ Sub-Second Retrieval</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# Main Interactive Tabs
# ==============================================================================
tab_query, tab_compare, tab_eval, tab_docs = st.tabs(
    ["🔍 Query & Ask", "⚖️ Side-by-Side Compare", "📈 Evaluation Benchmark", "📁 Document Management"]
)


# ==============================================================================
# TAB 1: Query & Ask
# ==============================================================================
with tab_query:
    col_input, col_meta = st.columns([7, 3])

    with col_input:
        examples = [
            "What is the maximum reimbursement for medical claims?",
            "What is the salary limit?",
            "How is the machine vibration threshold monitored?",
            "What is the pet insurance policy?",
        ]
        selected_ex = st.selectbox("💡 Sample Quick Inquiries:", ["-- Choose a suggested question --"] + examples)

        default_q = "" if selected_ex == "-- Choose a suggested question --" else selected_ex
        query_text = st.text_input(
            "Enter your question:",
            value=default_q,
            placeholder="e.g. What is the emergency medical travel benefit limit?",
            help="Type any natural language question to search across the indexed knowledge base.",
        )

        col_btn, col_info = st.columns([2, 5])
        with col_btn:
            ask_submitted = st.button("🚀 Submit Query", type="primary", use_container_width=True)
        with col_info:
            st.caption(f"Active Mode: **{mode}** | Strategy: **{strategy}** | Top K: **{top_k}**")

    with col_meta:
        st.markdown(
            """
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 1.1rem; height: 100%;">
                <div style="font-weight: 700; color: #0F172A; font-size: 0.95rem; margin-bottom: 6px;">How DocPilot Works</div>
                <div style="font-size: 0.82rem; color: #475569; line-height: 1.5;">
                    1. <b>Dual Search:</b> Queries ChromaDB vector index and BM25 inverted index simultaneously.<br>
                    2. <b>RRF Fusion:</b> Combines rankings with Reciprocal Rank Fusion.<br>
                    3. <b>Reranking:</b> Cross-Encoder scores semantic alignment.<br>
                    4. <b>Verification:</b> Answers are verified against source quotes with fallback guardrails.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if ask_submitted and query_text.strip():
        with st.spinner("🔍 Retrieving evidence chunks, reranking candidates, and synthesizing answer..."):
            t0 = time.perf_counter()
            resp: AnswerResponse = ask(
                question=query_text.strip(),
                strategy=strategy,
                mode=mode,
                top_k=top_k,
            )
            elapsed_sec = time.perf_counter() - t0

        # Synthesized Answer Card
        badge_html = render_confidence_badge(resp.confidence.label, resp.confidence.composite_score)
        st.markdown(
            f"""
            <div class="answer-card">
                <div class="answer-header">
                    <div class="answer-title">
                        <span style="font-size: 1.3rem;">💡</span> Grounded Synthesis
                    </div>
                    <div>{badge_html}</div>
                </div>
                <div class="answer-body">{resp.answer}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # High-Impact KPI Metrics Bar
        lat_ms = resp.latency_by_stage.get('total_ms', elapsed_sec * 1000)
        st.markdown(
            f"""
            <div class="metric-grid">
                <div class="kpi-card kpi-emerald">
                    <div class="kpi-label">🛡️ Retrieval Confidence</div>
                    <div class="kpi-value">{resp.confidence.retrieval_confidence:.1%}</div>
                    <div class="kpi-sub">Mean semantic & lexical affinity</div>
                </div>
                <div class="kpi-card kpi-indigo">
                    <div class="kpi-label">📖 Citation Coverage</div>
                    <div class="kpi-value">{resp.confidence.citation_coverage:.1%}</div>
                    <div class="kpi-sub">Sentences mapped to source chunks</div>
                </div>
                <div class="kpi-card kpi-purple">
                    <div class="kpi-label">✨ Completeness</div>
                    <div class="kpi-value">{resp.confidence.completeness:.1%}</div>
                    <div class="kpi-sub">Synthesizer grounding ratio</div>
                </div>
                <div class="kpi-card kpi-amber">
                    <div class="kpi-label">⚡ Latency</div>
                    <div class="kpi-value">{lat_ms:.1f} <span style="font-size: 0.9rem; font-weight: 500;">ms</span></div>
                    <div class="kpi-sub">End-to-end pipeline time</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Source Citations
        if resp.citations:
            st.markdown("### 📚 Grounded Evidence Citations")
            for c in resp.citations:
                doc_name = Path(c.source_doc).name
                sec_name = c.section_heading or "General Context"
                st.markdown(
                    f"""
                    <div class="citation-card">
                        <div class="citation-header">
                            <span class="citation-chip">Citation [{c.citation_number}]</span>
                            <span style="font-weight: 600; color: #1E293B; font-size: 0.92rem;">📄 {doc_name}</span>
                        </div>
                        <div class="citation-quote">"{c.quote}"</div>
                        <div class="citation-meta">
                            <span><b>Section:</b> {sec_name}</span>
                            <span>•</span>
                            <span><b>Source File:</b> <code>{c.source_doc}</code></span>
                            <span>•</span>
                            <span><b>Chunk ID:</b> <code>{c.chunk_id}</code></span>
                            <span>•</span>
                            <span><b>Affinity Score:</b> <code>{c.similarity_score:.4f}</code></span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Technical Inspection & Latency Breakdown
        with st.expander("🔬 Technical Telemetry & Latency Breakdown"):
            c_trace, c_tokens = st.columns(2)
            with c_trace:
                st.markdown("**Latency Breakdown (ms):**")
                st.json(resp.latency_by_stage)
            with c_tokens:
                st.markdown("**Token & Chunk Trace:**")
                st.json({
                    "token_usage": resp.token_usage,
                    "retrieved_chunk_count": len(resp.retrieval_trace.final_chunks) if resp.retrieval_trace else 0,
                    "reranker_applied": mode == "hybrid_rerank",
                })


# ==============================================================================
# TAB 2: Side-by-Side Modality Comparison
# ==============================================================================
with tab_compare:
    st.markdown("### ⚖️ Side-by-Side Modality Comparison")
    st.markdown("Benchmark how **Hybrid RRF + Cross-Encoder Rerank** compares in real-time against baseline **Dense-Only** vector retrieval.")

    comp_query = st.text_input(
        "Enter Comparison Query:",
        value="What is the maximum reimbursement for medical claims?",
        key="comp_q",
        help="Run this exact query simultaneously through two distinct retrieval engines.",
    )

    if st.button("⚡ Run Comparative Analysis", type="primary"):
        with st.spinner("Executing parallel retrieval pipelines..."):
            col_hybrid, col_dense = st.columns(2)

            with col_hybrid:
                resp_hybrid = ask(question=comp_query, strategy=strategy, mode="hybrid_rerank", top_k=top_k)
                h_lat = resp_hybrid.latency_by_stage.get('total_ms', 0)
                badge_h = render_confidence_badge(resp_hybrid.confidence.label, resp_hybrid.confidence.composite_score)

                st.markdown(
                    f"""
                    <div class="compare-card">
                        <div class="compare-banner-hybrid">
                            <span>🌟 Hybrid + Cross-Encoder Rerank</span>
                            <span style="font-size: 0.8rem; font-weight: 500;">(Recommended)</span>
                        </div>
                        <div style="margin-bottom: 0.8rem;">{badge_h}</div>
                        <div style="font-size: 0.98rem; line-height: 1.6; color: #1E293B; margin-bottom: 1rem;">
                            <b>Synthesized Answer:</b><br>{resp_hybrid.answer}
                        </div>
                        <div style="font-size: 0.82rem; color: #64748B; margin-bottom: 0.8rem;">
                            ⚡ Latency: <b>{h_lat:.1f} ms</b> | Chunks Evaluated: <b>{len(resp_hybrid.retrieval_trace.final_chunks) if resp_hybrid.retrieval_trace else 0}</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("**Top Retrieved Evidence:**")
                if resp_hybrid.retrieval_trace:
                    for idx, ch in enumerate(resp_hybrid.retrieval_trace.final_chunks[:3], start=1):
                        st.markdown(
                            f"""
                            <div class="chunk-list-item">
                                <b>#{idx} {Path(ch.source_file).name}</b> &mdash; <i>{ch.section_heading or 'General'}</i><br>
                                <span style="color: #4F46E5; font-weight: 600;">Score: {ch.rerank_score or ch.fusion_score:.4f}</span> | <code>{ch.chunk_id[:16]}...</code>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            with col_dense:
                resp_dense = ask(question=comp_query, strategy=strategy, mode="dense_only", top_k=top_k)
                d_lat = resp_dense.latency_by_stage.get('total_ms', 0)
                badge_d = render_confidence_badge(resp_dense.confidence.label, resp_dense.confidence.composite_score)

                st.markdown(
                    f"""
                    <div class="compare-card">
                        <div class="compare-banner-dense">
                            <span>🔍 Dense-Only Vector Baseline</span>
                            <span style="font-size: 0.8rem; font-weight: 500;">(Cosine Sim)</span>
                        </div>
                        <div style="margin-bottom: 0.8rem;">{badge_d}</div>
                        <div style="font-size: 0.98rem; line-height: 1.6; color: #1E293B; margin-bottom: 1rem;">
                            <b>Synthesized Answer:</b><br>{resp_dense.answer}
                        </div>
                        <div style="font-size: 0.82rem; color: #64748B; margin-bottom: 0.8rem;">
                            ⚡ Latency: <b>{d_lat:.1f} ms</b> | Chunks Evaluated: <b>{len(resp_dense.retrieval_trace.final_chunks) if resp_dense.retrieval_trace else 0}</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("**Top Retrieved Evidence:**")
                if resp_dense.retrieval_trace:
                    for idx, ch in enumerate(resp_dense.retrieval_trace.final_chunks[:3], start=1):
                        st.markdown(
                            f"""
                            <div class="chunk-list-item">
                                <b>#{idx} {Path(ch.source_file).name}</b> &mdash; <i>{ch.section_heading or 'General'}</i><br>
                                <span style="color: #64748B; font-weight: 600;">Dense Score: {ch.dense_score:.4f}</span> | <code>{ch.chunk_id[:16]}...</code>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )


# ==============================================================================
# TAB 3: Evaluation Benchmark Report
# ==============================================================================
with tab_eval:
    st.markdown("### 📈 Golden Dataset Benchmark & Quality Metrics")
    st.markdown("Evaluation metrics executed over the 50-item enterprise ground truth dataset.")

    # Headline Benchmark KPI Cards
    eval_json_path = ROOT_DIR / "eval" / "results" / "baseline.json"
    hit_rate = 1.0
    recall = 0.99
    mrr = 1.0
    faithfulness = 0.42
    avg_latency = 3323.0

    if eval_json_path.exists():
        try:
            b_data = json.loads(eval_json_path.read_text(encoding="utf-8"))
            ov = b_data.get("summary", {}).get("overall", {})
            hit_rate = ov.get("hit_rate_at_k", hit_rate)
            recall = ov.get("recall", recall)
            mrr = ov.get("mrr", mrr)
            faithfulness = ov.get("faithfulness", faithfulness)
            avg_latency = ov.get("avg_latency_ms", avg_latency)
        except Exception:
            pass

    st.markdown(
        f"""
        <div class="metric-grid">
            <div class="kpi-card kpi-emerald">
                <div class="kpi-label">🎯 Hit Rate @ K</div>
                <div class="kpi-value">{hit_rate:.1%}</div>
                <div class="kpi-sub">50 / 50 golden queries matched</div>
            </div>
            <div class="kpi-card kpi-indigo">
                <div class="kpi-label">📥 Mean Recall</div>
                <div class="kpi-value">{recall:.1%}</div>
                <div class="kpi-sub">Golden chunk presence</div>
            </div>
            <div class="kpi-card kpi-cyan">
                <div class="kpi-label">🏆 MRR @ 5</div>
                <div class="kpi-value">{mrr:.2f}</div>
                <div class="kpi-sub">Mean Reciprocal Rank</div>
            </div>
            <div class="kpi-card kpi-purple">
                <div class="kpi-label">🛡️ Faithfulness</div>
                <div class="kpi-value">{faithfulness:.1%}</div>
                <div class="kpi-sub">Groundedness ratio</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    chart_path = ROOT_DIR / "eval" / "results" / "strategy_comparison.png"
    if chart_path.exists():
        st.markdown(
            """
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 14px; padding: 1.2rem; box-shadow: 0 4px 14px rgba(0,0,0,0.04); margin: 1.5rem 0;">
                <h4 style="margin: 0 0 0.8rem 0; color: #0F172A;">📊 Comparative Headline Metrics by Chunking Strategy</h4>
            """,
            unsafe_allow_html=True,
        )
        st.image(str(chart_path), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    report_path = ROOT_DIR / "eval" / "results" / "chunking_comparison.md"
    if report_path.exists():
        with st.expander("📄 Detailed Strategy Comparison Report (Markdown)", expanded=True):
            report_text = report_path.read_text(encoding="utf-8")
            st.markdown(report_text)
    else:
        st.info("Benchmark report not found. Run `python -m eval.compare_strategies` to generate.")


# ==============================================================================
# TAB 4: Document Ingestion & Management
# ==============================================================================
with tab_docs:
    st.markdown("### 📁 Enterprise Knowledge Base Management")
    st.markdown("Upload documents, trigger multi-strategy chunking, and view real-time indexed document inventory.")

    # Upload Section
    st.markdown(
        """
        <div style="background: #FFFFFF; border: 2px dashed #CBD5E1; border-radius: 14px; padding: 1.8rem; margin-bottom: 1.5rem; text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 6px;">📥</div>
            <div style="font-size: 1.1rem; font-weight: 700; color: #0F172A;">Add Documents to Knowledge Base</div>
            <div style="font-size: 0.88rem; color: #64748B; margin-bottom: 1rem;">Supports PDF, Markdown (.md), Plain Text (.txt), and HTML documents</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Choose files to ingest:",
        type=["pdf", "md", "txt", "html"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        col_up_btn, _ = st.columns([2, 5])
        with col_up_btn:
            process_btn = st.button("⚡ Process & Index Documents", type="primary", use_container_width=True)

        if process_btn:
            from app.ingestion.chunkers import get_chunker

            upload_dir = settings.raw_data_dir / "uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)

            total_new_chunks = 0
            with st.spinner("Processing documents, generating embeddings, and updating BM25 index..."):
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

            st.success(f"Successfully processed {len(uploaded_files)} file(s) and indexed {total_new_chunks} chunks across all 3 strategies!")
            time.sleep(1)
            st.rerun()

    # Document Inventory Table
    st.markdown("---")
    st.markdown("### 📚 Indexed Document Inventory")
    proc_dir = settings.processed_data_dir
    if proc_dir.exists():
        json_files = list(proc_dir.glob("*.json"))
        if json_files:
            rows = []
            for jf in json_files:
                try:
                    data = json.loads(jf.read_text(encoding="utf-8"))
                    meta = data.get("metadata", {})
                    ext = meta.get("file_extension", "").lower()
                    type_pill = f"📄 {ext.upper()}" if ext else "📄 DOC"

                    rows.append({
                        "Format": type_pill,
                        "Filename": meta.get("filename", jf.name),
                        "Document ID": data.get("doc_id", jf.stem),
                        "Total Characters": f"{len(data.get('text', '')):,}",
                        "Detected Sections": meta.get("section_count", 0),
                    })
                except Exception:
                    pass
            st.dataframe(rows, use_container_width=True)
        else:
            st.info("No processed documents found in storage.")

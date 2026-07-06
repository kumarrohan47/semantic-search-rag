"""
app.py
------
Streamlit front-end for the Semantic Search Engine for Academic Notes (RAG System).

Run with:
    streamlit run app.py
"""

import streamlit as st

from core_rag import (
    RAGEngine,
    RAGEngineError,
    InvalidAPIKeyError,
    DocumentProcessingError,
    EmptyQueryError,
    VectorStoreNotReadyError,
)

# --------------------------------------------------------------------------- #
# Page Configuration
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Semantic Notes Search | RAG Engine",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Custom Styling
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.3rem;
            font-weight: 800;
            margin-bottom: 0rem;
            background: linear-gradient(90deg, #4F8BF9, #7C5CFF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle {
            font-size: 1.02rem;
            color: #8b8f9a;
            margin-top: 0.2rem;
            margin-bottom: 1.5rem;
        }
        .source-card {
            background-color: rgba(127, 127, 127, 0.08);
            border-left: 3px solid #4F8BF9;
            padding: 0.7rem 1rem;
            border-radius: 6px;
            margin-bottom: 0.6rem;
            font-size: 0.88rem;
        }
        .status-pill {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        .pill-green {
            background-color: rgba(46, 204, 113, 0.15);
            color: #2ecc71;
        }
        .pill-red {
            background-color: rgba(231, 76, 60, 0.15);
            color: #e74c3c;
        }
        .stChatMessage {
            border-radius: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Session State Initialization
# --------------------------------------------------------------------------- #
if "engine" not in st.session_state:
    st.session_state.engine = None
if "engine_ready" not in st.session_state:
    st.session_state.engine_ready = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0

# --------------------------------------------------------------------------- #
# Sidebar: API Key + Settings + File Upload
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    api_key_input = st.text_input(
        "Google Gemini API Key",
        type="password",
        placeholder="Paste your API key here...",
        help="Your key is used only in-memory for this session and is never stored on disk.",
    )

    if st.button("🔌 Initialize Engine", use_container_width=True):
        if not api_key_input or not api_key_input.strip():
            st.error("Please enter a valid Google Gemini API key before initializing.")
        else:
            with st.spinner("Initializing embedding + generation models..."):
                try:
                    st.session_state.engine = RAGEngine(google_api_key=api_key_input)
                    st.session_state.engine_ready = True
                    st.success("Engine initialized successfully.")
                except InvalidAPIKeyError as e:
                    st.session_state.engine_ready = False
                    st.error(f"Invalid API Key: {e}")
                except RAGEngineError as e:
                    st.session_state.engine_ready = False
                    st.error(f"Engine initialization failed: {e}")
                except Exception as e:
                    st.session_state.engine_ready = False
                    st.error(f"Unexpected error during initialization: {e}")

    # Engine status pill
    if st.session_state.engine_ready:
        st.markdown(
            '<span class="status-pill pill-green">● Engine Ready</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-pill pill-red">● Engine Not Initialized</span>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("## 📄 Upload Academic Notes")

    uploaded_files = st.file_uploader(
        "Drag and drop PDF files here",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can upload multiple PDFs at once. They will be chunked and embedded into ChromaDB.",
    )

    if st.button("🚀 Process Documents", use_container_width=True):
        if not st.session_state.engine_ready or st.session_state.engine is None:
            st.error("Please initialize the engine with a valid API key first.")
        elif not uploaded_files:
            st.warning("Please upload at least one PDF file before processing.")
        else:
            progress_bar = st.progress(0, text="Starting document processing...")
            try:
                progress_bar.progress(25, text="Extracting and chunking text...")
                files_processed, chunks_created = st.session_state.engine.ingest_pdfs(
                    uploaded_files
                )
                progress_bar.progress(75, text="Embedding chunks into ChromaDB...")
                st.session_state.processed_files.extend(
                    [f.name for f in uploaded_files]
                )
                st.session_state.total_chunks += chunks_created
                progress_bar.progress(100, text="Done.")
                st.success(
                    f"✅ Processed {files_processed} file(s) into {chunks_created} "
                    f"searchable chunks."
                )
            except DocumentProcessingError as e:
                progress_bar.empty()
                st.error(f"Document processing failed: {e}")
            except RAGEngineError as e:
                progress_bar.empty()
                st.error(f"Engine error: {e}")
            except Exception as e:
                progress_bar.empty()
                st.error(f"Unexpected error: {e}")

    if st.session_state.processed_files:
        st.divider()
        st.markdown("### 📚 Knowledge Base")
        st.caption(f"{st.session_state.total_chunks} chunks indexed in ChromaDB")
        for fname in st.session_state.processed_files:
            st.markdown(f"- 📄 {fname}")

        if st.button("🗑️ Reset Knowledge Base", use_container_width=True):
            if st.session_state.engine is not None:
                try:
                    st.session_state.engine.reset_database()
                    st.session_state.processed_files = []
                    st.session_state.total_chunks = 0
                    st.session_state.chat_history = []
                    st.success("Knowledge base cleared.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to reset database: {e}")

    st.divider()
    with st.expander("ℹ️ About this project"):
        st.markdown(
            "This app implements a **Retrieval-Augmented Generation (RAG)** "
            "pipeline: PDFs are chunked, embedded with Google's "
            "`embedding-001` model, stored in a persistent **ChromaDB** "
            "vector database, and retrieved at query time to ground "
            "**Gemini 1.5 Flash** responses in your actual notes."
        )

# --------------------------------------------------------------------------- #
# Main Panel: Header
# --------------------------------------------------------------------------- #
st.markdown(
    '<div class="main-title">📚 Semantic Search Engine for Academic Notes</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">A Retrieval-Augmented Generation (RAG) system that '
    "understands meaning, not just keywords — powered by LangChain, "
    "Google Gemini, and ChromaDB.</div>",
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Documents Indexed", len(st.session_state.processed_files))
with col2:
    st.metric("Chunks in Vector DB", st.session_state.total_chunks)
with col3:
    st.metric(
        "Engine Status",
        "Ready ✅" if st.session_state.engine_ready else "Not Ready ⚠️",
    )

st.divider()

# --------------------------------------------------------------------------- #
# Chat Interface
# --------------------------------------------------------------------------- #
st.markdown("### 💬 Ask Your Notes")

for entry in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(entry["question"])
    with st.chat_message("assistant"):
        st.markdown(entry["answer"])
        if entry.get("sources"):
            with st.expander("🔍 View retrieved sources"):
                for src in entry["sources"]:
                    st.markdown(
                        f"""<div class="source-card">
                        <b>📄 {src['source_file']}</b> — page {src['page']}<br>
                        <i>{src['snippet']}</i>
                        </div>""",
                        unsafe_allow_html=True,
                    )

user_question = st.chat_input("Ask a question about your uploaded notes...")

if user_question:
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        if not st.session_state.engine_ready or st.session_state.engine is None:
            st.error("⚠️ Please initialize the engine with a valid API key in the sidebar first.")
        else:
            with st.spinner("Retrieving relevant context and generating answer..."):
                try:
                    result = st.session_state.engine.query(user_question)
                    st.markdown(result["answer"])

                    if result.get("sources"):
                        with st.expander("🔍 View retrieved sources"):
                            for src in result["sources"]:
                                st.markdown(
                                    f"""<div class="source-card">
                                    <b>📄 {src['source_file']}</b> — page {src['page']}<br>
                                    <i>{src['snippet']}</i>
                                    </div>""",
                                    unsafe_allow_html=True,
                                )

                    st.session_state.chat_history.append(
                        {
                            "question": user_question,
                            "answer": result["answer"],
                            "sources": result.get("sources", []),
                        }
                    )
                except EmptyQueryError as e:
                    st.warning(str(e))
                except VectorStoreNotReadyError as e:
                    st.warning(f"⚠️ {e}")
                except InvalidAPIKeyError as e:
                    st.error(f"🔑 API Key issue: {e}")
                except RAGEngineError as e:
                    st.error(f"❌ Something went wrong: {e}")
                except Exception as e:
                    st.error(f"❌ Unexpected error: {e}")

# --------------------------------------------------------------------------- #
# Footer
# --------------------------------------------------------------------------- #
st.divider()
st.caption(
    "Built with LangChain · Google Gemini · ChromaDB · Streamlit — "
    "a local, persistent RAG pipeline for academic note search."
)

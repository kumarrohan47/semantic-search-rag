"""
core_rag.py
-----------
Modular backend service for a Retrieval-Augmented Generation (RAG) pipeline
built for semantic search over academic notes (PDF documents).

Responsibilities:
    1. PDF ingestion and text chunking (RecursiveCharacterTextSplitter)
    2. Vector database initialization with local persistence (ChromaDB)
    3. High-accuracy context retrieval
    4. RAG chain orchestration using modern LangChain Expression Language
       (LCEL) runnables — no dependency on the legacy Chain base class.

This module is UI-agnostic. It is imported and consumed by app.py (Streamlit),
but contains no Streamlit-specific code, which keeps it testable and reusable.
"""

import os

# Disable ChromaDB's anonymized telemetry before the chromadb package is
# imported anywhere else in the process. This avoids noisy, harmless
# "Failed to send telemetry event" warnings cluttering the console/logs
# during a live demo.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import shutil
import tempfile
import traceback
from typing import List, Optional, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from chromadb.config import Settings as ChromaSettings
from langchain.text_splitter import RecursiveCharacterTextSplitter


# --------------------------------------------------------------------------- #
# Custom Exceptions
# --------------------------------------------------------------------------- #
class RAGEngineError(Exception):
    """Base exception for all RAG engine failures."""


class InvalidAPIKeyError(RAGEngineError):
    """Raised when the Google Gemini API key is missing or invalid."""


class DocumentProcessingError(RAGEngineError):
    """Raised when a PDF fails to load, parse, or chunk correctly."""


class EmptyQueryError(RAGEngineError):
    """Raised when the user submits an empty or whitespace-only question."""


class VectorStoreNotReadyError(RAGEngineError):
    """Raised when a query is attempted before any documents have been ingested."""


# --------------------------------------------------------------------------- #
# Core Engine
# --------------------------------------------------------------------------- #
class RAGEngine:
    """
    Encapsulates the full RAG pipeline: ingestion, embedding, persistence,
    retrieval, and generation.

    Parameters
    ----------
    google_api_key : str
        Google Gemini API key used for both the embedding model
        (models/embedding-001) and the generation model (gemini-1.5-flash).
    persist_directory : str
        Local filesystem path where ChromaDB will persist its index so that
        the vector store survives application restarts.
    collection_name : str
        Name of the ChromaDB collection used to store document embeddings.
    chunk_size : int
        Maximum number of characters per text chunk.
    chunk_overlap : int
        Number of overlapping characters between consecutive chunks, used to
        preserve semantic continuity across chunk boundaries.
    """

    EMBEDDING_MODEL = "models/gemini-embedding-001"
    LLM_MODEL = "gemini-2.5-flash"

    SYSTEM_PROMPT = (
        "You are a precise, helpful academic assistant. Answer the user's "
        "question using ONLY the context provided below, which was retrieved "
        "from the student's own uploaded notes.\n\n"
        "Rules:\n"
        "1. Base your answer strictly on the provided context.\n"
        "2. If the answer cannot be found in the context, clearly say: "
        "'I could not find this in the uploaded notes.' Do not fabricate "
        "information.\n"
        "3. Be concise, well-structured, and use bullet points or short "
        "paragraphs where helpful.\n"
        "4. If relevant, mention which part of the notes the answer draws from.\n\n"
        "Context:\n{context}"
    )

    def __init__(
        self,
        google_api_key: str,
        persist_directory: str = "./chroma_db",
        collection_name: str = "academic_notes",
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
    ):
        if not google_api_key or not google_api_key.strip():
            raise InvalidAPIKeyError(
                "A Google Gemini API key is required to initialize the RAG engine."
            )

        self.google_api_key = google_api_key.strip()
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        os.makedirs(self.persist_directory, exist_ok=True)

        # Initialize embeddings model
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=self.EMBEDDING_MODEL,
                google_api_key=self.google_api_key,
            )
        except Exception as exc:
            raise InvalidAPIKeyError(
                f"Failed to initialize the embedding model. Please verify your "
                f"API key is correct and active. Details: {exc}"
            ) from exc

        # Initialize (or load) the persistent Chroma vector store
        try:
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
                client_settings=ChromaSettings(anonymized_telemetry=False),
            )
        except Exception as exc:
            raise RAGEngineError(
                f"Failed to initialize the persistent vector database at "
                f"'{self.persist_directory}'. Details: {exc}"
            ) from exc

        # Initialize the generation model
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=self.LLM_MODEL,
                google_api_key=self.google_api_key,
                temperature=0.2,
                convert_system_message_to_human=True,
            )
        except Exception as exc:
            raise InvalidAPIKeyError(
                f"Failed to initialize the Gemini chat model. Please verify "
                f"your API key. Details: {exc}"
            ) from exc

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        self._rag_chain = None  # Lazily built after ingestion or on first query

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #
    def ingest_pdfs(self, uploaded_files: List) -> Tuple[int, int]:
        """
        Ingests a list of uploaded PDF
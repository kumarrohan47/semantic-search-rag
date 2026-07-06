"""
core_rag.py
-----------
Modular backend service for a Retrieval-Augmented Generation (RAG) pipeline
built for semantic search over academic notes (PDF documents).

Responsibilities:
    1. PDF ingestion and text chunking (RecursiveCharacterTextSplitter)
    2. Vector database initialization with local persistence (ChromaDB)
    3. High-accuracy context retrieval
    4. RAG chain orchestration using modern LangChain syntax
       (create_retrieval_chain + create_stuff_documents_chain)

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
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from chromadb.config import Settings as ChromaSettings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


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

    EMBEDDING_MODEL = "models/embedding-001"
    LLM_MODEL = "gemini-1.5-flash"

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
        Ingests a list of uploaded PDF file-like objects (e.g. from Streamlit's
        st.file_uploader), splits them into overlapping chunks, embeds them,
        and persists them into the ChromaDB collection.

        Parameters
        ----------
        uploaded_files : List
            A list of file-like objects with `.name` and `.getbuffer()` /
            `.read()` methods (Streamlit's UploadedFile type satisfies this).

        Returns
        -------
        Tuple[int, int]
            (number_of_files_processed, number_of_chunks_created)

        Raises
        ------
        DocumentProcessingError
            If no valid PDF content could be extracted or chunked.
        """
        if not uploaded_files:
            raise DocumentProcessingError("No files were provided for ingestion.")

        all_chunks: List[Document] = []
        files_processed = 0
        failures = []

        for uploaded_file in uploaded_files:
            tmp_path = None
            try:
                # Persist the in-memory uploaded file to a temporary path on
                # disk because PyPDFLoader requires a filesystem path.
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf"
                ) as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    tmp_path = tmp_file.name

                loader = PyPDFLoader(tmp_path)
                raw_pages = loader.load()

                if not raw_pages:
                    failures.append(f"{uploaded_file.name}: no extractable pages found.")
                    continue

                # Attach a human-readable source filename to each page's metadata
                for page in raw_pages:
                    page.metadata["source_file"] = getattr(
                        uploaded_file, "name", "uploaded_document.pdf"
                    )

                chunks = self.text_splitter.split_documents(raw_pages)

                if not chunks:
                    failures.append(
                        f"{uploaded_file.name}: text extracted but chunking produced no segments."
                    )
                    continue

                all_chunks.extend(chunks)
                files_processed += 1

            except Exception as exc:
                failures.append(f"{getattr(uploaded_file, 'name', 'unknown file')}: {exc}")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass

        if not all_chunks:
            error_detail = "; ".join(failures) if failures else "Unknown processing error."
            raise DocumentProcessingError(
                f"No documents could be successfully processed. Details: {error_detail}"
            )

        try:
            self.vector_store.add_documents(all_chunks)
        except Exception as exc:
            raise DocumentProcessingError(
                f"Failed to embed and store document chunks in ChromaDB. "
                f"Details: {exc}"
            ) from exc

        # Invalidate the cached chain so it rebuilds with the latest retriever state
        self._rag_chain = None

        if failures:
            # Partial success: some files failed but at least one succeeded.
            # We surface this via the return value; app.py decides how to display it.
            pass

        return files_processed, len(all_chunks)

    # ------------------------------------------------------------------ #
    # Retrieval + Generation
    # ------------------------------------------------------------------ #
    def _build_chain(self, k: int = 4):
        """Builds (or rebuilds) the retrieval-augmented generation chain."""
        try:
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k},
            )

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", self.SYSTEM_PROMPT),
                    ("human", "{input}"),
                ]
            )

            document_chain = create_stuff_documents_chain(self.llm, prompt)
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            return retrieval_chain
        except Exception as exc:
            raise RAGEngineError(
                f"Failed to construct the retrieval-augmented generation chain. "
                f"Details: {exc}"
            ) from exc

    def get_document_count(self) -> int:
        """Returns the number of chunks currently stored in the vector database."""
        try:
            return self.vector_store._collection.count()
        except Exception:
            return 0

    def query(self, question: str, k: int = 4) -> dict:
        """
        Runs the full RAG pipeline for a single user question.

        Parameters
        ----------
        question : str
            The natural-language question asked by the user.
        k : int
            Number of top semantically similar chunks to retrieve as context.

        Returns
        -------
        dict
            {
                "answer": str,
                "sources": List[dict]  # [{"source_file": ..., "page": ..., "snippet": ...}, ...]
            }

        Raises
        ------
        EmptyQueryError
            If the question is empty or whitespace only.
        VectorStoreNotReadyError
            If no documents have been ingested yet.
        RAGEngineError
            If the LLM call or retrieval step fails.
        """
        if not question or not question.strip():
            raise EmptyQueryError("Please enter a non-empty question.")

        if self.get_document_count() == 0:
            raise VectorStoreNotReadyError(
                "No documents have been ingested yet. Please upload and process "
                "at least one PDF before asking a question."
            )

        try:
            if self._rag_chain is None:
                self._rag_chain = self._build_chain(k=k)

            result = self._rag_chain.invoke({"input": question.strip()})

            answer = result.get("answer", "").strip()
            if not answer:
                answer = "I could not generate a response. Please try rephrasing your question."

            source_docs = result.get("context", [])
            sources = []
            for doc in source_docs:
                sources.append(
                    {
                        "source_file": doc.metadata.get("source_file", "unknown"),
                        "page": doc.metadata.get("page", "N/A"),
                        "snippet": doc.page_content[:250].replace("\n", " ").strip() + "...",
                    }
                )

            return {"answer": answer, "sources": sources}

        except (EmptyQueryError, VectorStoreNotReadyError):
            raise
        except Exception as exc:
            traceback.print_exc()
            raise RAGEngineError(
                f"The query failed during retrieval or generation. This is often "
                f"caused by an invalid/expired API key or a network issue. "
                f"Details: {exc}"
            ) from exc

    # ------------------------------------------------------------------ #
    # Maintenance
    # ------------------------------------------------------------------ #
    def reset_database(self) -> None:
        """
        Completely wipes the persistent ChromaDB collection and the on-disk
        storage directory. Useful for demos where you want a clean slate.
        """
        try:
            self.vector_store.delete_collection()
        except Exception:
            pass

        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory, ignore_errors=True)

        os.makedirs(self.persist_directory, exist_ok=True)

        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
            client_settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._rag_chain = None

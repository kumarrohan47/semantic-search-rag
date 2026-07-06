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
       (LCEL) runnables - no dependency on the legacy Chain base class.
This module is UI-agnostic. It is imported and consumed by app.py (Streamlit),
but contains no Streamlit-specific code, which keeps it testable and reusable.
"""
import os
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
class RAGEngineError(Exception):
    pass
class InvalidAPIKeyError(RAGEngineError):
    pass
class DocumentProcessingError(RAGEngineError):
    pass
class EmptyQueryError(RAGEngineError):
    pass
class VectorStoreNotReadyError(RAGEngineError):
    pass
class RAGEngine:
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
        self._rag_chain = None
    def ingest_pdfs(self, uploaded_files: List) -> Tuple[int, int]:
        if not uploaded_files:
            raise DocumentProcessingError("No files were provided for ingestion.")
        all_chunks: List[Document] = []
        files_processed = 0
        failures = []
        for uploaded_file in uploaded_files:
            tmp_path = None
            try:
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
        self._rag_chain = None
        return files_processed, len(all_chunks)
    @staticmethod
    def _format_docs(docs) -> str:
        return "\n\n".join(doc.page_content for doc in docs)
    def _build_chain(self, k: int = 4):
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
            generation_chain = prompt | self.llm | StrOutputParser()
            rag_chain = RunnablePassthrough.assign(
                context=(lambda x: x["input"]) | retriever
            ) | RunnablePassthrough.assign(
                answer=(
                    lambda x: {
                        "input": x["input"],
                        "context": self._format_docs(x["context"]),
                    }
                )
                | generation_chain
            )
            return rag_chain
        except Exception as exc:
            raise RAGEngineError(
                f"Failed to construct the retrieval-augmented generation chain. "
                f"Details: {exc}"
            ) from exc
    def get_document_count(self) -> int:
        try:
            return self.vector_store._collection.count()
        except Exception:
            return 0
    def query(self, question: str, k: int = 4) -> dict:
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
    def reset_database(self) -> None:
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

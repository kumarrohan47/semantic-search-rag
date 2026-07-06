# 📚 Semantic Search Engine for Academic Notes (RAG System)

> A local-first Retrieval-Augmented Generation pipeline that turns static PDF notes into a conversational, context-aware knowledge base.

**Semantic Search Engine for Academic Notes** is a production-style RAG application that lets a student upload their PDF notes and ask natural-language questions against them. Instead of relying on brittle keyword matching, the system converts both the notes and the query into dense vector embeddings, retrieves the most semantically relevant passages from a persistent **ChromaDB** vector store, and passes that context to **Google's Gemini 1.5 Flash** to generate a grounded, hallucination-resistant answer. It was built to demonstrate how enterprises architect real-world Generative AI pipelines — covering document ingestion, chunking strategy, embedding generation, vector persistence, and LLM orchestration with LangChain's modern chain APIs.

---

## 🧠 Why This Project

Traditional search (`Ctrl+F`, SQL `LIKE`, Elasticsearch keyword queries) fails the moment a user phrases a question differently than the source text. This project solves that by searching on **meaning**, not string matches — the same core pattern behind enterprise tools like internal knowledge-base copilots, customer support bots, and legal/medical document assistants.

---

## 🏗️ Technical Architecture

```
                         ┌──────────────────────┐
                         │   PDF Notes Upload    │
                         │   (Streamlit UI)      │
                         └──────────┬────────────┘
                                    │
                                    ▼
                    ┌────────────────────────────────┐
                    │   PyPDFLoader (Text Extraction) │
                    └──────────────┬───────────────────┘
                                    │
                                    ▼
                ┌──────────────────────────────────────────┐
                │  RecursiveCharacterTextSplitter           │
                │  chunk_size=1000, chunk_overlap=150       │
                └──────────────────┬─────────────────────────┘
                                    │
                                    ▼
                ┌──────────────────────────────────────────┐
                │  Google Embedding Model                   │
                │  (models/embedding-001)                   │
                └──────────────────┬─────────────────────────┘
                                    │
                                    ▼
                ┌──────────────────────────────────────────┐
                │  ChromaDB (Persistent Local Vector Store) │
                │  ./chroma_db  →  survives app restarts    │
                └──────────────────┬─────────────────────────┘
                                    │
              ┌─────────────────────┴─────────────────────┐
              │            User asks a question             │
              └─────────────────────┬─────────────────────┘
                                    ▼
                ┌──────────────────────────────────────────┐
                │  Similarity Retriever (top-k = 4)         │
                └──────────────────┬─────────────────────────┘
                                    │
                                    ▼
                ┌──────────────────────────────────────────┐
                │  create_stuff_documents_chain             │
                │  + create_retrieval_chain (LangChain)     │
                │  → Gemini 1.5 Flash generates the answer  │
                └──────────────────┬─────────────────────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │  Grounded Answer +    │
                         │  Cited Source Chunks  │
                         └──────────────────────┘
```

**Pipeline in one line:**
`PDF → Chunk → Embed → Store (ChromaDB) → Retrieve (Top-K Similarity) → Augment Prompt → Generate (Gemini) → Answer`

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangChain (`create_retrieval_chain`, `create_stuff_documents_chain`) |
| LLM | Google Gemini `gemini-1.5-flash` |
| Embeddings | Google `models/embedding-001` |
| Vector Database | ChromaDB (persistent, local disk storage) |
| Frontend | Streamlit |
| PDF Parsing | PyPDF |

---

## 📁 Project Structure

```
semantic-search-rag/
│
├── app.py              # Streamlit frontend — chat UI, file uploader, sidebar
├── core_rag.py          # Backend RAG engine — ingestion, embedding, retrieval, generation
├── requirements.txt      # Pinned dependencies
├── README.md            # You are here
└── chroma_db/            # Auto-created on first run — persistent vector store
```

---

## 🚀 Setup and Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/semantic-search-rag.git
cd semantic-search-rag
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Get a Google Gemini API Key
1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Generate a free API key.
3. Keep it handy — you'll paste it directly into the app's sidebar (it is never written to disk).

### 5. Run the application
```bash
streamlit run app.py
```

### 6. Use the app
1. Paste your Gemini API key into the sidebar and click **Initialize Engine**.
2. Upload one or more PDF files (lecture notes, textbook chapters, etc.).
3. Click **Process Documents** — this chunks, embeds, and stores them in ChromaDB.
4. Ask questions in the chat box and get grounded, cited answers.

---

## 🧪 Design Decisions Worth Highlighting in an Interview

- **Persistent, not in-memory, vector storage** — `ChromaDB` is initialized with a `persist_directory="./chroma_db"`, so the knowledge base survives app restarts instead of rebuilding embeddings every session (saving API cost and latency).
- **Overlapping chunks (150 characters)** — prevents a sentence or idea that spans a chunk boundary from losing context during retrieval.
- **`create_retrieval_chain` + `create_stuff_documents_chain`** — the modern, composable LangChain pattern (replacing the deprecated `RetrievalQA`), which cleanly separates the retriever from the generation/prompt logic.
- **Strict system prompt grounding** — the LLM is explicitly instructed to answer only from retrieved context and to say so when it cannot find an answer, reducing hallucination risk.
- **Defensive error handling** — every failure mode (bad API key, corrupt PDF, empty question, empty vector store) raises a specific custom exception that the UI catches and displays gracefully, so the app never crashes mid-demo.

---

## 🎤 Mock Interview Q&A

### Q1: "Why did you use chunk overlap, and how did you pick the chunk size and overlap values?"

**A:** When you split a document into fixed-size chunks, a sentence or idea can get cut in half exactly at a chunk boundary — if that boundary happens to fall in the middle of the most relevant sentence, the retriever might miss it or return an incomplete thought. I used `RecursiveCharacterTextSplitter` with a `chunk_size` of 1000 characters and a `chunk_overlap` of 150 characters, so the last ~15% of one chunk is repeated at the start of the next. That overlap acts as a buffer that preserves continuity across boundaries. I chose 1000 characters because it's roughly 200-250 tokens — small enough to keep retrieval precise (so the top-k chunks I feed to the LLM are tightly relevant), but large enough to retain full sentences and paragraph-level context rather than fragmenting ideas into meaningless slices. In production, I'd tune this empirically using retrieval-eval metrics like context precision/recall against a labeled Q&A set.

### Q2: "How does your system prevent or reduce hallucinations from the LLM?"

**A:** I addressed this at two levels. First, at the **retrieval level** — I only ever pass the LLM the top-k (k=4) most semantically similar chunks from the student's own notes, so the model's "working context" is restricted to actual source material rather than its own pretrained knowledge. Second, at the **prompting level** — my system prompt explicitly instructs the model: "Answer using ONLY the context provided" and "If the answer cannot be found in the context, clearly say you could not find it — do not fabricate." This is a form of grounding/constrained generation. I also surface the retrieved source chunks (with filename and page number) in the UI so a user can independently verify the answer against the original text, which is a common production pattern for building trust in RAG systems. What I haven't implemented yet, but would in a v2, is a citation-verification step or a secondary "faithfulness" check using something like RAGAS to automatically flag answers that drift from the retrieved context.

### Q3: "Walk me through what a vector embedding actually is and why semantic search works better than keyword search."

**A:** An embedding is a fixed-length numerical vector — essentially a list of floating-point numbers — that represents the *meaning* of a piece of text in a high-dimensional space, produced by a neural network (here, Google's `embedding-001` model) trained so that semantically similar text ends up close together in that space, regardless of the exact words used. For example, "What causes inflation?" and "Why do prices rise across an economy?" would produce vectors that are close together by cosine similarity, even though they share almost no keywords. Keyword search (like SQL `LIKE` or basic full-text search) only matches literal substrings, so it fails on synonyms, paraphrasing, or conceptual questions. In my pipeline, I embed every chunk of the uploaded notes once at ingestion time and store those vectors in ChromaDB; at query time, I embed the user's question with the same model and ask ChromaDB for the k nearest vectors by similarity. That's the core mechanism that lets the system "understand" a question instead of just pattern-matching characters.

---

## 🔮 Future Improvements

- Add hybrid search (BM25 keyword + vector similarity) for queries with exact technical terms (formulas, proper nouns).
- Add RAGAS-based automated evaluation of answer faithfulness and context relevance.
- Support multi-user collections with namespaced ChromaDB persistence.
- Stream LLM responses token-by-token for a snappier UX.
- Add OCR fallback (e.g., Tesseract) for scanned/image-based PDFs.

---

## 📄 License

This project is open-sourced under the MIT License — feel free to fork, extend, and use it in your own portfolio.

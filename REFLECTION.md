# REFLECTION.md

## 1. What makes your chatbot feel intelligent rather than just doing keyword search? What did you specifically do to get there?
 
Several deliberate design decisions separate this from a keyword search system:
 
**Two-stage retrieval with cross-encoder reranking.**
A keyword search returns documents that contain the query terms. This system first embeds the query and retrieves the top-20 semantically similar chunks using `BAAI/bge-large-en-v1.5` — so a question like "how profitable was Infosys?" retrieves passages about operating margin even if the word "profitable" doesn't appear. Then a cross-encoder (`ms-marco-MiniLM-L-6-v2`) re-reads the query and each of the 20 passages together, scoring them by how well the passage actually answers the question. I manually verified this across a representative set of financial questions: before reranking, the top result was often a section heading or introductory sentence; after reranking, it was almost always the specific table or data passage the question referred to. The reranker consistently promoted numerically precise chunks over topically related but less useful ones.
 
**Query reformulation for follow-ups.**
Multi-turn conversation is where most RAG systems break down. A follow-up like "how does that compare to Q2?" is useless as a search query, the retriever has no memory of what "that" refers to. This system detects coreference signals in the query and rewrites it using conversation history before it hits the retriever, so "how does that compare to Q2?" becomes "How does Infosys revenue in Q3 FY26 compare to Q2 FY26?" The user experience this creates is meaningful: you can have a real back-and-forth conversation about financials without repeating context in every message, which is exactly what talking to a human analyst feels like. The reformulation only triggers when coreference signals are detected clean,
self-contained queries pass straight through without an extra LLM call.
 
**Grounded prompting with citation enforcement.**
The LLM never sees the raw documents. It only sees the top-8 retrieved chunks, each wrapped in `<source id=N>` tags. The system prompt makes citation non-negotiable: every factual claim must be tagged `[source:N]`, and if the sources don't contain the answer, the model must say so explicitly rather than guess. This eliminates hallucination of financial figures — the model can only cite numbers it was given.
 
**Format routing driven by question intent.**
The model is instructed to decide output format as part of its response — not as a separate classification step. It sees the question, the retrieved content, and the conversation history, and emits a FORMAT directive before answering. A question like "give me a comprehensive analysis of FY25" produces a narrative PDF, "compare revenue across all four quarters" produces a structured Excel workbook. This works because the model understands what kind of answer the question is actually asking for, a distinction no keyword rule could reliably make.
 
---
 
## 2. Where does it still fall short? What would a real analyst notice that your system gets wrong or misses?
 
**Chunking breaks financial tables.**
The PDF parser uses sliding-window chunking on page text, which sometimes splits a table header from its data rows across two chunks. A real analyst would notice that retrieved numbers occasionally lack context — a revenue figure without the label "USD million" or a percentage without the denominator it refers to.
 
**The annual report is only partially indexed.**
Due to embedding quota constraints during development, `infosys-ar-25.pdf` (365 pages) had to be ingested in stages. Questions requiring deep annual report content — detailed MD&A, segment breakdowns, full risk register — may retrieve less relevant chunks than questions about quarterly results, which are better represented in the vector store.
 
**No numerical reasoning.**
The system retrieves and presents numbers but cannot compute. If asked "what is the revenue CAGR from FY22 to FY26?", it will retrieve relevant figures but the LLM may calculate incorrectly or refuse. A real analyst would verify any derived calculation independently.
 
**No cross-document numerical reconciliation.**
The investor sheet, quarterly press releases, and annual report sometimes report slightly different figures for the same metric (restatements, rounding, IFRS vs. local GAAP). The system retrieves whichever chunk scores highest without flagging the discrepancy. A real analyst would immediately reconcile these differences.
 
---
 
## 3. Which AI tools did you use to build this, and what did you have to fix or override yourself?
 
**Tools used:**
 
- **Claude (Anthropic)**  used as a coding assistant during development, primarily for debugging errors  and generating boilerplate for PDF/Excel formatting. The retrieval pipeline, reranking strategy, citation enforcement and format-routing logic were designed manually and then implemented with Claude’s assistance during development.
- **Gemini API (Google)** — used as specified in the assignment brief. The project was built targeting `gemini-embedding-001` for embeddings and `gemini-2.0-flash` for generation. Both were attempted. However, the free tier quota limits were extremely restrictive — 1,000 embedding requests per day was insufficient to embed 1,247 chunks in a single run, and the generation quota was near-zero during development. These were genuine quota exhaustion issues on the free tier, not an intentional deviation from the brief.
- **Groq (LLaMA-3.3-70B / LLaMA-3.1-8B)** — used as a fallback for LLM generation after Gemini's free tier generation quota was repeatedly exhausted. The retrieval pipeline, reranking, and all document processing are LLM-provider-agnostic and would work identically with Gemini generation if quota permits.
- **BAAI/bge-large-en-v1.5** (via `sentence-transformers`) — used for document and query embedding after hitting the Gemini embedding quota mid-ingestion. Runs entirely locally with no quota constraints. Quality is competitive with API embedding models for financial retrieval.
- **cross-encoder/ms-marco-MiniLM-L-6-v2** — used for reranking retrieved chunks, runs entirely locally.
**What I had to fix or override:**
 
- **Google SDK migration** — the project originally used `google-generativeai` (deprecated). Migrated to `google-genai`, which required rewriting all client instantiation, model call syntax, and embedding API calls.
- **Embedding quota and model switch** — started with `gemini-embedding-001`, hit the 1,000/day quota at batch 99 of 125. Added a checkpoint/resume system to survive quota resets, then switched to local `BAAI/bge-large-en-v1.5` to eliminate the quota dependency entirely.
- **Format routing enforcement** — the LLM frequently chose Markdown for multi-quarter comparison queries that were better suited to Excel output. Added lightweight post-processing heuristics for unambiguous cases such as trend comparisons and quarter-wise tabular analysis.
- **FPDF rendering failures** — the PDF generator crashed repeatedly on Unicode characters, non-latin1 LLM output, and `multi_cell(0, ...)` calls where zero width remained. Fixed by sanitizing all text through a `_safe()` function and enforcing `pdf.set_x(20)` before every `multi_cell` call.
- **ChromaDB duplicate ID errors** — deterministic chunk IDs based on content hash caused collisions when two chunks had identical text (repeated PDF headers across pages). Fixed by incorporating chunk position index into the hash.
- **Groq token rate limits** — Groq's free tier has a 12,000 TPM limit. Initial chunk sizes (3,000 chars × 8 chunks) exceeded this per query. Fixed by truncating chunks to 600 chars in `format_context_block` and using the smaller `llama-3.1-8b-instant` for query reformulation to preserve budget for the main generation call.
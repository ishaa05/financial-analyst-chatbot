# Infosys Financial Analyst Chatbot

> A RAG-powered financial analyst that reads documents, remembers your conversation, cites every claim and delivers answers as Markdown, PDF reports or Excel workbooks automatically.

---

## What It Does

Ask it anything about given financials. It doesn't keyword-match. It retrieves semantically relevant passages, reranks them with a cross-encoder, feeds only the grounded context to the LLM, and enforces citations on every factual claim.

```
You: What was Infosys revenue and operating margin for Q4 FY26?
Bot: Revenue was $5,040 million [source:1], operating margin was 20.9% [source:1].

You: How does that compare to Q3?
Bot: Q3 revenue was $5,099 million [source:2] — Q4 was slightly lower by ~1.2%...
     (follow-up resolved automatically using conversation history)

You: Give me a full analysis of FY25 performance.
Bot: [generates branded PDF report + Download button]

You: Export quarterly revenue and headcount for all of FY26.
Bot: [generates formatted Excel workbook + Download button]
```

---

## Documents Covered

| File | Contents |
|------|----------|
| `infosys-ar-25.pdf` | Infosys Integrated Annual Report FY2024-25 (365 pages) |
| `ifrs-usd-press-release_q1.pdf` | Q1 FY26 Earnings Press Release (Jul 2025) |
| `ifrs-usd-press-release_q2.pdf` | Q2 FY26 Earnings Press Release (Oct 2025) |
| `ifrs-usd-press-release_q3.pdf` | Q3 FY26 Earnings Press Release (Jan 2026) |
| `ifrs-usd-press-release_q4.pdf` | Q4 FY26 Earnings Press Release (Apr 2026) |
| `investor-sheet.xls` | Multi-year P&L, balance sheet, employee data |
| `500209.csv` | Infosys BSE stock price history FY26 |

---

## Architecture

```
User query
    │
    ▼
Query Reformulation  ←  conversation history
(Groq LLaMA-3.3-70B — expands coreferences like "how does that compare?")
    │
    ▼
Dense Retrieval
(BAAI/bge-large-en-v1.5 → ChromaDB cosine ANN, top-20 candidates)
    │
    ▼
Cross-Encoder Reranking
(ms-marco-MiniLM-L-6-v2 — sees query + passage together, selects top-8)
    │
    ▼
Grounded Prompt
(retrieved chunks in <source id=N> tags + conversation history)
    │
    ▼
Groq LLaMA-3.3-70B
    │
    ├── FORMAT: markdown  →  rendered in chat with citation pills
    ├── FORMAT: pdf       →  fpdf2 branded Infosys report + Download button
    └── FORMAT: excel     →  openpyxl workbook, styled + auto-sized + Download button
```

**Why two retrieval stages?**
The bi-encoder (bge-large) retrieves the top-20 semantically similar chunks fast. The cross-encoder then re-scores all 20 by reading the query and each passage together — much higher precision for financial questions where exact numbers matter.

**Why query reformulation?**
"What about Q2?" is a useless search query without context. A preprocessing step expands it to "How does Infosys revenue in Q3 FY26 compare to Q2 FY26?" before hitting the vector store.

**Why let the LLM decide the output format?**
Pattern-matching "export" or "report" is brittle. The model sees the full question, retrieved content, and history — it has everything needed to decide whether a quick Markdown answer, a narrative PDF, or a data Excel is right. A post-processing layer also enforces format based on query signals as a safety net.

---

## Setup

### 1. Prerequisites

- Python 3.10+
- A free [Groq API key](https://console.groq.com) 

### 2. Clone and install

```bash
git clone https://github.com/ishaa05/financial-analyst-chatbot.git
cd financial-analyst-chatbot
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```
GROQ_API_KEY=your_groq_key_here
```

### 4. Add source documents

Place all 7 files in the `data/` directory:

```
data/
  infosys-ar-25.pdf
  ifrs-usd-press-release_q1.pdf
  ifrs-usd-press-release_q2.pdf
  ifrs-usd-press-release_q3.pdf
  ifrs-usd-press-release_q4.pdf
  investor-sheet.xls
  500209.csv
```

### 5. Run ingestion (one-time)

Parses all documents, generates embeddings locally, and builds the ChromaDB vector store. Re-runs are idempotent — existing chunks are skipped.

```bash
python ingest.py
```

Expected output:
```
Parsing PDF: ifrs-usd-press-release_q3.pdf
Parsing PDF: ifrs-usd-press-release_q4.pdf
Parsing PDF: infosys-ar-25.pdf
Parsing Excel: investor-sheet.xls
Parsing CSV: 500209.csv

Total chunks after dedup: 1247
Embedding…
  Embedding batch 1 (32 chunks)…
  Embedding batch 2 (32 chunks)…
  ...
Writing to ChromaDB…
  ✓ Upserted 1247 chunks. Collection now has 1247 total.

✅ Ingestion complete.
```

> Note: Embedding runs locally via `sentence-transformers` — no API calls, no quota limits.

### 6. Run the chatbot

**Option A — Streamlit web UI (recommended):**

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501)

**Option B — Terminal CLI:**

```bash
python cli.py
```

CLI commands: `/sources` · `/save` · `/clear` · `/quit`

---

## Project Structure

```
financial-analyst-chatbot/
├── data/                        # Source documents (add your 7 files here)
├── embeddings/
│   └── chroma_db/               # Persistent ChromaDB vector store (auto-created)
├── outputs/                     # Generated PDFs and Excel files (auto-created)
├── sample_conversations/
│   ├── pdf_01_annual_highlights/
│   ├── pdf_02_fy25_analysis/
│   ├── pdf_03_risks_strategy/
│   ├── excel_01_quarterly_revenue/
│   ├── excel_02_headcount/
│   └── excel_03_deal_wins/
├── ingest.py                    # Document parsing + local embedding pipeline
├── retriever.py                 # Two-stage retrieval: dense ANN + cross-encoder rerank
├── engine.py                    # LLM layer: query reformulation, grounded prompt, format routing
├── formatters.py                # PDF (fpdf2) and Excel (openpyxl) output generators
├── app.py                       # Streamlit web UI
├── cli.py                       # Terminal interface with rich formatting
├── requirements.txt
├── .env.example
├── REFLECTION.md
└── README.md
```

---

## Sample Questions

**Quick facts (Markdown):**
```
What was Infosys revenue and operating margin for Q4 FY26?
What is Infosys's attrition rate?
What is the 52-week high and low for Infosys stock in FY26?
```

**Data exports (Excel):**
```
Compare Infosys revenue across all four quarters of FY26.
Show me headcount changes quarter by quarter in FY26.
How did deal wins trend across Q1 to Q4 FY26?
```

**Reports (PDF):**
```
Summarise the key highlights from the Infosys annual report.
Give me a comprehensive analysis of Infosys financial performance in FY25.
What are the key risks and strategic priorities mentioned in the annual report?
```

**Follow-ups (conversation memory):**
```
You:  What was revenue in Q3 FY26?
Bot:  $5,099 million...
You:  How does that compare to Q2?   ← automatically resolved
You:  What about operating margin for those quarters?
```

**Edge cases (honest uncertainty):**
```
What will Infosys revenue be in FY27?      → "I could not find this..."
Who are Infosys's top clients by name?     → "I could not find this..."
What is the Infosys share price today?     → "I could not find this..."
```

---

## Requirements

```
# Core RAG pipeline
chromadb
sentence-transformers
pdfplumber
openpyxl
xlrd
fpdf2
pandas
tabulate
python-dotenv

# LLM
groq

# UI
streamlit
rich
```

Install all:
```bash
pip install -r requirements.txt
```

---

## Key Design Decisions

**Local embeddings, not API embeddings.**
`BAAI/bge-large-en-v1.5` runs entirely on CPU — no quota, no rate limits, no cost. Quality is competitive with API embedding models for financial retrieval tasks.

**Cross-encoder reranking.**
The bi-encoder retrieves fast but imprecisely. The cross-encoder (ms-marco-MiniLM-L-6-v2) re-reads query + passage together and gives dramatically better precision — critical when a question asks for a specific number across hundreds of chunks.

**Citation enforcement.**
Every factual claim must include `[source:N]` in the LLM output. The system prompt treats this as a hard rule, not a suggestion. Citations are then parsed and rendered as source pills in the UI and as an appendix in PDF reports.

**Honest uncertainty.**
The system prompt explicitly instructs the model to say "I could not find this in the provided documents" rather than hallucinate. The grounded prompt ensures the model only sees retrieved chunks — never raw documents — so there's no room to invent figures.

---

"""
app.py  –  Streamlit chat UI for the Infosys Financial Analyst chatbot.

Run:
    streamlit run app.py

Features:
  - Chat interface with streaming-style display
  - Automatic format routing: markdown / PDF / Excel
  - Download buttons for generated files
  - Source citations as expandable footnotes
  - Conversation history visible in sidebar
  - Clear conversation button
"""

from __future__ import annotations

import os
from pathlib import Path

import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv

from engine import chat, Turn, OutputFormat, EngineResponse
from formatters import generate_pdf, generate_excel

# ─── Setup ─────────────────────────────────────────────────────────────────────

load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

st.set_page_config(
    page_title="Infosys Financial Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS — clean professional look
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .stChatMessage { border-radius: 12px; }
    .source-pill {
        display: inline-block;
        background: #EBF3FF;
        color: #0066CC;
        border-radius: 4px;
        padding: 1px 6px;
        font-size: 11px;
        margin: 1px;
        font-family: monospace;
    }
    .format-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .fmt-pdf   { background: #FFF0EA; color: #C04A00; }
    .fmt-excel { background: #EAFAF0; color: #1A7C44; }
    .fmt-md    { background: #EEF2FF; color: #3B4CBB; }
</style>
""", unsafe_allow_html=True)


# ─── Session state ──────────────────────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state.history: list[Turn] = []

if "display_messages" not in st.session_state:
    # Each entry: {"role", "content", "format", "citations", "file_path"}
    st.session_state.display_messages: list[dict] = []


# ─── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/9/95/Infosys_logo.svg",
             width=140)
    st.markdown("### Infosys Financial Analyst")
    st.markdown(
        "Ask me anything about Infosys financials across FY25–FY26. "
        "I can answer questions from:\n"
        "- Annual Report FY2024-25\n"
        "- Q1–Q4 FY26 Earnings Releases\n"
        "- Multi-year Investor Sheet\n"
        "- FY26 BSE Stock Price Data"
    )
    st.divider()

    st.markdown("**Output types**")
    st.markdown(
        "🔵 **Markdown** — quick facts, tables, comparisons\n\n"
        "🟠 **PDF** — narrative reports and summaries\n\n"
        "🟢 **Excel** — data exports and time-series"
    )
    st.divider()

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.session_state.display_messages = []
        st.rerun()

    st.caption("The chatbot decides output format automatically based on the question.")


# ─── Helpers ────────────────────────────────────────────────────────────────────

def _format_badge(fmt: OutputFormat) -> str:
    label = {"markdown": "Markdown", "pdf": "PDF Report", "excel": "Excel"}[fmt.value]
    css_class = {"markdown": "fmt-md", "pdf": "fmt-pdf", "excel": "fmt-excel"}[fmt.value]
    return f'<span class="format-badge {css_class}">{label}</span>'


def _render_citations(citations: list[dict]) -> str:
    if not citations:
        return ""
    parts = []
    for c in citations:
        page_str = f", p.{c['page']}" if c.get("page") else ""
        parts.append(
            f'<span class="source-pill">[{c["id"]}] {c["label"][:30]}{page_str}</span>'
        )
    return "**Sources:** " + " ".join(parts)


def _clean_answer(answer: str) -> str:
    """Remove [source:N] tags from displayed answer — they're shown as pills instead."""
    import re
    return re.sub(r"\[source:\d+\]", "", answer).strip()


# ─── Chat display ───────────────────────────────────────────────────────────────

st.title("📊 Infosys Financial Analyst")
st.caption("Powered by Gemini 1.5 Flash · Grounded in official Infosys documents")

# Replay existing messages
for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.markdown(_format_badge(msg["format"]), unsafe_allow_html=True)
            st.markdown(_clean_answer(msg["content"]))

            if msg.get("citations"):
                st.markdown(_render_citations(msg["citations"]), unsafe_allow_html=True)

            if msg.get("file_path"):
                fp = Path(msg["file_path"])
                if fp.exists():
                    with open(fp, "rb") as f:
                        btn_label = "📥 Download PDF" if fp.suffix == ".pdf" else "📥 Download Excel"
                        mime = "application/pdf" if fp.suffix == ".pdf" else \
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        st.download_button(btn_label, f.read(), file_name=fp.name, mime=mime)

            if msg.get("citations"):
                with st.expander("📎 View source passages"):
                    for chunk in msg.get("source_chunks", []):
                        st.markdown(f"**{chunk['label']}** "
                                    f"{'· page ' + str(chunk['page']) if chunk.get('page') else ''}")
                        st.code(chunk["content"][:600] + "…", language=None)
        else:
            st.markdown(msg["content"])


# ─── Input & processing ─────────────────────────────────────────────────────────

SAMPLE_QUESTIONS = [
    "What was Infosys revenue and operating margin for Q4 FY26?",
    "How has headcount changed across FY26? Show me a quarter-by-quarter breakdown.",
    "Compare Infosys's revenue growth across all four quarters of FY26.",
    "Give me a comprehensive analysis of Infosys's financial performance in FY25.",
    "What is the 52-week high and low for Infosys stock in FY26?",
    "How did deal wins trend across Q1 to Q4 FY26? Export as Excel.",
]

if not st.session_state.display_messages:
    st.markdown("**Try asking:**")
    cols = st.columns(2)
    for i, q in enumerate(SAMPLE_QUESTIONS):
        if cols[i % 2].button(q, key=f"sample_{i}", use_container_width=True):
            st.session_state["prefill_query"] = q
            st.rerun()

query = st.chat_input("Ask about Infosys financials…")

# Handle prefilled query from sample buttons
if not query and "prefill_query" in st.session_state:
    query = st.session_state.pop("prefill_query")

if query:
    # Display user message
    with st.chat_message("user"):
        st.markdown(query)

    st.session_state.display_messages.append({"role": "user", "content": query})
    st.session_state.history.append(Turn(role="user", text=query))

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer…"):
            response: EngineResponse = chat(query, st.session_state.history[:-1])

        # Display format badge
        st.markdown(_format_badge(response.output_format), unsafe_allow_html=True)

        # Display answer
        clean_answer = _clean_answer(response.answer)
        st.markdown(clean_answer)

        # Show citation pills
        if response.citations:
            st.markdown(_render_citations(response.citations), unsafe_allow_html=True)

        # Generate and offer download file
        file_path: Path | None = None
        if response.output_format == OutputFormat.PDF:
            with st.spinner("Generating PDF report…"):
                file_path = generate_pdf(response, query)
            with open(file_path, "rb") as f:
                st.download_button("📥 Download PDF Report", f.read(),
                                   file_name=file_path.name, mime="application/pdf")

        elif response.output_format == OutputFormat.EXCEL:
            with st.spinner("Building Excel workbook…"):
                file_path = generate_excel(response, query)
            with open(file_path, "rb") as f:
                st.download_button(
                    "📥 Download Excel",
                    f.read(),
                    file_name=file_path.name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

        # Source passage expander
        if response.sources:
            with st.expander("📎 View source passages"):
                for chunk in response.sources[:5]:
                    st.markdown(f"**{chunk.doc_label}**"
                                f"{' · page ' + str(chunk.page) if chunk.page else ''}"
                                f"{' · ' + chunk.section if chunk.section else ''}")
                    st.code(chunk.content[:600] + "…", language=None)

    # Update session state
    st.session_state.history.append(Turn(role="assistant", text=response.answer))
    st.session_state.display_messages.append({
        "role": "assistant",
        "content": response.answer,
        "format": response.output_format,
        "citations": response.citations,
        "file_path": str(file_path) if file_path else None,
        "source_chunks": [
            {"label": c.doc_label, "page": c.page,
             "section": c.section, "content": c.content}
            for c in response.sources[:5]
        ],
    })

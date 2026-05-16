"""
engine.py  –  The LLM layer: prompt construction, response parsing, format routing.

This is where the chatbot earns the label "intelligent":

1.  Query reformulation  – short follow-up questions ("what about Q2?") are
    rewritten to include conversation context before hitting the retriever.
2.  Grounded prompt      – the LLM sees only retrieved chunks, never raw docs,
    so it can't hallucinate facts outside the provided context.
3.  Citation enforcement – the system prompt instructs the model to tag every
    factual claim with [source:N] markers, which we later render as footnotes.
4.  Format decision      – the model emits a structured header that tells our
    router whether to render markdown, generate a PDF, or build an Excel file.
5.  Honest uncertainty   – the prompt explicitly asks the model to say "I
    cannot find this in the provided documents" rather than guessing.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from dotenv import load_dotenv
from google import genai
from google.genai import types
from groq import Groq
from retriever import retrieve, format_context_block, RetrievedChunk


# ─── Constants ─────────────────────────────────────────────────────────────────

LLM_MODEL = "llama-3.3-70b-versatile"
MAX_HISTORY_TURNS = 6      # keep last N user/assistant turns in the prompt
load_dotenv()


class OutputFormat(str, Enum):
    MARKDOWN = "markdown"
    PDF      = "pdf"
    EXCEL    = "excel"


# ─── Data models ───────────────────────────────────────────────────────────────

@dataclass
class Turn:
    role: str    # "user" | "assistant"
    text: str


@dataclass
class EngineResponse:
    answer: str                          # markdown-formatted answer text
    output_format: OutputFormat          # routing decision
    sources: list[RetrievedChunk]        # chunks that were used
    citations: list[dict[str, Any]]      # [{id, label, page}]
    raw_response: str                    # full model output, for debugging


# ─── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert financial analyst specialising in Infosys Limited.
You have been given a set of source passages (inside <source> tags) retrieved
from official Infosys documents: the FY2024-25 Annual Report, four quarterly
earnings press releases (Q1–Q4 FY26), a multi-year investor spreadsheet, and
FY26 BSE stock price data.

RULES YOU MUST FOLLOW:
1. Ground every factual claim in the provided sources. After each fact, write
   [source:N] where N is the id of the <source> tag it came from.
2. If the sources don't contain enough information, say exactly:
   "I could not find this in the provided documents. [source: none]"
   Do NOT invent numbers or quote figures you haven't seen.
3. Begin your response with a FORMAT line (no other text before it):
   FORMAT: markdown   – for quick factual answers, comparisons, short tables
   FORMAT: pdf        – for narrative summaries, multi-topic reports, analyses
                        longer than ~400 words, or when the user asks for a report
   FORMAT: excel      – for data-heavy outputs: time-series, multi-metric tables,
                        comparative data across quarters or years
4. After the FORMAT line, write your full answer in Markdown.
5. At the very end of your answer, add a CITATIONS section listing every source
   you referenced:
   ## Sources
   - [1] <doc label>, <page or section>
   - [2] ...
6. Treat tables in the sources as structured data — summarise key numbers rather
   than copying the raw table verbatim.
7. Be precise with numbers: always include units (USD million / ₹ crore /
   percentage points) and the period they refer to.
8. If the user asks a follow-up that references "the previous answer" or "that
   number", you have been given conversation history — use it.
"""


REWRITE_PROMPT = """You are a query reformulation assistant.

Given a conversation history and a new user message, rewrite the user message as
a fully self-contained search query that includes all necessary context from the
history. Do NOT answer the question — only output the reformulated query as a
single sentence. If the message is already self-contained, return it unchanged.

Conversation history:
{history}

New user message: {query}

Reformulated query:"""


# ─── Query reformulation ───────────────────────────────────────────────────────

def _reformulate_query(query: str, history: list[Turn]) -> str:
    """
    Expand short follow-ups using conversation history.

    Example:
      history: Q: "What was Infosys revenue in Q3 FY26?"  A: "$4.9 billion"
      query:   "How does that compare to Q2?"
      result:  "How does Infosys revenue in Q3 FY26 ($4.9B) compare to Q2 FY26?"
    """
    if not history:
        return query

    # Only reformulate if the query seems to reference prior context
    coreference_signals = re.compile(
        r"\b(that|this|those|it|its|they|them|the same|previous|prior|"
        r"compared to|vs|versus|last|earlier|above|below)\b",
        re.IGNORECASE,
    )
    if not coreference_signals.search(query):
        return query

    history_str = "\n".join(
        f"{t.role.upper()}: {t.text[:300]}" for t in history[-4:]
    )
    prompt = REWRITE_PROMPT.format(history=history_str, query=query)

    _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    response = _client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=200,
    )
    reformulated = response.choices[0].message.content.strip()
    print(f"[query rewrite] '{query}' → '{reformulated}'")
    return reformulated


# ─── Prompt construction ───────────────────────────────────────────────────────

def _build_prompt(
    query: str,
    context_block: str,
    history: list[Turn],
) -> str:
    history_section = ""
    if history:
        turns = history[-(MAX_HISTORY_TURNS * 2):]   # last N turns (each = 2 turns)
        lines = []
        for turn in turns:
            prefix = "User:" if turn.role == "user" else "Assistant:"
            lines.append(f"{prefix} {turn.text}")
        history_section = (
            "\n\n## Conversation so far\n"
            + "\n\n".join(lines)
        )

    return (
        f"## Retrieved source passages\n\n"
        f"{context_block}"
        f"{history_section}"
        f"\n\n## Current question\n\n{query}"
    )


# ─── Response parsing ──────────────────────────────────────────────────────────

def _parse_response(raw: str, sources: list[RetrievedChunk]) -> EngineResponse:
    # Extract FORMAT directive
    fmt_match = re.match(r"FORMAT:\s*(markdown|pdf|excel)", raw.strip(), re.IGNORECASE)
    if fmt_match:
        fmt_str = fmt_match.group(1).lower()
        answer = raw[fmt_match.end():].strip()
    else:
        fmt_str = "markdown"
        answer = raw.strip()

    output_format = OutputFormat(fmt_str)

    # Extract source references from answer: [source:N] → map to actual chunks
    ref_ids = [int(m) for m in re.findall(r"\[source:(\d+)\]", answer)]
    used_source_ids = sorted(set(ref_ids))

    citations = []
    for sid in used_source_ids:
        idx = sid - 1   # source ids are 1-indexed
        if 0 <= idx < len(sources):
            chunk = sources[idx]
            citations.append({
                "id":    sid,
                "label": chunk.doc_label,
                "page":  chunk.page,
                "section": chunk.section,
            })

    return EngineResponse(
        answer=answer,
        output_format=output_format,
        sources=sources,
        citations=citations,
        raw_response=raw,
    )


# ─── Main chat function ────────────────────────────────────────────────────────

def chat(query: str, history: list[Turn]) -> EngineResponse:
    """
    Process one user turn and return a grounded, cited, format-tagged response.

    Args:
        query:   The user's latest message.
        history: Previous turns in the conversation (oldest first).

    Returns:
        EngineResponse with answer, format decision, and cited sources.
    """
    # 1. Reformulate follow-up queries
    search_query = _reformulate_query(query, history)

    # 2. Retrieve relevant chunks
    chunks = retrieve(search_query)
    if not chunks:
        return EngineResponse(
            answer="I could not find any relevant information in the provided documents for your query.",
            output_format=OutputFormat.MARKDOWN,
            sources=[],
            citations=[],
            raw_response="",
        )

    # 3. Build grounded prompt
    context_block = format_context_block(chunks)
    prompt = _build_prompt(query, context_block, history)

    # 4. Call Groq
    _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    response = _client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=2048,
    )
    raw = response.choices[0].message.content

    # 5. Parse and return
    return _parse_response(raw, chunks)

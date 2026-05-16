"""
cli.py  –  Terminal-based chat interface.

Usage:
    python cli.py

Commands during chat:
    /quit or /exit   — end the session
    /clear           — clear conversation history
    /sources         — show source passages from last answer
    /save            — save last response to file (respects format decision)
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich import box

from engine import chat, Turn, OutputFormat, EngineResponse
from formatters import generate_pdf, generate_excel

load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

console = Console(width=100)

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║        INFOSYS FINANCIAL ANALYST  ·  CLI Interface          ║
║  Powered by Gemini 1.5 Flash  ·  RAG over 7 source docs     ║
╚══════════════════════════════════════════════════════════════╝
"""


def _print_answer(response: EngineResponse, query: str) -> None:
    fmt = response.output_format.value.upper()
    color_map = {"MARKDOWN": "blue", "PDF": "orange3", "EXCEL": "green"}
    color = color_map.get(fmt, "white")

    console.print(f"\n[{color}]▶ Output format: {fmt}[/{color}]")
    console.print(Markdown(response.answer))

    if response.citations:
        console.rule("[dim]Sources[/dim]")
        for c in response.citations:
            page_str = f", page {c['page']}" if c.get("page") else ""
            console.print(f"  [[cyan]{c['id']}[/cyan]] {c['label']}{page_str}")


def _save_output(response: EngineResponse, query: str) -> None:
    if response.output_format == OutputFormat.PDF:
        path = generate_pdf(response, query)
        console.print(f"[green]✓ PDF saved:[/green] {path}")
    elif response.output_format == OutputFormat.EXCEL:
        path = generate_excel(response, query)
        console.print(f"[green]✓ Excel saved:[/green] {path}")
    else:
        # Save markdown to txt
        out = Path("outputs")
        out.mkdir(exist_ok=True)
        from datetime import datetime
        fn = out / f"answer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        fn.write_text(response.answer)
        console.print(f"[green]✓ Markdown saved:[/green] {fn}")


def run_cli() -> None:
    console.print(BANNER, style="bold cyan")
    console.print("Type your question. Commands: [bold]/quit  /clear  /sources  /save[/bold]\n")

    history: list[Turn] = []
    last_response: EngineResponse | None = None
    last_query: str = ""

    while True:
        try:
            query = Prompt.ask("[bold blue]You[/bold blue]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not query:
            continue

        # ── Commands ──────────────────────────────────────────────────────────
        if query.lower() in {"/quit", "/exit", "quit", "exit"}:
            console.print("[dim]Goodbye.[/dim]")
            break

        if query.lower() == "/clear":
            history = []
            last_response = None
            console.print("[yellow]Conversation cleared.[/yellow]")
            continue

        if query.lower() == "/sources":
            if not last_response:
                console.print("[yellow]No previous answer to show sources for.[/yellow]")
            else:
                for chunk in last_response.sources[:6]:
                    console.print(Panel(
                        textwrap.shorten(chunk.content, width=400, placeholder="…"),
                        title=f"[cyan]{chunk.doc_label}[/cyan]"
                              + (f" · page {chunk.page}" if chunk.page else ""),
                        border_style="dim",
                    ))
            continue

        if query.lower() == "/save":
            if not last_response:
                console.print("[yellow]Nothing to save yet.[/yellow]")
            else:
                _save_output(last_response, last_query)
            continue

        # ── Normal query ──────────────────────────────────────────────────────
        with console.status("[bold green]Searching and generating answer…[/bold green]"):
            response = chat(query, history)

        _print_answer(response, query)

        # Auto-save non-markdown outputs
        if response.output_format != OutputFormat.MARKDOWN:
            _save_output(response, query)

        # Update history
        history.append(Turn(role="user", text=query))
        history.append(Turn(role="assistant", text=response.answer))
        last_response = response
        last_query = query
        console.print()


if __name__ == "__main__":
    run_cli()

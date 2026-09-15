#!/usr/bin/env python3
"""Enterprise RAG Engine interactive terminal CLI.

Usage:
    python -m src.cli                  # interactive chat
    python -m src.cli -q "question?"   # single question
    python -m src.cli --reindex        # rebuild index then chat
"""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from src.core.logger import logger
from src.engine.pipeline import get_rag_engine

console = Console()


def render_answer(res: dict) -> None:
    if res.get("abstained"):
        console.print("[bold yellow]Abstained — insufficient grounded evidence.[/bold yellow]")
    else:
        grounding = res.get("grounding") or {}
        unverified = grounding.get("unverified") or []
        if unverified:
            console.print(
                f"[bold yellow]Warning: {len(unverified)} citation(s) could not be "
                "verified against retrieved passages.[/bold yellow]"
            )
    console.print("\n[bold green]Assistant:[/bold green]")
    console.print(Markdown(res["response"]))

    if res.get("sources"):
        table = Table(
            title="Referenced Sources",
            border_style="dim",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("#", justify="right", style="dim", width=4)
        table.add_column("Document", style="cyan")
        table.add_column("Page", justify="center", style="yellow")
        table.add_column("Relevance Score", justify="right", style="green")
        for idx, src in enumerate(res["sources"], 1):
            table.add_row(str(idx), src["document"], f"Page {src['page']}", f"{src['score']:.2f}")
        console.print(table)

    console.print(f"[dim]Latency: {res['latency_ms'] / 1000:.2f}s | Model: {res['model']}[/dim]\n")


def run_interactive(engine, top_n: int = 4) -> None:
    console.print(
        "[dim]Type your question and press Enter. Enter 'exit' or 'quit' to exit.[/dim]\n"
    )
    while True:
        try:
            question = console.input("[bold blue]You:[/bold blue] ").strip()
            if not question:
                continue
            if question.lower() in ("exit", "quit", "q"):
                console.print("[yellow]Exiting Enterprise RAG Engine CLI. Goodbye![/yellow]")
                break

            with console.status("[bold magenta]Retrieving and generating answer...[/bold magenta]"):
                res = engine.query(question, top_n=top_n)
            render_answer(res)

        except KeyboardInterrupt:
            console.print("\n[yellow]Session interrupted. Exiting.[/yellow]")
            break
        except Exception as e:  # keep the REPL alive on per-query failures
            logger.error(f"Query failed: {e}")
            console.print(f"[red]Error: {e}[/red]\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enterprise RAG Engine terminal chat")
    parser.add_argument("-q", "--question", default=None, help="Ask a single question and exit")
    parser.add_argument(
        "--top-n", type=int, default=4, help="Retrieved passages to use (default: 4)"
    )
    parser.add_argument("--doc", default=None, help="Restrict context to a single document name")
    parser.add_argument("--reindex", action="store_true", help="Rebuild the index before querying")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    console.print(
        Panel.fit(
            "[bold cyan]Enterprise RAG Engine Terminal[/bold cyan]\n"
            "[dim]Hybrid Dense-Sparse Retrieval + Neural Re-ranking[/dim]",
            border_style="blue",
        )
    )

    with console.status("[bold green]Loading knowledge base and models...[/bold green]"):
        engine = get_rag_engine()
        if args.reindex:
            engine.reindex()

    console.print(
        f"[green]✓ Knowledge base ready![/green] Active model: "
        f"[bold yellow]{engine.llm.active_model_name}[/bold yellow]"
    )

    if args.question:
        res = engine.query(args.question, top_n=args.top_n, doc_name=args.doc)
        render_answer(res)
        return 0

    run_interactive(engine, top_n=args.top_n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

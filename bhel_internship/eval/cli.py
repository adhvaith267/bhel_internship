#!/usr/bin/env python3
"""Eval CLI: `python -m bhel_internship eval goldens.json [--threshold 0.5]`.

Exit code is 0 when all cases pass, 1 otherwise (CI-friendly).
"""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from bhel_internship.eval.harness import load_goldens, run_eval

console = Console()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enterprise RAG Engine eval harness")
    parser.add_argument("goldens", help="Path to goldens JSON file")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Minimum keyword coverage to pass (default: 0.5)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    goldens = load_goldens(args.goldens)

    from bhel_internship.engine.pipeline import get_rag_engine

    with console.status("[bold green]Running eval...[/bold green]"):
        engine = get_rag_engine()
        report = run_eval(engine, goldens, keyword_threshold=args.threshold)

    table = Table(title="Eval Results", show_header=True, header_style="bold cyan")
    table.add_column("#", justify="right", style="dim", width=4)
    table.add_column("Question")
    table.add_column("Ctx", justify="center")
    table.add_column("Cover", justify="right")
    table.add_column("Ground", justify="right")
    table.add_column("Result", justify="center")
    for idx, case in enumerate(report["cases"], 1):
        table.add_row(
            str(idx),
            case["question"][:60],
            "✓" if case["context_hit"] else "✗",
            f"{case['keyword_coverage']:.2f}",
            f"{case['grounding_rate']:.2f}",
            "[green]PASS[/green]" if case["passed"] else "[red]FAIL[/red]",
        )
    console.print(table)
    console.print(f"[dim]{json.dumps(report['summary'], indent=2)}[/dim]")

    return 0 if report["summary"]["passed"] == report["summary"]["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

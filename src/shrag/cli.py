"""Command-line interface: shrag ingest | ask | chat | graph | eval."""

import argparse
import json
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from shrag.config import PROJECT_ROOT, load_settings

console = Console()

STEP_STYLE = {
    "retrieve": "cyan", "grade_documents": "blue", "generate": "magenta", "critique": "yellow",
    "rewrite_query": "bright_red", "finalize": "green", "fallback": "red",
}
STATUS_STYLE = {"answered": "green", "partial": "yellow", "insufficient": "red"}


def cmd_ingest(args):
    from shrag.vectorstore import ingest

    settings = load_settings()
    with console.status("Embedding documents locally (first run downloads the model)..."):
        n_docs, n_chunks = ingest(Path(args.path), settings, reset=not args.append)
    console.print(f"[green]Indexed {n_docs} documents as {n_chunks} chunks[/] in {settings.chroma_dir}")


def _load_rag(args):
    from shrag.pipeline import SelfHealingRAG

    overrides = {k: v for k, v in (("max_retrievals", args.max_retrievals),
                                   ("max_generations", args.max_generations)) if v}
    return SelfHealingRAG.from_settings(**overrides)


def _print_result(result):
    style = STATUS_STYLE.get(result.status, "white")
    console.print(Panel(Markdown(result.answer), title=f"[{style}]{result.status.upper()}[/]",
                        border_style=style))
    if result.sources:
        console.print("[dim]Sources:[/] " + ", ".join(result.sources))
    console.print(f"[dim]{result.retrieval_attempts} retrieval round(s), "
                  f"{result.llm_calls} LLM calls, {result.seconds}s[/]")


def _answer(rag, question, show_trace):
    result = None
    with console.status("Thinking...") as status:
        for node, update in rag.stream(question):
            if node == "result":
                result = update
                continue
            status.update(f"{node}...")
            if show_trace:
                for event in update.get("trace", []):
                    style = STEP_STYLE.get(event["step"], "white")
                    console.print(f"[{style}]● {event['step']:<16}[/] {event['summary']}")
    _print_result(result)
    return result


def cmd_ask(args):
    rag = _load_rag(args)
    if args.baseline:
        base = rag.baseline(args.question)
        console.print(Panel(Markdown(base.answer), title="[dim]Plain RAG (no critic)[/]"))
    _answer(rag, args.question, show_trace=not args.quiet)


def cmd_chat(args):
    rag = _load_rag(args)
    console.print("[bold]Self-Healing RAG[/]: ask about the knowledge base. Ctrl+C or 'exit' to quit.")
    while True:
        try:
            question = console.input("\n[bold cyan]You>[/] ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            _answer(rag, question, show_trace=not args.quiet)


def cmd_graph(args):
    from langchain_core.language_models import FakeListChatModel

    from shrag.graph import build_graph
    from shrag.nodes import RAGNodes

    fake = FakeListChatModel(responses=[""])
    mermaid = build_graph(RAGNodes(None, fake, fake)).get_graph().draw_mermaid()
    if args.out:
        Path(args.out).write_text(f"```mermaid\n{mermaid}```\n")
        console.print(f"Wrote {args.out}")
    else:
        print(mermaid)


# ----- evaluation -------------------------------------------------------------

REFUSAL_MARKERS = (
    "don't have enough information", "do not have enough information", "not mentioned",
    "does not mention", "doesn't mention", "not provided", "no information", "not specified",
    "does not specify", "doesn't specify", "not included", "not available in", "cannot find",
    "can't find", "does not contain", "doesn't contain", "not stated", "not covered",
    "does not cover", "doesn't cover", "insufficient_context", "unable to answer",
)


def declined(answer: str) -> bool:
    text = answer.lower().replace("’", "'")
    return any(marker in text for marker in REFUSAL_MARKERS)


def score(case: dict, result) -> dict:
    answer = result.answer.lower().replace(",", "")
    if case["answerable"]:
        hits = [kw for kw in case["expected"] if kw.lower().replace(",", "") in answer]
        correct = len(hits) == len(case["expected"])
        return {"correct": correct, "hallucinated": False}
    refused = result.status == "insufficient" or declined(result.answer)
    return {"correct": refused, "hallucinated": not refused}


def cmd_eval(args):
    cases = json.loads((PROJECT_ROOT / "eval" / "questions.json").read_text())
    if args.limit:
        cases = cases[: args.limit]
    rag = _load_rag(args)
    systems = ["baseline", "self_healing"] if args.system == "both" else [args.system]
    rows = []
    for i, case in enumerate(cases, start=1):
        row = {"id": case["id"], "question": case["question"], "answerable": case["answerable"]}
        for system in systems:
            with console.status(f"[{i}/{len(cases)}] {system}: {case['question']}"):
                try:
                    result = rag.baseline(case["question"]) if system == "baseline" else rag.ask(case["question"])
                    row[system] = {"answer": result.answer, "status": result.status,
                                   "llm_calls": result.llm_calls, "seconds": result.seconds,
                                   "retrieval_attempts": result.retrieval_attempts,
                                   **score(case, result)}
                except Exception as exc:  # keep going if one free-model call dies
                    row[system] = {"error": str(exc)[:300], "correct": False, "hallucinated": False}
            mark = "[green]✓[/]" if row[system]["correct"] else "[red]✗[/]"
            console.print(f"{mark} {system:<13} {case['id']}")
        rows.append(row)
        time.sleep(args.pause)

    out_dir = PROJECT_ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(rows, indent=2))

    table = Table(title="Evaluation")
    table.add_column("Metric")
    for system in systems:
        table.add_column(system)

    def pct(system, pred, subset):
        items = [r[system] for r in rows if subset(r)]
        return f"{100 * sum(pred(x) for x in items) / len(items):.0f}%" if items else "-"

    table.add_row("Accuracy (answerable)", *[pct(s, lambda x: x["correct"], lambda r: r["answerable"]) for s in systems])
    table.add_row("Correctly declined (unanswerable)", *[pct(s, lambda x: x["correct"], lambda r: not r["answerable"]) for s in systems])
    table.add_row("Hallucination rate (unanswerable)", *[pct(s, lambda x: x["hallucinated"], lambda r: not r["answerable"]) for s in systems])
    table.add_section()
    table.add_row("Avg LLM calls", *[f"{sum(r[s].get('llm_calls', 0) for r in rows) / len(rows):.1f}" for s in systems])
    table.add_row("Avg seconds", *[f"{sum(r[s].get('seconds', 0) for r in rows) / len(rows):.1f}" for s in systems])
    console.print(table)
    console.print(f"[dim]Full results: {out}[/]")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="shrag", description="Self-healing RAG pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="index documents into the vector store")
    p.add_argument("path", nargs="?", default=str(PROJECT_ROOT / "data" / "docs"))
    p.add_argument("--append", action="store_true", help="add to the index instead of rebuilding it")
    p.set_defaults(func=cmd_ingest)

    for name, func, help_text in (("ask", cmd_ask, "ask one question"),
                                  ("chat", cmd_chat, "interactive questions"),
                                  ("eval", cmd_eval, "compare against plain RAG")):
        p = sub.add_parser(name, help=help_text)
        if name == "ask":
            p.add_argument("question")
            p.add_argument("--baseline", action="store_true", help="also show plain RAG's answer")
        if name in ("ask", "chat"):
            p.add_argument("-q", "--quiet", action="store_true", help="hide the step-by-step trace")
        if name == "eval":
            p.add_argument("--limit", type=int, help="only run the first N questions")
            p.add_argument("--system", choices=["both", "baseline", "self_healing"], default="both")
            p.add_argument("--pause", type=float, default=2.0, help="seconds between questions (rate limits)")
        p.add_argument("--max-retrievals", type=int)
        p.add_argument("--max-generations", type=int)
        p.set_defaults(func=func)

    p = sub.add_parser("graph", help="print the LangGraph workflow as a Mermaid diagram")
    p.add_argument("--out", help="write to a markdown file")
    p.set_defaults(func=cmd_graph)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/]")
        sys.exit(1)


if __name__ == "__main__":
    main()

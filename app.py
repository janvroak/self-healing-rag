"""Streamlit frontend for the self-healing RAG pipeline.

Run with:  uv run streamlit run app.py
"""

import json
import re
from pathlib import Path

import streamlit as st

from shrag.config import PROJECT_ROOT, load_settings
from shrag.pipeline import RAGResult, SelfHealingRAG
from shrag.vectorstore import SUPPORTED, ingest, knowledge_base_stats

st.set_page_config(page_title="Self-Healing RAG", page_icon="🩹", layout="wide")

SAMPLE_DOCS = PROJECT_ROOT / "data" / "docs"
UPLOADS = PROJECT_ROOT / "data" / "uploads"
EXAMPLES = [
    "How many unused leave days can I carry over, and when do they expire?",
    "Can I work from Bali for two months? Who do I need to tell?",
    "What was the maximum flight time of the Kestrel K1?",
    "Who is the CFO of Veridian Dynamics?",
]
STEPS = {  # node -> (icon, label)
    "retrieve": ("🔎", "Retrieve"),
    "grade_documents": ("🧹", "Grade chunks"),
    "generate": ("✍️", "Generate"),
    "critique": ("🧐", "Critic"),
    "rewrite_query": ("🔁", "Rewrite query"),
    "finalize": ("✅", "Verified"),
    "fallback": ("🛑", "Fallback"),
}
STATUS = {  # status -> (badge text, colour, icon)
    "answered": ("Verified answer", "green", ":material/verified:"),
    "partial": ("Partial answer", "orange", ":material/rule:"),
    "insufficient": ("Not enough information", "red", ":material/block:"),
}

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 1100px;}
      [data-testid="stSidebar"] .block-container {padding-top: 1.5rem;}
      .hero h1 {margin-bottom: 0.2rem;}
      .hero p {color: var(--text-muted, #6b7280); font-size: 1.05rem; margin-top: 0;}
      .step {display: flex; gap: .6rem; align-items: baseline; padding: .15rem 0;}
      .step .name {font-weight: 600; min-width: 7.5rem;}
      .step .text {opacity: .85;}
      .chunk-id {font-family: ui-monospace, monospace; font-size: .85rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

settings = load_settings()


# ----- cached resources ------------------------------------------------------

@st.cache_resource(show_spinner="Loading models and vector store...")
def get_rag(max_retrievals: int, max_generations: int) -> SelfHealingRAG:
    return SelfHealingRAG.from_settings(max_retrievals=max_retrievals, max_generations=max_generations)


@st.cache_data(show_spinner=False)
def get_stats(_version: int) -> dict:
    return knowledge_base_stats(settings)


def latest_eval() -> list[dict] | None:
    results = sorted((PROJECT_ROOT / "eval" / "results").glob("eval_*.json"))
    return json.loads(results[-1].read_text()) if results else None


def refresh_index():
    st.session_state.kb_version = st.session_state.get("kb_version", 0) + 1
    get_rag.clear()


# ----- sidebar ---------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🩹 Self-Healing RAG")
    st.caption("RAG that fact-checks itself before it answers.")

    st.markdown("#### Knowledge base")
    stats = get_stats(st.session_state.get("kb_version", 0))
    c1, c2 = st.columns(2)
    c1.metric("Files", len(stats["sources"]))
    c2.metric("Chunks", stats["chunks"])
    with st.expander("Indexed files"):
        for name, count in stats["sources"].items():
            st.markdown(f"<span class='chunk-id'>{name}</span> · {count} chunks", unsafe_allow_html=True)

    uploads = st.file_uploader("Add your own documents", type=[s.lstrip(".") for s in SUPPORTED],
                               accept_multiple_files=True)
    if uploads and st.button("Index uploaded files", type="primary", width="stretch"):
        UPLOADS.mkdir(parents=True, exist_ok=True)
        for f in uploads:
            (UPLOADS / Path(f.name).name).write_bytes(f.getvalue())
        with st.spinner("Embedding locally..."):
            for f in uploads:
                ingest(UPLOADS / Path(f.name).name, settings, reset=False)
        refresh_index()
        st.toast(f"Indexed {len(uploads)} file(s)", icon="✅")
        st.rerun()
    if st.button("Reset to sample documents", width="stretch"):
        with st.spinner("Rebuilding index..."):
            ingest(SAMPLE_DOCS, settings, reset=True)
        refresh_index()
        st.toast("Index rebuilt from the sample documents", icon="🔄")
        st.rerun()

    st.markdown("#### Self-correction loop")
    max_retrievals = st.slider("Max retrieval rounds", 1, 5, settings.max_retrievals,
                               help="How many times it may search again with a rewritten query.")
    max_generations = st.slider("Max drafts per round", 1, 4, settings.max_generations,
                                help="How many times it may rewrite an answer the critic rejected.")
    compare = st.toggle("Compare with plain RAG", help="Also run a normal retrieve-then-generate "
                        "pipeline with no critic, side by side.")

    st.markdown("#### Models (all free)")

    def short(model: str) -> str:
        return model.split("/")[-1].removesuffix(":free")

    st.markdown(
        f"<small>✍️ Generator: <b>{short(settings.generator_models[0])}</b><br>"
        f"🧐 Critic: <b>{short(settings.critic_models[0])}</b><br>"
        f"💻 Fallback: <b>{settings.ollama_model}</b> (Ollama, local)<br>"
        f"🧬 Embeddings: <b>{short(settings.embedding_model)}</b> (local)</small>",
        unsafe_allow_html=True,
    )
    st.write("")
    if st.button("Clear chat", icon=":material/delete:", width="stretch"):
        st.session_state.messages = []
        st.rerun()


# ----- rendering helpers -----------------------------------------------------

def with_citation_badges(answer: str) -> str:
    """Turn [2] citations into small badges."""
    return re.sub(r"\s*\[(\d+)\]", r" :blue-badge[\1]", answer)


def render_step(event: dict):
    icon, label = STEPS.get(event["step"], ("•", event["step"]))
    st.markdown(f"<div class='step'><span>{icon}</span><span class='name'>{label}</span>"
                f"<span class='text'>{event['summary']}</span></div>", unsafe_allow_html=True)


def render_trace(result: RAGResult):
    for event in result.trace:
        render_step(event)
        details = event["details"]
        if event["step"] == "generate" and details.get("answer"):
            with st.container(border=True):
                st.caption("Draft")
                st.markdown(with_citation_badges(details["answer"]))


def render_critique(result: RAGResult):
    claims = result.critique.get("claims", [])
    if not claims:
        st.caption("The critic did not run (no relevant documents were found).")
        return
    st.dataframe(
        [{"": "✅" if c["supported"] else "❌", "Claim": c["claim"],
          "Evidence": f"[{c['evidence']}]" if c.get("evidence") else "—"} for c in claims],
        hide_index=True, width="stretch",
        column_config={"": st.column_config.TextColumn(width="small"),
                       "Evidence": st.column_config.TextColumn(width="small")},
    )
    if result.critique.get("missing"):
        st.caption(f"**Missing from the documents:** {result.critique['missing']}")


def render_sources(result: RAGResult):
    if not result.context:
        st.caption("No sources: nothing relevant was found.")
        return
    for i, chunk in enumerate(result.context, start=1):
        with st.container(border=True):
            score = f" · similarity {chunk['score']:.2f}" if chunk.get("score") is not None else ""
            st.markdown(f":blue-badge[{i}] <span class='chunk-id'>{chunk['id']}</span>{score}",
                        unsafe_allow_html=True)
            st.caption(chunk["text"])


def render_result(result: RAGResult):
    text, colour, icon = STATUS[result.status]
    st.badge(text, color=colour, icon=icon)
    st.markdown(with_citation_badges(result.answer))
    st.caption(f"{result.retrieval_attempts} retrieval round(s) · {result.llm_calls} LLM calls · "
               f"{result.seconds}s · searched: " + " → ".join(f"“{q}”" for q in result.queries))
    trace_tab, critic_tab, sources_tab = st.tabs(["🧭 Trace", "🧐 Critic report", "📚 Sources"])
    with trace_tab:
        render_trace(result)
    with critic_tab:
        render_critique(result)
    with sources_tab:
        render_sources(result)


def render_baseline(result: RAGResult):
    st.badge("Unchecked answer", color="gray", icon=":material/help:")
    st.markdown(result.answer)
    st.caption(f"1 retrieval · 1 LLM call · {result.seconds}s · no critic, no retries")
    with st.expander("📚 Retrieved chunks"):
        render_sources(result)


def render_assistant(message: dict):
    if message.get("error"):
        st.error(message["error"])
    elif message.get("baseline"):
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("##### 🩹 Self-healing")
            render_result(message["result"])
        with right:
            st.markdown("##### Plain RAG")
            render_baseline(message["baseline"])
    else:
        render_result(message["result"])


# ----- pages -------------------------------------------------------------------

def ask_page():
    messages = st.session_state.setdefault("messages", [])

    if not messages:
        st.markdown("<div class='hero'><h1>Ask the knowledge base</h1>"
                    "<p>Every answer is checked claim by claim against the source documents. "
                    "If it can't be verified, the pipeline searches again, or tells you it doesn't "
                    "know.</p></div>", unsafe_allow_html=True)
        st.caption("Try one: the last two aren't in the documents, so watch it refuse to guess.")
        cols = st.columns(2)
        for i, example in enumerate(EXAMPLES):
            if cols[i % 2].button(example, key=f"ex{i}", width="stretch"):
                st.session_state.pending = example
                st.rerun()

    for message in messages:
        with st.chat_message(message["role"], avatar="🧑" if message["role"] == "user" else "🩹"):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                render_assistant(message)

    question = st.chat_input("Ask about the documents...") or st.session_state.pop("pending", None)
    if not question:
        return

    messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🩹"):
        try:
            rag = get_rag(max_retrievals, max_generations)
            result = None
            with st.status("Retrieving...", expanded=True) as status:
                for node, update in rag.stream(question):
                    if node == "result":
                        result = update
                        continue
                    status.update(label=f"{STEPS.get(node, ('', node))[1]}...")
                    for event in update.get("trace", []):
                        render_step(event)
                status.update(label=STATUS[result.status][0], expanded=False,
                              state="complete" if result.status == "answered" else "error")
            baseline = None
            if compare:
                with st.spinner("Running plain RAG for comparison..."):
                    baseline = rag.baseline(question)
            message = {"role": "assistant", "result": result, "baseline": baseline}
        except Exception as exc:  # show provider/index problems in the chat, not a stack trace
            message = {"role": "assistant", "error": f"Something went wrong: {exc}"}
        messages.append(message)
        render_assistant(message)


GRAPH_DOT = """
digraph {
  rankdir=TB; bgcolor="transparent"; nodesep=0.5; ranksep=0.45;
  node [shape=box, style="rounded,filled", fillcolor="#eef2ff", color="#6366f1", fontname="Helvetica", fontsize=11, margin="0.2,0.06"];
  edge [color="#94a3b8", fontname="Helvetica", fontsize=9, fontcolor="#64748b"];
  start [label="question", shape=oval, fillcolor="#e2e8f0", color="#94a3b8"];
  done [label="verified answer", shape=oval, fillcolor="#dcfce7", color="#16a34a"];
  refuse [label="partial answer /\\n“not enough info”", shape=oval, fillcolor="#fee2e2", color="#dc2626"];
  start -> retrieve -> grade_documents;
  grade_documents -> generate [label="relevant"];
  grade_documents -> rewrite_query [label="none relevant"];
  generate -> critique [label="draft"];
  generate -> rewrite_query [label="insufficient"];
  critique -> finalize [label="grounded + complete"];
  critique -> generate [label="hallucinated"];
  critique -> rewrite_query [label="incomplete"];
  rewrite_query -> retrieve [label="retry"];
  rewrite_query -> fallback [label="limit hit", style=dashed];
  finalize -> done; fallback -> refuse;
}
"""


def how_page():
    st.markdown("## How it works")
    st.write("The pipeline is a **cyclical LangGraph state machine** instead of a linear chain. "
             "A separate critic model checks every draft, and a rejected draft sends the graph back "
             "to an earlier step. Retry counters in the graph state make sure the loop always ends.")
    diagram, cards = st.columns([3, 2], gap="large")
    with diagram:
        st.graphviz_chart(GRAPH_DOT, width="stretch")
    with cards:
        with st.container(border=True):
            st.markdown("**🔎 Retrieve and grade**")
            st.caption("Vector search in Chroma with local embeddings. An LLM drops off-topic chunks. "
                       "Every retry fetches more chunks, and relevant chunks build up across rounds.")
        with st.container(border=True):
            st.markdown("**🧐 Critic agent**")
            st.caption("A different, larger model splits the answer into separate claims and checks "
                       "each one against the chunks. Citations to chunks that don't exist are "
                       "rejected in code. An answer it can't verify counts as rejected.")
        with st.container(border=True):
            st.markdown("**🔁 Self-correction**")
            st.caption("Hallucinated → rewrite with the critic's feedback. Incomplete → rewrite the "
                       "search query and retrieve again. Out of retries → partial answer or "
                       "“I don't have enough information”.")

    st.markdown("#### Evaluation")
    rows = latest_eval()
    if not rows:
        st.caption("No evaluation results yet. Run `uv run shrag eval`.")
        return

    def rate(system, key, answerable):
        items = [r[system] for r in rows if r["answerable"] == answerable and system in r]
        return 100 * sum(bool(x.get(key)) for x in items) / len(items) if items else 0

    m = st.columns(3)
    m[0].metric("Accuracy on answerable questions", f"{rate('self_healing', 'correct', True):.0f}%",
                f"{rate('self_healing', 'correct', True) - rate('baseline', 'correct', True):+.0f} pts vs plain RAG")
    m[1].metric("Hallucination rate on unanswerable", f"{rate('self_healing', 'hallucinated', False):.0f}%",
                f"{rate('self_healing', 'hallucinated', False) - rate('baseline', 'hallucinated', False):+.0f} pts vs plain RAG",
                delta_color="inverse")
    m[2].metric("Questions", len(rows))
    st.dataframe(
        [{"Question": r["question"], "Answerable": "yes" if r["answerable"] else "no",
          "Plain RAG": "✅" if r.get("baseline", {}).get("correct") else "❌",
          "Self-healing": "✅" if r.get("self_healing", {}).get("correct") else "❌",
          "Self-healing status": r.get("self_healing", {}).get("status", "error")} for r in rows],
        hide_index=True, width="stretch",
    )


try:
    get_rag(max_retrievals, max_generations)
except RuntimeError as exc:
    st.error(f"{exc}")
    st.stop()

pg = st.navigation([st.Page(ask_page, title="Ask", icon="💬", default=True),
                    st.Page(how_page, title="How it works", icon="⚙️", url_path="how-it-works")], position="top")
pg.run()

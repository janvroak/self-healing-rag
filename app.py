"""Streamlit UI: watch the self-healing loop work step by step.

Run with:  uv run streamlit run app.py
"""

import streamlit as st

from shrag.config import load_settings
from shrag.pipeline import SelfHealingRAG

st.set_page_config(page_title="Self-Healing RAG", page_icon="🩹", layout="wide")

STEP_ICON = {
    "retrieve": "🔎", "grade_documents": "🧹", "generate": "✍️", "critique": "🧐",
    "rewrite_query": "🔁", "finalize": "✅", "fallback": "🛑",
}
STATUS_BOX = {"answered": st.success, "partial": st.warning, "insufficient": st.error}

settings = load_settings()

with st.sidebar:
    st.header("Loop limits")
    max_retrievals = st.slider("Max retrieval rounds", 1, 5, settings.max_retrievals)
    max_generations = st.slider("Max answers per round", 1, 4, settings.max_generations)
    compare = st.toggle("Compare with plain RAG", value=False)
    st.caption(f"Generator: `{settings.generator_models[0]}`  \nCritic: `{settings.critic_models[0]}`  \n"
               f"Fallback: `{settings.ollama_model}` (Ollama)  \nEmbeddings: `{settings.embedding_model}`")


@st.cache_resource(show_spinner="Loading models and vector store...")
def get_rag(max_retrievals: int, max_generations: int) -> SelfHealingRAG:
    return SelfHealingRAG.from_settings(max_retrievals=max_retrievals, max_generations=max_generations)


st.title("🩹 Self-Healing RAG")
st.write("A RAG pipeline that fact-checks its own answers against the retrieved documents, "
         "retries with a better query when they fall short, and says *\"I don't know\"* instead of "
         "making things up.")

try:
    rag = get_rag(max_retrievals, max_generations)
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

examples = [
    "How many unused leave days can I carry over, and when do they expire?",
    "Can I work from Bali for two months? Who do I need to tell?",
    "What was the maximum flight time of the Kestrel K1?",
    "Who is the CFO of Veridian Dynamics?",
]
choice = st.selectbox("Try an example", ["(type your own)"] + examples)
question = st.text_input("Question", value="" if choice == "(type your own)" else choice)

if st.button("Ask", type="primary", disabled=not question):
    cols = st.columns(2) if compare else [st.container()]

    with cols[0]:
        st.subheader("Self-healing pipeline")
        result = None
        with st.status("Running the graph...", expanded=True) as status:
            for node, update in rag.stream(question):
                if node == "result":
                    result = update
                    continue
                for event in update.get("trace", []):
                    st.markdown(f"{STEP_ICON.get(event['step'], '•')} **{event['step']}**: {event['summary']}")
                    details = event["details"]
                    if event["step"] == "generate" and details.get("answer"):
                        with st.expander("Draft answer"):
                            st.write(details["answer"])
                    if event["step"] == "critique" and details.get("unsupported"):
                        st.caption("Unsupported claims: " + "; ".join(details["unsupported"]))
            status.update(label=f"Done: {result.status}", state="complete", expanded=False)

        STATUS_BOX[result.status](result.answer)
        st.caption(f"{result.retrieval_attempts} retrieval round(s) · {result.llm_calls} LLM calls · "
                   f"{result.seconds}s")
        if result.sources:
            st.caption("Sources: " + ", ".join(f"`{s}`" for s in result.sources))

    if compare:
        with cols[1]:
            st.subheader("Plain RAG (no critic)")
            with st.spinner("Retrieve → generate..."):
                base = rag.baseline(question)
            st.info(base.answer)
            st.caption(f"1 LLM call · {base.seconds}s")

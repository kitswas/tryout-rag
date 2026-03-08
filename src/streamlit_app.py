import os
from concurrent.futures import ThreadPoolExecutor

import streamlit as st


@st.cache_resource
def get_rag_pipeline():
    model_id = os.getenv("RAG_MODEL", "HuggingFaceTB/SmolLM2-360M-Instruct")
    try:
        from src.rag import build_rag

        rag = build_rag(model_id)
        return rag
    except Exception as e:
        st.error(f"Failed to build RAG pipeline: {e}")
        raise


st.set_page_config(page_title="RAG UI (Streamlit)")
st.title("RAG UI — Streamlit")

st.write("Ask questions and get answers using the local RAG pipeline.")

with st.expander("Pipeline status"):
    if "rag" not in st.session_state:
        st.write("Initializing RAG pipeline (may take a while)...")
    else:
        st.write("Pipeline ready.")

question = st.text_input("Question about Virtual Gamepad", value="")
ask = st.button("Ask")

rag = None
try:
    rag = get_rag_pipeline()
    st.session_state["rag"] = True
except Exception:
    rag = None

if ask:
    if not question.strip():
        st.warning("Please enter a question.")
    elif rag is None:
        st.error("RAG pipeline not available yet. Check logs or try again.")
    else:
        with st.spinner("Generating answer..."):
            # Run the blocking call in a thread to keep UI responsive
            with ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(rag.invoke, question)
                answer = future.result()

        st.subheader("Answer")
        st.code(answer.strip())

st.markdown("---")
st.caption(
    "Note: building the pipeline downloads models and crawls the configured sitemap; expect high CPU/memory usage on first run."
)

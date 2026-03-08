from concurrent.futures import ThreadPoolExecutor
import streamlit as st


@st.cache_resource
def get_rag_pipeline(model_id, sitemap_url):
    try:
        from src.rag import build_rag

        rag = build_rag(model_id, sitemap_url)
        return rag
    except Exception as e:
        st.error(f"Failed to build RAG pipeline: {e}")
        return None


st.set_page_config(page_title="RAG UI (Streamlit)")
st.title("RAG UI")

st.write("Ask questions and get answers using the local RAG pipeline.")

# Sidebar for configuration
st.sidebar.header("Configuration")
sitemap_url_input = st.sidebar.text_input(
    "Sitemap URL",
    value="https://kitswas.github.io/VirtualGamePad/sitemap.xml",
    help="The URL of the sitemap to crawl for documents.",
)
model_id_input = st.sidebar.selectbox(
    "Model",
    options=["google/gemma-3-1b-it", "HuggingFaceTB/SmolLM2-360M-Instruct"],
    index=0,
    help="The Hugging Face model ID to use for generation.",
)

rag = None

# Use session state to store the "active" config
if "active_model" not in st.session_state:
    st.session_state["active_model"] = model_id_input
if "active_sitemap" not in st.session_state:
    st.session_state["active_sitemap"] = sitemap_url_input

if st.sidebar.button("Build/Update Pipeline"):
    st.session_state["active_model"] = model_id_input
    st.session_state["active_sitemap"] = sitemap_url_input
    # Clearing the cache for get_rag_pipeline is tricky, but Streamlit
    # will handle it automatically because the arguments passed to it
    # will change.

active_model = st.session_state["active_model"]
active_sitemap = st.session_state["active_sitemap"]

with st.expander("Pipeline status"):
    st.write(f"**Current Model:** {active_model}")
    st.write(f"**Current Sitemap:** {active_sitemap}")
    if "rag_ready" not in st.session_state:
        st.write("Initializing RAG pipeline (may take a while)...")
    else:
        st.write("Pipeline ready.")

question = st.text_input("Question about Virtual Gamepad", value="")
ask = st.button("Ask")

try:
    # Use the active config from session state
    rag = get_rag_pipeline(active_model, active_sitemap)
    if rag:
        st.session_state["rag_ready"] = True
    else:
        st.session_state["rag_ready"] = False
except Exception:
    rag = None

if ask:
    if not question.strip():
        st.warning("Please enter a question.")
    elif rag is None:
        st.error("RAG pipeline not available yet. Check logs or try again.")
    else:
        with st.spinner(f"Generating answer using {active_model}..."):
            # Run the blocking call in a thread to keep UI responsive
            with ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(rag.invoke, question)
                answer = future.result()

        st.subheader("Answer")
        st.code(answer.strip(), language="markdown", wrap_lines=True)

st.markdown("---")
st.caption(
    "Note: building the pipeline downloads models and crawls the configured sitemap; expect high CPU/memory usage on first run."
)

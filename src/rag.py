import os
import warnings

from langchain_community.document_loaders.sitemap import SitemapLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEmbeddings,
    HuggingFacePipeline,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Suppress HuggingFace/Langchain warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def build_rag(model_id, sitemap_url):
    # 1. Load documents directly from the sitemap
    print(f"🌐 Crawling sitemap: {sitemap_url}")

    # SitemapLoader parses the XML and extracts text from all listed URLs
    loader = SitemapLoader(web_path=sitemap_url)
    docs = loader.load()
    print(f"✅ Loaded {len(docs)} pages.")

    # 2. Split documents into manageable chunks
    print("✂️ Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # 3. Create Local Vector Store (FAISS)
    print("🧠 Creating vector embeddings...")
    # all-MiniLM-L6-v2 is a fast, lightweight local embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # 4. Set up Local LLM
    print("🤖 Loading local HuggingFace LLM...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.1,  # Low temperature for more factual responses
        do_sample=True,
        repetition_penalty=1.1,
        return_full_text=False,
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    chat_model = ChatHuggingFace(llm=llm)

    # 1. Define a cleaner prompt template tailored for instruction models
    messages = [
        (
            "system",
            "You are a helpful assistant. Answer the question based only on the following context. If the context contains technical tradeoffs or steps, explain them clearly and logically. Keep your answer strictly concise. Do not add any extra conversational text, code, or explanations.\n\nContext: {context}",
        ),
        ("human", "{question}"),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)

    # 2. Helper function to combine retrieved document text
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 3. Build the LCEL Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | chat_model
        | StrOutputParser()
    )

    return rag_chain


if __name__ == "__main__":
    print("Initializing RAG system...")

    # TinyLlama is used here so it runs reasonably fast on CPU.
    # Swap this with "meta-llama/Meta-Llama-3-8B-Instruct" or similar if you have enough VRAM.
    # model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model_id = "google/gemma-3-1b-it"
    sitemap_url = "https://kitswas.github.io/VirtualGamePad/sitemap.xml"

    rag_pipeline = build_rag(model_id, sitemap_url)

    # Example Query
    query = "What is Virtual Gamepad?"
    print(f"\nQuestion: {query}")
    print("⏳ Generating answer...")

    response = rag_pipeline.invoke(query)

    print("\n" + "=" * 50)
    # The pipeline returns the generated text, but we split out the prompt to get just the answer
    answer = response.strip()
    print(f"📝 Answer: {answer}")
    print("=" * 50)

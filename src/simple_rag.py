"""Simple RAG demo - minimal, easy to run."""

from langchain_text_splitters import CharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from embeddings import get_embeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import requests
import torch


def setup_rag_simple(document_url: str):
    """
    Minimal RAG setup with excellent performance on CPU/GPU.
    Uses a tiny efficient model.
    """

    print("Step 1: Fetching document...")
    response = requests.get(document_url)
    text = response.text
    print(f"✓ Got {len(text)} characters")

    print("\nStep 2: Creating embeddings...")
    embedder = get_embeddings()

    print("\nStep 3: Splitting and chunking...")
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_text(text)
    print(f"✓ Split into {len(docs)} chunks")

    print("\nStep 4: Building vector database...")
    # Create Document objects from text chunks
    documents = [Document(page_content=chunk) for chunk in docs]
    db = InMemoryVectorStore.from_documents(documents, embedder)

    print("\nStep 5: Setting up language model...")
    # Using a small, fast model that works well locally
    model_id = (
        "microsoft/phi-2"  # Or "TinyLlama/TinyLlama-1.1B-Chat-v1.0" for even smaller
    )

    print(f"   Loading {model_id}...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map="auto",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map="auto",
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=256,
        temperature=0.5,
        do_sample=True,
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    print("\nStep 6: Creating RAG chain...")
    # Modern LCEL-based RAG chain
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""Answer the question based on the context.
        
Context: {context}

Question: {question}

Answer:""",
    )

    # Retriever
    retriever = db.as_retriever(search_kwargs={"k": 2})

    # Simple RAG chain using LCEL
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": lambda x: x["question"]}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\n✓ RAG system ready!\n")
    return {"chain": chain, "retriever": retriever}


def demo():
    """Run a simple demo."""
    url = "https://raw.githubusercontent.com/kitswas/VirtualGamePad/refs/heads/main/FAQ.md"

    rag_system = setup_rag_simple(url)
    chain = rag_system["chain"]
    retriever = rag_system["retriever"]

    # Interactive query
    while True:
        question = input("\n❓ Ask a question (or 'quit' to exit):\n> ").strip()

        if question.lower() in ["quit", "exit", "q"]:
            break

        if not question:
            continue

        print("\n💭 Thinking...\n")
        answer = chain.invoke({"question": question})

        print(f"💬 {answer}\n")

        # Get source documents separately
        try:
            docs = retriever.invoke(question)
            print("\n📚 Source:")
            for doc in docs[:1]:
                preview = doc.page_content[:100].replace("\n", " ")
                print(f'   "{preview}..."')
        except Exception:
            pass


if __name__ == "__main__":
    demo()

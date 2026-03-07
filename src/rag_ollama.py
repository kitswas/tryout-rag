"""RAG system with Ollama for local LLM inference."""

from langchain_text_splitters import CharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from embeddings import get_embeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import requests


def load_document_from_url(url: str) -> str:
    """Load document content from a URL."""
    response = requests.get(url)
    response.raise_for_status()
    return response.text


def create_rag_with_ollama(
    document_url: str,
    model_name: str = "mistral",
    ollama_base_url: str = "http://localhost:11434",
):
    """
    Create a RAG chain using Ollama for local LLM inference.

    Requirements:
    - Ollama installed and running (download from https://ollama.ai)
    - Model pulled: ollama pull mistral (or your chosen model)

    Args:
        document_url: URL of the knowledge base document
        model_name: Name of the Ollama model to use
        ollama_base_url: Base URL of Ollama server

    Returns:
        RetrievalQA chain
    """

    # Load document
    print(f"📥 Fetching document from {document_url}...")
    document_text = load_document_from_url(document_url)
    print(f"✓ Document loaded ({len(document_text)} characters)")

    # Split into chunks
    print("\n📄 Splitting document into chunks...")
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separator="\n"
    )
    chunks = text_splitter.split_text(document_text)
    print(f"✓ Created {len(chunks)} chunks")

    # Create embeddings
    print("\n🔢 Setting up embeddings...")
    embeddings = get_embeddings()

    # Create vector store
    print("\n🗄️  Creating vector store...")
    documents = [Document(page_content=chunk) for chunk in chunks]
    vector_store = InMemoryVectorStore.from_documents(documents, embeddings)
    print("✓ Vector store ready")

    # Set up Ollama LLM
    print(f"\n🤖 Connecting to Ollama ({model_name})...")
    print(f"   Make sure Ollama is running: ollama serve")
    print(f"   And the model is pulled: ollama pull {model_name}")

    llm = Ollama(
        model=model_name,
        base_url=ollama_base_url,
        temperature=0.5,
    )

    # Create RAG chain using LCEL
    print("\n⚙️  Creating RAG chain...")

    prompt = PromptTemplate.from_template(
        """Answer the question based on the context.
        
Context: {context}

Question: {question}

Answer:"""
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": lambda x: x["question"]}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("✓ RAG system ready!\n")
    return {"chain": chain, "retriever": retriever}


def query(rag_system: dict, question: str) -> None:
    """Query the RAG system and display results."""
    print(f"❓ {question}")
    print("-" * 60)

    chain = rag_system["chain"]
    retriever = rag_system["retriever"]

    # Get answer
    answer = chain.invoke({"question": question})
    print(f"\n💬 Answer:\n{answer}\n")

    # Get source documents
    try:
        docs = retriever.invoke(question)
        if docs:
            print("📎 Sources:")
            for i, doc in enumerate(docs, 1):
                preview = doc.page_content[:80].replace("\n", " ")
                print(f"   {i}. {preview}...")
    except Exception:
        pass


if __name__ == "__main__":
    # Document to use as knowledge base
    faq_url = "https://raw.githubusercontent.com/kitswas/VirtualGamePad/refs/heads/main/FAQ.md"

    # Create the RAG system
    rag = create_rag_with_ollama(faq_url, model_name="mistral")

    # Example queries
    questions = [
        "What games work well with VirtualGamePad?",
        "How do I set up USB connection?",
        "What should I do about laggy/unresponsive controls?",
    ]

    # Ask questions
    for i, q in enumerate(questions, 1):
        print(f"\n{'=' * 60}")
        print(f"Query {i}/{len(questions)}")
        print("=" * 60)
        query(rag, q)

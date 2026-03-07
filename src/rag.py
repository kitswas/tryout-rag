"""Simple RAG system using LangChain with local models."""

from langchain_text_splitters import CharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from embeddings import get_embeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
import requests


def load_document_from_url(url: str) -> str:
    """Load document content from a URL."""
    response = requests.get(url)
    response.raise_for_status()
    return response.text


def create_rag_chain(document_url: str, model_name: str = "huggingFaceM-7B-Instruct"):
    """
    Create a RAG chain with local models.

    Args:
        document_url: URL of the document to use as knowledge base
        model_name: HuggingFace model name for text generation

    Returns:
        RetrievalQA chain ready to use for question answering
    """

    # Load document
    print(f"Fetching document from {document_url}...")
    document_text = load_document_from_url(document_url)

    # Split document into chunks
    print("Splitting document into chunks...")
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separator="\n"
    )
    chunks = text_splitter.split_text(document_text)
    print(f"Created {len(chunks)} chunks")

    # Create embeddings
    print("Setting up embeddings...")
    embeddings = get_embeddings()

    # Create vector store
    print("Creating vector store...")
    documents = [Document(page_content=chunk) for chunk in chunks]
    vector_store = InMemoryVectorStore.from_documents(documents, embeddings)

    # Set up text generation pipeline
    print(f"Loading language model: {model_name}...")
    try:
        # Try to use a local model via transformers
        text_gen_pipeline = pipeline(
            "text-generation",
            model=model_name,
            device_map="auto",
            max_length=512,
            do_sample=True,
            temperature=0.5,
            top_p=0.95,
        )
    except Exception as e:
        print(f"Note: Could not load {model_name}. Using smaller model instead: {e}")
        # Fallback to a smaller model
        text_gen_pipeline = pipeline(
            "text-generation",
            model="distilgpt2",
            device_map="auto",
            max_length=256,
        )

    llm = HuggingFacePipeline(pipeline=text_gen_pipeline)

    # Create RAG chain using LCEL
    print("Creating RAG chain...")

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

    return {"chain": chain, "retriever": retriever}


def query_rag(rag_system: dict, question: str) -> dict:
    """
    Query the RAG system with a question.

    Args:
        rag_system: Dictionary with 'chain' and 'retriever'
        question: The question to ask

    Returns:
        Dictionary with answer and source documents
    """
    chain = rag_system["chain"]
    retriever = rag_system["retriever"]

    # Get answer
    answer = chain.invoke({"question": question})

    # Get source documents
    docs = retriever.invoke(question)

    return {"result": answer, "source_documents": docs}


if __name__ == "__main__":
    # URL to the FAQ document
    faq_url = "https://raw.githubusercontent.com/kitswas/VirtualGamePad/refs/heads/main/FAQ.md"

    # Create RAG chain
    rag_system = create_rag_chain(faq_url)

    # Example questions
    questions = [
        "What games can you play with VirtualGamePad?",
        "How do I connect via USB?",
        "What causes latency issues?",
    ]

    # Ask questions
    for question in questions:
        print(f"\n{'=' * 60}")
        print(f"Question: {question}")
        print("=" * 60)

        result = query_rag(rag_system, question)

        print(f"\nAnswer:\n{result['result']}")
        print(f"\nSource documents:")
        for doc in result.get("source_documents", []):
            print(f"  - {doc.page_content[:100]}...")

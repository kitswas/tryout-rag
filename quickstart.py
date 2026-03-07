#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Quick-start RAG demo - minimal setup, just works."""

import sys
import io


def main():
    """Run the simplest RAG demo possible."""

    # Force UTF-8 output on Windows
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    print("""
╔════════════════════════════════════════════════════════════════╗
║           RAG System - Virtual GamePad FAQ Demo                ║
║                                                                ║
║  This demo creates a RAG system from a GitHub FAQ document     ║
║  and lets you ask questions about it.                          ║
╚════════════════════════════════════════════════════════════════╝
    """)

    # Check dependencies
    print("Checking dependencies...")
    try:
        import torch
        import requests
        from langchain_text_splitters.markdown import MarkdownTextSplitter
        from langchain_core.vectorstores import InMemoryVectorStore
        from langchain_core.documents import Document
        from src.embeddings import get_embeddings
        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from langchain_huggingface import HuggingFacePipeline
        from transformers import pipeline

        print("✓ All dependencies found\n")
    except ImportError as e:
        print(f"\n✗ Missing dependency: {e}")
        print("\nTo install dependencies with uv:")
        print("  uv sync --extra cpu")
        print("\nOr for CUDA GPU:")
        print("  uv sync --extra gpu")
        sys.exit(1)

    print("Setting up RAG system...")
    print("=" * 60)

    # Load document
    print("\n1. Fetching FAQ document...")
    url = "https://raw.githubusercontent.com/kitswas/VirtualGamePad/refs/heads/main/FAQ.md"
    response = requests.get(url)
    text = response.text
    print(f"   ✓ Loaded {len(text)} characters")

    # Split documents
    print("\n2. Splitting into chunks...")
    splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_text(text)
    print(f"   ✓ Created {len(docs)} chunks")

    # Create embeddings
    print("\n3. Setting up embeddings...")
    embedder = get_embeddings()
    print("   ✓ Embeddings ready")

    # Create vector store
    print("\n4. Building vector database...")
    documents = [Document(page_content=chunk) for chunk in docs]
    db = InMemoryVectorStore.from_documents(documents, embedder)
    print("   ✓ Vector store created")

    # Setup LLM
    print("\n5. Loading language model...")
    print("   (This may take a minute on first run)")

    try:
        pipe = pipeline(
            "text-generation",
            model="HuggingFaceTB/SmolLM2-135M-Instruct",
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            max_new_tokens=256,
        )
    except Exception as e:
        print(f"   (Model loading issue: {e})")
        print("   Using fallback distilgpt2...")
        pipe = pipeline(
            "text-generation",
            model="distilgpt2",
            max_new_tokens=128,
        )

    llm = HuggingFacePipeline(pipeline=pipe)

    # Custom output parser to clean up LLM output
    class AnswerExtractor(StrOutputParser):
        def parse(self, text: str) -> str:
            # Extract answer after "Answer:" in the text
            if "Answer:" in text:
                answer = text.split("Answer:")[-1].strip()
            else:
                answer = text.strip()
            # Clean up any leading/trailing whitespace and quotes
            return answer.strip('"').strip()

    print("   ✓ Model loaded")

    # Create RAG chain using LCEL
    print("\n6. Creating RAG chain...")

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""Answer the question based on the context.
        
Context: {context}

Question: {question}

Answer:""",
    )

    retriever = db.as_retriever(search_kwargs={"k": 2})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def get_question(x):
        if isinstance(x, dict):
            return x.get("question", x)
        return x

    chain = (
        {
            "context": (lambda x: get_question(x)) | retriever | format_docs,
            "question": lambda x: get_question(x),
        }
        | prompt
        | llm
        | AnswerExtractor()
    )
    print("   ✓ RAG system ready!")

    print("\n" + "=" * 60)
    print("System ready! Ask a question about VirtualGamePad FAQ.")
    print("Type 'quit' or 'exit' to stop.\n")

    # Interactive loop
    example_questions = [
        "What games work with this?",
        "How do I connect via USB?",
        "How do I fix laggy controls?",
    ]

    print("Example questions:")
    for i, q in enumerate(example_questions, 1):
        print(f"  {i}. {q}")

    while True:
        try:
            question = input("\n❓ Your question: ").strip()

            if not question:
                continue

            if question.lower() in ["quit", "exit", "q"]:
                print("\nThank you for using RAG! Goodbye! 👋")
                break

            print("\n💭 Thinking...")
            answer = chain.invoke({"question": question})

            print(f"\n💬 Answer:\n{answer}\n")

            try:
                docs = retriever.invoke(question)
                if docs:
                    print("📚 Source:")
                    for doc in docs[:1]:
                        preview = doc.page_content[:100].replace("\n", " ").strip()
                        print(f'   "{preview}..."')
            except Exception:
                pass

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again.")


if __name__ == "__main__":
    main()

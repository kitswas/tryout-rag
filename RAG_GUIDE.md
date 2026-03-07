# RAG System Setup Guide

This project contains two RAG (Retrieval-Augmented Generation) implementations using LangChain with local models. Both run completely offline with no external dependencies.

## Quick Start

### Prerequisites

- Python 3.11+
- `uv` package manager (install from <https://github.com/astral-sh/uv>)

### Install Dependencies

```bash
# CPU (recommended to start)
uv sync --extra cpu

# Or for CUDA GPU support
uv sync --extra gpu
```

## Implementations

### 1. **Simple RAG** (`src/simple_rag.py`) ⭐ Recommended for Getting Started

Best for: Quick setup, interactive demos, research

**Features:**

- Uses Microsoft Phi-2 or TinyLlama for text generation
- Local transformers-based embeddings (Windows-friendly, no sentence-transformers)
- Interactive Q&A loop
- Runs on CPU or GPU
- In-memory vector store, no database setup

**Run:**

```bash
uv run python -m src.simple_rag
```

**Usage:**

```
> What games work with VirtualGamePad?
> How do I set up USB connection?
> quit
```

**Model Options:**

- `microsoft/phi-2` - Good balance of quality and speed (7B)
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` - Smaller, faster, CPU-friendly
- `microsoft/phi-1.5` - Very fast but smaller

---

### 2. **HuggingFace Transformers RAG** (`src/rag.py`)

Best for: Custom pipelines, API integration, batch processing

**Features:**

- Direct transformers pipeline integration
- Fallback to distilgpt2 if large model fails
- Full control over generation parameters
- Source document retrieval

**Run:**

```bash
uv run python -m src.rag
```

**Customization:**
Edit these in the script:

```python
model_name = "your-model-here"  # Try: EleutherAI/pythia-1b, mistral-7b, etc.
chunk_size = 1000               # Adjust chunk size
k = 3                           # Number of retrieved documents
```

---

## How RAG Works

```
1. Load Document
   └─> FAQ.md from GitHub
   
2. Split into Chunks
   └─> 1000 char chunks with 200 char overlap
   
3. Create Embeddings
   └─> Local transformers (all-MiniLM-L6-v2, Windows-friendly)
   
4. Build Vector Store
   └─> InMemoryVectorStore (in-memory, no DB needed)
   
5. Query Process:
   └─> User question
       └─> Convert to embeddings
           └─> Similarity search (k=3)
               └─> Retrieve relevant chunks
                   └─> Pass to LLM with prompt
                       └─> Generate answer
```

---

## Comparison

| Feature | simple_rag | rag.py |
|---------|-----------|---------|
| Setup Ease | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Speed | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Interactive | 🟢 | 🔴 |
| Windows Compatible | 🟢 | 🟢 |
| No External Services | 🟢 | 🟢 |

---

## Performance Tips

### For CPU-only systems

```python
# Use smaller models in simple_rag.py
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Reduce chunk size
chunk_size = 500
```

### For GPU systems

```python
# Use larger models for better quality
model_id = "microsoft/phi-2"

# Increase parameters for better quality
max_tokens = 512
temperature = 0.7
```

### For low-memory systems

- Reduce model size: use TinyLlama or Phi-1.5
- Reduce `search_kwargs={"k": 2}` from 3 to 2
- Reduce `chunk_size` from 1000 to 500

---

## Troubleshooting

**"Model not found" error:**

- Models are auto-downloaded from HuggingFace on first run
- May take a few minutes depending on model size
- GPU greatly speeds up download and inference

**Out of memory:**

- Reduce model size: use TinyLlama or Phi-1.5
- Reduce max_length parameter
- Reduce batch size or chunk overlap

**ImportError for torch or transformers:**

- Make sure dependencies are installed: `uv sync --extra cpu` or `uv sync --extra gpu`
- Verify pytorch installation for your OS

**Slow responses:**

- First inference is slower (model loading)
- GPU availability makes huge difference
- Smaller models (phi-1.5, TinyLlama) are much faster
- Windows users: ensure transformers and torch are properly installed

---

## Sample Queries

Try these to test the system:

- "What games work with VirtualGamePad?"
- "How do I connect via USB?"
- "Why is my connection laggy?"
- "How do I change the control mapping?"
- "What causes game input lag?"

---

## Next Steps

1. **Run quickstart.py** for an interactive demo
2. **Try simple_rag.py** for a feature-complete example
3. **Experiment with different models** from HuggingFace
4. **Customize the prompt** for your use case
5. **Add more documents** by modifying `load_document_from_url()`
6. **Build an API** using FastAPI + async queries

---

## Resources

- [LangChain Docs](https://python.langchain.com/)
- [Ollama Models](https://ollama.ai/library)
- [HuggingFace Models](https://huggingface.co/models)
- [Sentence Transformers](https://www.sbert.net/)
- [FAISS Vector Database](https://github.com/facebookresearch/faiss)

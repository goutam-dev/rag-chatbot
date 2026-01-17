# Multi-Document RAG System

A production-ready Retrieval-Augmented Generation (RAG) system for intelligent question-answering over multiple PDF documents. Built with hybrid retrieval, cross-encoder re-ranking, and a Gradio web interface.

## Overview

This system allows users to:
- Ingest multiple PDF documents into a unified knowledge base
- Ask natural language questions and receive cited answers
- Compare information across documents
- Generate comprehensive document summaries

The architecture combines semantic (vector) search with keyword-based (BM25) retrieval, fused using Reciprocal Rank Fusion (RRF), and refined through cross-encoder re-ranking for optimal relevance.

## Architecture

```
User Query
    │
    ├── Query Classification (factoid/summary/comparison/extraction/reasoning)
    ├── Multi-Query Expansion (3 alternative phrasings)
    └── HyDE Generation (hypothetical answer document)
           │
           ▼
    ┌──────────────────────────────────────┐
    │         Hybrid Retrieval             │
    │  ┌─────────────┐  ┌─────────────┐    │
    │  │ ChromaDB    │  │ BM25        │    │
    │  │ (Vector)    │  │ (Keyword)   │    │
    │  └─────────────┘  └─────────────┘    │
    │           │              │           │
    │           └──────┬───────┘           │
    │                  ▼                   │
    │         RRF Fusion + Deduplication   │
    └──────────────────────────────────────┘
                       │
                       ▼
              Cross-Encoder Re-ranking
              (BAAI/bge-reranker-v2-m3)
                       │
                       ▼
              LLM Generation (Llama 3.3 70B)
              with inline source citations
                       │
                       ▼
              Answer Verification (for complex queries)
```

## Tech Stack

| Component         | Technology                        |
|-------------------|-----------------------------------|
| LLM               | Llama 3.3 70B (via Groq API)      |
| Embeddings        | BAAI/bge-large-en-v1.5            |
| Re-ranker         | BAAI/bge-reranker-v2-m3           |
| Vector Database   | ChromaDB                          |
| Keyword Search    | BM25 (rank-bm25)                  |
| PDF Processing    | PyPDF                             |
| Web Interface     | Gradio                            |
| Framework         | LangChain 0.2.x                   |

## Requirements

- Python 3.10 or higher
- Groq API key (free tier available at [console.groq.com](https://console.groq.com))
- GPU recommended but not required (models will run on CPU if unavailable)

*Developed and tested on Google Colab. Dependency versions are pinned to ensure compatibility at the time of development.*

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd rag-project
```

2. Install dependencies:
```bash
pip install numpy==1.26.4
pip install pandas==2.2.2
pip install scipy==1.13.1
pip install langchain-core==0.2.40
pip install langchain-community==0.2.16
pip install langchain==0.2.16
pip install langchain-groq==0.1.9
pip install langchain-text-splitters==0.2.4
pip install chromadb==0.5.5
pip install sentence-transformers==3.0.1
pip install pypdf==4.3.1
pip install rank-bm25==0.2.2
pip install gradio
pip install torch
```

3. Restart your Python environment after installation.

## Usage

### Running the Application

Open `rag.ipynb` in Jupyter Notebook or VS Code and execute the cells sequentially. The final cell launches a Gradio web interface accessible at `http://localhost:7860`.

Alternatively, the interface generates a public shareable link via `gradio.live`.

### Workflow

1. **Initialize**: Enter your Groq API key in the Setup tab and click "Initialize"
2. **Upload Documents**: Add one or more PDF files to build the knowledge base
3. **Ask Questions**: Use the Chat tab to query your documents
4. **Summarize**: Generate a comprehensive summary of all loaded documents

### Query Options

- **HyDE (Hypothetical Document Embeddings)**: Generates a hypothetical answer to improve retrieval quality. Enabled by default.
- **Multi-Query**: Expands the original query into multiple phrasings for broader coverage. Enabled by default.

## Key Features

### Intelligent Query Classification

The system automatically classifies queries into five types and adjusts retrieval strategy accordingly:

| Query Type   | Retrieval Depth (k) | Answer Style |
|--------------|---------------------|--------------|
| Factoid      | 6                   | Direct       |
| Summary      | 10                  | Bullets      |
| Comparison   | 12                  | Bullets      |
| Extraction   | 8                   | Direct       |
| Reasoning    | 10                  | Steps        |

### Semantic Chunking

Documents are split based on semantic similarity between sentences rather than fixed character counts, preserving coherent ideas within each chunk.

### Multi-Document Support

- Upload multiple PDFs to build a combined knowledge base
- Automatic PDF diversity enforcement for cross-document queries
- Clear source attribution with document name and page number

### Answer Verification

For complex queries (comparisons, summaries, reasoning), the system performs a self-verification step to ensure answers are direct, structured, and grounded in sources.

### Caching

Repeated queries return cached responses for faster performance.

## Example Questions

**Single Document Analysis**:
- "What is the main contribution of this paper?"
- "Explain the methodology in detail"
- "What are the limitations mentioned by the authors?"

**Multi-Document Comparison** (with two or more PDFs loaded):
- "Compare the approaches discussed in these papers"
- "What are the key differences between the methodologies?"

**Summarization**:
- Use the Summarize tab to generate a map-reduce summary of all documents

## Project Structure

```
rag project/
└── rag.ipynb          # Main notebook containing all code
    ├── Dependencies   # Installation cell
    ├── Imports        # Library imports and device setup
    ├── Data Classes   # QueryProfile, QueryCache, SemanticChunker, RRF
    ├── EnhancedRAGv3  # Core RAG engine class
    ├── Gradio UI      # Web interface definition
    └── Launch         # Application startup
```

## Performance

| Operation              | Typical Duration      |
|------------------------|-----------------------|
| Model initialization   | 30-60 seconds         |
| PDF ingestion (per doc)| 10-30 seconds         |
| Simple queries         | 5-8 seconds           |
| Complex queries        | 10-15 seconds         |
| Full document summary  | 30-90 seconds         |

## Configuration Parameters

Key parameters that can be adjusted in the code:

| Parameter              | Default | Description                                    |
|------------------------|---------|------------------------------------------------|
| `max_chunk_size`       | 1000    | Maximum characters per semantic chunk          |
| `similarity_threshold` | 0.5     | Cosine similarity threshold for chunk grouping |
| `chunk_size`           | 800     | Fallback text splitter chunk size              |
| `chunk_overlap`        | 150     | Character overlap between chunks               |
| `fetch_factor`         | 2       | Multiplier for initial retrieval pool          |
| `lambda_mult`          | 0.6     | MMR diversity parameter (0=diverse, 1=relevant)|
| `cache_max_size`       | 100     | Maximum number of cached query responses       |

## Limitations

- Requires active internet connection for Groq API calls
- PDF quality affects text extraction accuracy
- Large documents may take longer to process
- Query cache does not persist between sessions
- Package versions are pinned to those available during development; future updates to underlying libraries may require adjustments

## License

This project is provided for educational and research purposes.

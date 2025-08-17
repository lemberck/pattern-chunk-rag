# Pareto RAG Pipeline

A 7-step RAG chunking pipeline designed to identify and represent **abstract multi-document patterns**. Unlike traditional document-level chunking, this system discovers canonical topics that span multiple documents and creates semantically coherent chunks based on these patterns.

## 🎯 Project Overview

The Pareto RAG Pipeline solves a key challenge in RAG systems: **how to identify and represent knowledge that spans multiple documents**. Traditional chunking treats each document independently, missing important patterns that emerge across your entire knowledge base.

### Key Innovation
- **Multi-Document Pattern Detection**: Identifies topics that appear across multiple documents (e.g., "verification abandonment" appearing in 6+ files)
- **Canonical Topic Promotion**: Elevates frequently-occurring patterns to first-class entities
- **Pattern-Aware Chunking**: Creates chunks that respect semantic boundaries rather than arbitrary token limits

```
📄 Documents → 🏷️ Topics → 🔗 Groups → ⭐ Canonical → 📦 Chunks → 🕸️ Knowledge Graph → 🧠 LightRAG
```

## 🏗️ Architecture

### Pipeline Flow Diagram

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Step 0    │    │   Step 1    │    │   Step 2    │    │   Step 3    │
│   Change    │───▶│  Document   │───▶│   Topic     │───▶│ Similarity  │
│ Detection   │    │ Extraction  │    │ Labeling    │    │  Grouping   │
│             │    │             │    │ (Agent 1)   │    │ (Embeddings)│
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                             │                    │
                                      ┌───── ▼─────┐        ┌─────▼─────┐
                                      │ Async LLM │         │ Cosine    │
                                      │ Batching  │         │Similarity │
                                      └───────────┘         └───────────┘

┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Step 4    │    │   Step 5    │    │   Step 6    │    │   Step 7    │
│ Canonical   │───▶│   Smart     │───▶│ Knowledge   │───▶│  LightRAG   │
│ Promotion   │    │  Chunking   │    │    Graph    │    │Integration  │
│ (Agent 2)   │    │             │    │Construction │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
        │                  │                  │                  │
┌───────▼───────┐  ┌───────▼───────┐  ┌───────▼───────┐  ┌───────▼───────┐
│ Async LLM     │  │Pattern-Aware  │  │ Nodes: Pattern│  │ Custom KG     │
│ Canonicalizat.│  │Token Limits   │  │Chunk, Document│  │ Insertion     │
└───────────────┘  └───────────────┘  └───────────────┘  └───────────────┘
```

## 📁 Scripts Documentation

### `src/` Core Modules

#### **Configuration & Infrastructure**
- **`config.py`** - Pydantic-based configuration management with environment variable integration and validation
- **`utils.py`** - Shared utilities for hashing, JSON I/O, tokenization, and ID generation
- **`prompts/prompts.py`** - Centralized LLM prompts for safe iteration without breaking business logic

#### **Pipeline Steps**
- **`change_tracker.py`** - SHA1-based incremental processing with surgical deletion handling and caching
- **`extractor.py`** - Document extraction with tiktoken tokenization and sentence splitting
- **`labeler_agent.py`** - Async LLM topic extraction (Agent 1) with batched processing for rate limit management
- **`grouper.py`** - Embedding-based similarity grouping using OpenAI text-embedding-3-small and cosine similarity
- **`promoter_agent.py`** - Canonical topic promotion (Agent 2) with async LLM canonicalization for multi-document patterns
- **`chunker.py`** - Smart chunking that respects both token limits and canonical pattern boundaries
- **`kg_builder.py`** - Knowledge graph construction with pattern, chunk, and document nodes plus typed relationships
- **`lightrag_client.py`** - Custom KG insertion to LightRAG bypassing standard document processing pipeline

### `tests/` Test Suite

- **`test_incremental_processing.py`** - Comprehensive change detection tests covering new files, modifications, deletions, and cache management
- **`test_full_integration.py`** - Q&A demo with 8 sample queries testing canonical pattern retrieval and LightRAG integration

### **Main Entry Point**

- **`main.py`** - Pipeline orchestrator with command-line interface, surgical deletion handling, and incremental processing logic

### **Test Data**

- **`data/`** - 24 LLM-generated test documents covering verification patterns, user activation, SMS delivery, coffee roasting, team activities, and office management (designed to test multi-document pattern detection)

## ✨ Features

### **Pipeline Outputs** (in `output/` directory)

1. **`01_sentences.jsonl`** - Extracted sentences with metadata, token counts, and document provenance
2. **`02_sentence_topics.jsonl`** - Topic labels and descriptions for each sentence from Agent 1
3. **`03_candidate_groups.json`** - Similar topic groupings with support document lists and similarity scores
4. **`04_pattern_bank.json`** - Canonical topics promoted to represent multi-document patterns
5. **`04_sentence_to_canonical.jsonl`** - Mapping from sentences to their canonical topics
6. **`05_chunks.json`** - Smart chunks with pattern-aware boundaries and rich metadata
7. **`06_kg_nodes.json`** - Knowledge graph nodes (patterns, chunks, documents)
8. **`06_kg_edges.json`** - Knowledge graph relationships (supports, part_of)
9. **`07_upsert_manifest.json`** - LightRAG integration status and configuration

### **Processing Features**

- **🔍 Incremental Processing**: SHA1-based change detection processes only new/modified files
- **🗑️ Surgical Deletion**: Intelligently removes deleted files from outputs without full rebuild when possible
- **⚡ Async Optimization**: Concurrent LLM calls with batching strategies optimized per step
- **📊 Progress Logging**: Detailed pipeline.log with processing statistics and timing
- **🔄 Cache Management**: Intermediate result caching in `.pipeline_cache/` for efficiency
- **🎛️ Configuration**: All parameters configurable via environment variables

## ⚙️ Configuration

### Environment Variables
**The only required environment variable is `OPENAI_API_KEY`.**

All other variables have default values in config.py and can be deleted from the `.env` file.

Create `.env` file (copy from `.env.example`):

```bash
### Required
OPENAI_API_KEY=your_openai_api_key_here

### Optional
# Data & Processing (defaults)
DATA_DIR=./data
OUTPUT_DIR=./output
TOKENIZER=tiktoken:cl100k_base

# OpenAI Models (defaults)
EMBEDDING_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4o-mini
LLM_MODEL=gpt-4o-mini

# Similarity & Canonicalization (defaults)
LABEL_SIM_THRESHOLD=0.60
CANONICAL_SUPPORT_FRACTION=0.10
CANONICAL_MIN_DOCS=2

# Chunking Parameters (defaults)
CHUNK_MIN_TOKENS=180
CHUNK_TARGET_TOKENS=400
CHUNK_MAX_TOKENS=500
CHUNK_OVERLAP_SENTS=1
INCLUDE_NON_PATTERN_CHUNKS=false

# LightRAG Integration (defaults)
LIGHTRAG_WORKING_DIR=./lightrag_storage
```

### Key Parameter Explanations

- **`LABEL_SIM_THRESHOLD`** - Cosine similarity threshold for grouping topics (0.68 = semantic similarity)
- **`CANONICAL_SUPPORT_FRACTION`** - Minimum fraction of documents for canonical promotion (0.10 = 10%)
- **`CANONICAL_MIN_DOCS`** - Absolute minimum documents needed for canonical status
- **`CHUNK_TARGET_TOKENS`** - Preferred chunk size for optimal LLM processing
- **`INCLUDE_NON_PATTERN_CHUNKS`** - Whether to include document chunks for non-pattern content

## 🚀 Installation & Usage

### Prerequisites

- Python 3.11+
- [UV package manager](https://github.com/astral-sh/uv) 
- OpenAI API key

### Installation

```bash
# Clone repository
git clone https://github.com/lemberck/pattern-chunk-rag.git
cd pattern-chunk-rag

# Install dependencies with UV
uv sync

# Set up environment
cp .env.example .env
# Edit .env with your OpenAI API key
```

### Running the Pipeline

```bash
# Normal run - processes only changed files or all files for the first run
uv run python main.py

# Force complete rebuild
uv run python main.py --reset

# Verbose output with processing details
uv run python main.py --verbose
```

### Testing with Sample Queries

```bash
# Run integration test with sample Q&A
uv run python tests/test_full_integration.py

# Run incremental processing tests
uv run pytest tests/test_incremental_processing.py -v
```

The integration test demonstrates the pipeline's ability to answer questions like:
- "What are the main patterns causing user abandonment during verification?"
- "How does verification timing impact user activation and retention?"
- "What solutions exist for reducing verification abandonment on mobile?"

## 📂 Project Structure

```
pareto/
├── 📄 main.py                    # Pipeline orchestrator and CLI
├── 📄 README.md                  # This file
├── 📄 REASONING.md               # Detailed design decisions
├── 📄 pipeline.log               # Processing logs and debug info
├── 📄 pyproject.toml             # UV dependency configuration
├── 📄 uv.lock                    # Dependency lock file
├── 📄 .env.example               # Environment variable template
├── 📄 .gitignore                 # Git ignore patterns
│
├── 📁 src/                       # Core pipeline modules
│   ├── 📄 __init__.py
│   ├── 📄 config.py              # Configuration management
│   ├── 📄 change_tracker.py      # Step 0: Incremental processing
│   ├── 📄 extractor.py           # Step 1: Document extraction
│   ├── 📄 labeler_agent.py       # Step 2: Topic labeling (Agent 1)
│   ├── 📄 grouper.py             # Step 3: Similarity grouping
│   ├── 📄 promoter_agent.py      # Step 4: Canonical promotion (Agent 2)
│   ├── 📄 chunker.py             # Step 5: Smart chunking
│   ├── 📄 kg_builder.py          # Step 6: Knowledge graph
│   ├── 📄 lightrag_client.py     # Step 7: LightRAG integration
│   ├── 📄 utils.py               # Shared utilities
│   └── 📁 prompts/               # LLM prompts
│       ├── 📄 __init__.py
│       └── 📄 prompts.py         # Centralized prompt definitions
│
├── 📁 data/                      # Input documents (24 LLM-generated test files)
│   ├── 📄 ab_testing_verification.txt
│   ├── 📄 accessibility_verification.txt
│   ├── 📄 coffee_roasting_basics.txt
│   ├── 📄 mobile_verification_study.txt
│   └── ... (20 more diverse test documents)
│
├── 📁 output/                    # Pipeline artifacts (generated)
│   ├── 📄 01_sentences.jsonl     # Extracted sentences
│   ├── 📄 02_sentence_topics.jsonl # Topic labels
│   ├── 📄 03_candidate_groups.json # Topic groupings
│   ├── 📄 04_pattern_bank.json   # Canonical patterns
│   ├── 📄 04_sentence_to_canonical.jsonl # Sentence mappings
│   ├── 📄 05_chunks.json         # Smart chunks
│   ├── 📄 06_kg_nodes.json       # Knowledge graph nodes
│   ├── 📄 06_kg_edges.json       # Knowledge graph edges
│   └── 📄 07_upsert_manifest.json # LightRAG status
│
├── 📁 .pipeline_cache/           # Incremental processing cache
│   ├── 📄 file_manifest.json     # File hashes and timestamps
│   └── 📁 intermediate/          # Cached step results
│
├── 📁 lightrag_storage/          # LightRAG vector database
│   └── ... (LightRAG internal files)
│
└── 📁 tests/                     # Test suite
    ├── 📄 __init__.py
    ├── 📄 test_incremental_processing.py # Change detection tests
    └── 📄 test_full_integration.py       # Q&A demo
```

## 🔍 Example Results

### Canonical Patterns Discovered

From the test dataset, the pipeline typically identifies patterns like:

- **`verification_timing_impact`** - How verification timing affects user activation (6 documents)
- **`deferred_verification_benefits`** - Advantages of deferred verification strategies (4 documents)  
- **`sms_delivery_optimization`** - SMS delivery reliability and optimization (5 documents)
- **`mobile_verification_friction`** - Mobile-specific verification challenges (7 documents)

### Smart Chunks Created

Chunks respect canonical pattern boundaries and include rich metadata:

```json
{
  "id": "canonical_chunk_verification_timing_impact",
  "text": "Combined content from all sentences about verification timing...",
  "metadata": {
    "canonical_pattern": {
      "id": "canonical_topic_12abc",
      "label": "verification_timing_impact",
      "desc": "Impact of verification timing on user activation"
    },
    "document_sources": [
      {"doc_id": "abc123", "doc_name": "mobile_verification_study.txt"},
      {"doc_id": "def456", "doc_name": "user_onboarding_research.txt"}
    ],
    "sentence_count": 23,
    "token_count": 387,
    "chunk_type": "canonical_pattern_chunk"
  }
}
```

## 🛠️ Development

### Performance Tips

- **Incremental Processing**: Only processes changed files, reducing API costs and processing time
- **Async Optimization**: Concurrent LLM calls improve throughput
- **Batch Tuning**: Adjust batch sizes based on API rate limits
- **Cache Management**: Intermediate caching avoids reprocessing unchanged data

### Adding Custom Processing

1. **Modify Pipeline Step**: Edit relevant module in `src/`
2. **Update Configuration**: Add parameters to `src/config.py`
3. **Test Changes**: Run `uv run python main.py --verbose`
4. **Add Tests**: Create tests in `tests/` directory

## 📋 Future Enhancements

- **LLM-as-a-Judge Quality Assurance**: Hallucination detection and relevance scoring
- **Pattern Evolution Tracking**: Monitor how canonical patterns change over time
- **Advanced Chunking Strategies**: Overlapping chunks with different strategies
- **Interactive Web Interface**: GUI for parameter tuning and result visualization
- **Multi-threading**: Parallel processing for independent operations

For detailed design decisions and architectural reasoning, see [REASONING.md](./REASONING.md).
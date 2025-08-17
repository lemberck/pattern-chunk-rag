# REASONING.md - Design Decisions & Justifications

## Project Overview

This project implements a 7-step RAG chunking pipeline designed to identify and represent abstract multi-document patterns. The pipeline extracts documents, labels topics, groups similar patterns, promotes canonical topics, creates smart chunks, builds a knowledge graph, and integrates with LightRAG.

## Why Direct Agents Instead of Multi-Agent Frameworks

**Decision**: Implement direct agent classes (TopicLabeler, TopicPromoter) rather than using multi-agent frameworks like LangChain or CrewAI.

**Justification**:
- **Algorithmic Flexibility**: The pipeline naturally mixes LLM calls with traditional algorithms (Step 2: LLM → Step 3: cosine similarity → Step 4: LLM). Multi-agent frameworks are optimized for LLM-to-LLM interactions, not LLM-to-algorithm workflows
- **Reduced Complexity**: Direct implementation avoids framework abstractions, tool definitions, and orchestration overhead that would add unnecessary complexity
- **Performance Control**: We can optimize async processing patterns specifically for the use case (batched processing in Step 2, unbatched in Step 4) without framework constraints
- **Debugging Simplicity**: Direct agent classes make it easier to trace execution, log intermediate results, and debug issues compared to framework-wrapped agents
- **Dependency Minimization**: Fewer dependencies mean faster startup, smaller deployment size, and reduced version conflict risks
- **Custom Integration**: LightRAG custom KG insertion bypasses standard document processing - frameworks would add unnecessary abstraction layers

**The Trade-off**: We lose framework conveniences like built-in memory management and agent communication patterns, but gain precise control over the hybrid LLM+algorithm pipeline.

**Alternative Considered**: LangChain/langgraph with custom tools for similarity calculations
**Why Rejected**: The tool abstraction would complicate the natural flow of the algorithm-heavy pipeline without providing meaningful benefits

## Architecture Decisions

### 1. 7-Step Pipeline Design

**Decision**: Implement a linear 7-step pipeline with clear separation of concerns.

**Justification**: 
- Each step has a single responsibility and clear inputs/outputs
- Enables debugging and incremental development
- Allows for easy modification of individual steps
- Facilitates testing and validation at each stage

**Alternative Considered**: Monolithic processing with all steps in one module
**Why Rejected**: Would be harder to debug, test, and maintain

### 2. Sentence-Level vs Paragraph-Level Extraction

**Decision**: Use sentence-level topic extraction rather than paragraph-level processing.

**Justification**:
- **Cleaner Topic Boundaries**: Sentences are the smallest reliable unit for topic switching
- **Less Topic Mixing**: Paragraphs often contain multiple ideas that would blur pattern boundaries
- **Precise Pattern Detection**: Enables more accurate identification of where topics start and end
- **Aggregation Flexibility**: Can always aggregate sentences into larger windows later if needed
- **Better Multi-Document Alignment**: Sentences provide finer granularity for matching similar concepts across documents

**Why This Choice is Critical**:
- Multi-document pattern detection requires precise topic boundaries
- Mixing topics within processing units would make cross-document pattern matching less accurate
- Sentence-level precision enables better canonical pattern identification

**Alternative Considered**: Paragraph-level topic extraction
**Why Rejected**: Paragraphs often contain multiple ideas, making it harder to identify clean pattern boundaries across documents

### 3. Agent-Based Topic Extraction

**Decision**: Use two separate LLM agents - Agent 1 for initial topic extraction, Agent 2 for canonical naming.

**Justification**:
- Agent 1 focuses on extracting raw topics without overthinking
- Agent 2 synthesizes multiple similar topics into canonical form
- Separation of concerns improves consistency and quality
- Allows for different prompting strategies per task

**Alternative Considered**: Single agent handling both extraction and canonicalization
**Why Rejected**: Would lead to inconsistent outputs and mixing of concerns

### 4. Embedding-Based Similarity Grouping

**Decision**: Use OpenAI's text-embedding-3-small for grouping similar topic labels.

**Justification**:
- Semantic similarity captures related concepts better than string matching
- OpenAI embeddings are reliable and well-tested
- Cosine similarity threshold (0.6) allows semantically related topics to group while maintaining distinction
- Cost-effective compared to larger embedding models

**Alternative Considered**: String-based fuzzy matching
**Why Rejected**: Would miss semantic relationships between different phrasings

### 4. Deterministic Output with Temperature=0

**Decision**: Set temperature=0 in both TopicLabeler and TopicPromoter agents for consistent, reproducible results.

**Justification**:
- **Deterministic Grouping**: Eliminates randomness in similarity-based grouping decisions
- **Debugging**: Makes it easier to track changes and validate pipeline behavior
- **Canonical Stability**: Ensures canonical patterns remain consistent between runs

**Implementation**:
- TopicLabeler (Step 2): Uses temperature=0 for consistent topic extraction
- TopicPromoter (Step 4): Uses temperature=0 for deterministic canonical descriptions
- Combined with deterministic SHA1 IDs for complete reproducibility

**Alternative Considered**: Higher temperature for creative outputs
**Why Rejected**: Consistency and reproducibility are more important than creative variation for this use case

### 5. Configurable Parameters

**Decision**: Make all thresholds and limits configurable via environment variables.

**Justification**:
- Enables tuning for different datasets
- Supports experimentation and optimization
- Allows deployment-specific configurations
- Facilitates A/B testing of parameters

**Alternative Considered**: Hard-coded parameters
**Why Rejected**: Would limit flexibility and reusability

### 6. Deterministic ID Generation

**Decision**: Use SHA1 hashes for generating stable IDs across runs.

**Justification**:
- Enables idempotent pipeline runs
- Same content always generates same ID
- Supports incremental updates
- Facilitates debugging and tracking

**Alternative Considered**: Random UUIDs
**Why Rejected**: Would break idempotency and make incremental updates impossible

### 7. Canonical Topic Promotion Criteria

**Decision**: Promote topics to canonical status when they appear in ≥10% of documents with minimum 2 documents.

**Justification**:
- 10% threshold ensures patterns are truly multi-document
- With 23 documents, this creates a threshold of 5 documents for canonical promotion
- Minimum 2 documents prevents single-doc topics from becoming canonical (edge case)
- Dynamic threshold scales with dataset size
- Current configuration produces 4 canonical patterns from verification-related content
- Based on empirical observation of pattern distribution

**Alternative Considered**: Fixed document count threshold
**Why Rejected**: Wouldn't scale well with different dataset sizes

### 8. Prompt Isolation Strategy

**Decision**: Extract all LLM prompts into a dedicated `src/prompts/` module with prompts stored as variables in `prompts.py`.

**Justification**:
- **Separation of Concerns**: Prompts are isolated from business logic, making the code cleaner and more maintainable
- **Easy Modification**: Prompt engineers can modify prompts without touching core pipeline code
- **Version Control**: Prompt changes can be tracked independently from code changes
- **Single Source of Truth**: All prompts are centralized in one file for easy discovery
- **Safe Iterations**: Changing prompts doesn't risk breaking pipeline logic or imports

**File Renaming**: Also renamed agent files for clarity:
- `src/labeler.py` → `src/labeler_agent.py` 
- `src/promoter.py` → `src/promoter_agent.py`

This makes it immediately clear which files contain LLM agents vs. other pipeline components.

**Alternative Considered**: Keep prompts inline with agent code
**Why Rejected**: Makes prompt iteration risky and harder to track changes independently

### 9. Knowledge Graph Choice for Multi-Document Patterns

**Decision**: Use a knowledge graph approach to identify and represent abstract multi-document patterns rather than traditional document-level chunking.

**Justification**:
**This IS the core requirement** - The project specifically asks for "identifying & representing abstract multi-document patterns":

- **Multi-Document Pattern Detection**: THis approach identifies topics that appear across multiple documents (e.g., "verification abandonment" appearing in 6+ different files)
- **Pattern Representation**: We create canonical topics with descriptions and support document lists
- **Semantic Relationships**: Knowledge graphs excel at representing relationships that span multiple documents
- **Custom KG Insertion**: We use `insert_custom_kg` specifically to bypass LightRAG's processing and insert the pre-identified multi-document patterns

**Core Value Delivered**:
1. **Identify patterns**: Find topics that appear across multiple documents (semantic analysis)
2. **Represent patterns**: Create canonical topics with descriptions and provenance  
3. **Chunk based on patterns**: Create semantically coherent chunks that respect the identified pattern boundaries
4. **Preserve relationships**: Maintain explicit connections between patterns, chunks, and source documents

**Why This Approach was chosen**:
- The requirement asks for "abstract multi-document patterns"
- Pattern-aware chunking is superior to naive document chunking for RAG
- Custom KG insertion preserves the carefully crafted pattern structure in LightRAG

**Alternative Considered**: Traditional document chunking with LightRAG's built-in entity extraction
**Why Rejected**: Would miss the core requirement of identifying patterns that span multiple documents

### 10. Token-Aware Chunking Strategy

**Decision**: Implement chunking that respects both token limits and canonical topic boundaries.

**Justification**:
- Maintains semantic coherence within chunks
- Respects LLM context window limitations
- Preserves canonical patterns as single units when possible
- Configurable parameters allow tuning for different use cases

**Alternative Considered**: Fixed-size chunking ignoring topic boundaries
**Why Rejected**: Would break semantic coherence and pattern integrity

### 11. Knowledge Graph Structure

**Decision**: Three-node-type graph: Patterns, Chunks, Documents with "supports" and "part_of" relationships.

**Justification**:
- Simple structure is easy to understand and query
- Captures essential relationships for RAG
- Compatible with LightRAG's custom KG format
- Extensible for future relationship types

**Alternative Considered**: More complex graph with sentence-level nodes
**Why Rejected**: Would be overly complex for the use case - keep it simple, ship fast

## Implementation Decisions

### 12. Python with UV Package Management

**Decision**: Use Python 3.11+ with UV for dependency management.

**Justification**:
- UV provides faster, more reliable dependency resolution
- Python ecosystem has excellent AI/ML libraries
- Modern Python features improve code quality
- UV's lock files ensure reproducible builds

**Alternative Considered**: Poetry or pip
**Why Rejected**: UV is faster and more modern

### 13. Pydantic for Configuration

**Decision**: Use Pydantic for configuration management with environment variables.

**Justification**:
- Type safety and validation
- Environment variable integration
- Clear configuration schema
- Good error messages for misconfigurations

**Alternative Considered**: Plain Python dictionaries
**Why Rejected**: Lacks validation and type safety

### 14. JSON Lines for Intermediate Artifacts

**Decision**: Store intermediate results in JSONL format.

**Justification**:
- Streaming-friendly for large datasets
- Easy to inspect and debug
- Language-agnostic format
- Efficient for line-by-line processing

**Alternative Considered**: CSV
**Why Rejected**: JSONL preserves complex nested data better

### 15. Modular Architecture

**Decision**: Separate modules for each pipeline step with clear interfaces.

**Justification**:
- Enables independent testing of components
- Facilitates code reuse and modification
- Clear separation of concerns
- Easy to understand and maintain

**Alternative Considered**: Single large module
**Why Rejected**: Would be hard to maintain and test

## LightRAG Integration Decisions

### 16. Custom KG Insertion Approach

**Decision**: Use LightRAG's `insert_custom_kg` method instead of document insertion.

**Justification**:
- Maintains control over chunk boundaries
- Preserves canonical pattern relationships
- Avoids duplicate extraction by LightRAG
- Enables deterministic chunk IDs

**Alternative Considered**: Standard document insertion with LightRAG extraction
**Why Rejected**: Would lose the carefully crafted chunks and patterns (the case goal)

### 16a. Understanding LightRAG Query References

**LightRAG Query Results**: When users query the knowledge graph, they will see different types of references indicating data sources.

**Reference Types Explained**:

1. **[Knowledge Graph] (entities from pattern analysis) (pattern_label)**: 
   - Source: Canonical pattern entities we created
   - Example: `(verification_timing_impact)` or `(deferred_verification_impact)`
   - Meaning: Answer drew from a specific canonical pattern by its semantic label

2. **[Knowledge Graph] (entities from pattern analysis)**:
   - Source: Multiple canonical patterns aggregated  
   - Meaning: Answer synthesized information from several canonical patterns
   - Used when the answer draws from multiple patterns without citing specific ones

3. **[Document Chunk] data/filename.txt**:
   - Source: Raw document content accessed by LightRAG
   - Example: `data/product_verification.txt`, `data/ab_testing_verification.txt`
   - Meaning: LightRAG accessed original document files directly, not the processed chunks
   - Note: With `include_non_pattern_chunks=False`, there are no chunks for non-pattern content

**Reference Format Note**: These reference formats are generated by LightRAG's internal prompting system and can be customized by modifying the framework's response generation templates. The current format prioritizes readability over technical precision.

**Why Mixed References**: Most answers draw from canonical patterns (Knowledge Graph references), but some also include document chunks when queries require specific details not captured in the pattern abstractions.

## Error Handling & Resilience

### 17. Graceful Degradation

**Decision**: Continue processing even if individual sentences fail labeling.

**Justification**:
- Prevents single failures from breaking entire pipeline
- Provides fallback labels for error cases
- Logs errors for debugging
- Maximizes data processing throughput

**Alternative Considered**: Fail-fast on any error
**Why Rejected**: Would be too brittle for production use

### 18. Incremental Processing Implementation

**Decision**: Implement comprehensive change detection system with file hashing and caching to avoid reprocessing unchanged files.

**Justification**:
- **Efficiency**: Only processes changed files, dramatically reducing API token usage
- **Performance**: Skips unchanged documents, reducing runtime from minutes to seconds
- **Reliability**: SHA1 file hashing ensures accurate change detection
- **Cache Management**: Preserves intermediate results for unchanged files
- **Reset Capability**: `--reset` flag forces complete reprocessing when needed

**Implementation Details**:
- **Step 0**: Change detection using `ChangeTracker` class
- **File Manifest**: Stores SHA1 hashes and modification timestamps in `.pipeline_cache/file_manifest.json`
- **Intermediate Caching**: Saves pipeline step results in `.pipeline_cache/intermediate/`
- **Incremental Extractor**: Modified to process only specified files
- **Reset Functionality**: Clears cache, output, and LightRAG storage for fresh start

**Alternative Considered**: Always reprocess everything
**Why Rejected**: Would be wasteful of API tokens and processing time for production use

### 18a. Change Detection Algorithm

**Decision**: Use SHA1 file content hashing combined with modification time tracking.

**Justification**:
- **Content-based**: Detects actual content changes, not just timestamp updates
- **Consistent**: Uses same SHA1 algorithm as existing codebase ID generation
- **Fast**: Hashing is faster than full content comparison
- **Deterministic**: Same content always produces same hash
- **Sufficient**: SHA1 provides adequate collision resistance for change detection

**Process**:
1. Calculate SHA1 hash for each `.txt` file in `data/` directory
2. Compare against cached hashes in file manifest
3. Identify new files (not in manifest)
4. Identify modified files (different hash)
5. Identify deleted files (in manifest but not on disk)
6. Process only new and modified files

### 18b. Command Line Interface for Incremental Processing

**Decision**: Add `--reset` and `--verbose` flags for pipeline control.

**Usage Patterns**:
```bash
# Normal run - only process changes
python main.py

# Reset everything - full reprocess
python main.py --reset

# Verbose mode - show processing details
python main.py --verbose
```

**Reset Behavior**:
- Deletes `.pipeline_cache/` completely
- Deletes `lightrag_storage/` completely  
- Deletes `output/` completely
- Forces processing of ALL files in `data/`

### 18c. File Deletion Handling Strategy

**Decision**: Implement surgical removal of deleted files from pipeline outputs when possible, with fallback to full rebuild.

**Justification**:
- **Efficiency**: Saves significant API tokens by avoiding reprocessing unchanged files
- **Performance**: Reduces execution time from minutes to seconds for deletions when surgical removal succeeds
- **Data Integrity**: Falls back to full rebuild if existing pipeline outputs are missing or corrupted
- **Adaptive**: Chooses optimal strategy based on current pipeline state

**Surgical Removal Process** (when existing outputs are available):
1. Identify deleted files by comparing manifest to disk
2. Remove all entries matching deleted filenames from pipeline outputs (Steps 1-5)
3. Recalculate canonical pattern thresholds after removal
4. Rebuild only Steps 6-7 (Knowledge Graph and LightRAG integration)
5. Update manifest and clean storage

**Full Rebuild Fallback** (when outputs are missing):
1. Log: "No existing pipeline outputs found. Falling back to full rebuild..."
2. Clear `output/`, `lightrag_storage/`, and cache directories
3. Process ALL remaining files from scratch
4. Rebuild complete knowledge base with clean data

**Alternative Considered**: Always use full rebuild for deletions
**Why Rejected**: Too expensive for API tokens and processing time when surgical removal is sufficient

## Performance Considerations

### 19. Async Processing Architecture for LLM Agents

**Decision**: Implement different async strategies for Steps 2 and 4 based on their processing characteristics and API constraints.

#### Step 2 (TopicLabeler): Batched Async Processing

**Decision**: Use batched async processing with `batch_size=10` for sentence labeling.

**Justification**:
- **High Volume**: Processing 215+ sentences requires rate limit management
- **API Constraints**: OpenAI rate limits require controlled concurrency
- **Error Isolation**: Batch failures don't break entire pipeline
- **Progress Tracking**: Batch-level logging provides meaningful progress updates
- **Memory Management**: Processes 10 sentences concurrently, then releases memory

#### Step 4 (TopicPromoter): Unbatched Async Processing

**Decision**: Process all qualified groups concurrently without batching.

**Justification**:
- **Low Volume**: Typically 20-50 groups to promote (much fewer than sentences)
- **No Rate Limit Pressure**: Volume is well within API limits
- **Faster Completion**: All groups processed simultaneously for maximum speed
- **Simpler Logic**: No batching complexity needed for small datasets

#### Why Different Strategies

**Volume-Based Decision Making**:
- **High Volume (Step 2)**: Batch to respect API limits and manage memory
- **Low Volume (Step 4)**: Process all at once for maximum throughput

**Error Handling Benefits**:
- Both approaches use `return_exceptions=True` for graceful degradation
- Batch approach provides more granular error reporting
- Individual failures don't break the entire processing step

**Alternative Considered**: Same strategy for both steps
**Why Rejected**: One-size-fits-all approach would either be too conservative for Step 4 or too aggressive for Step 2

## Future Enhancement Opportunities

1. **Parallel Processing**: Add multi-threading for independent operations
2. **LLM-as-a-Judge Quality Assurance**: Implement evaluation pipeline using LLM to assess retrieval quality:
   - **Hallucination Detection**: Validate that retrieved content supports generated answers without introducing false information
   - **Relevance Scoring**: Score how well retrieved chunks match the user's query intent and information needs
   - **Pattern Quality Assessment**: Evaluate canonical pattern coherence and semantic consistency across support documents
3. **Pattern Evolution**: Track how canonical patterns change over time
4. **Add LLM Monitoring System**: Use tools like langfuse/langsmith for LLM monitoring
5. **Interactive Tuning**: Web interface for adjusting parameters and reviewing results

## Lessons Learned

1. **Dependency Management**: Modern tools like UV significantly improve development experience
2. **Agent Design**: Separate agents for different tasks produces better results than multipurpose agents
3. **Incremental Development**: Building step-by-step with clear interfaces enables faster debugging
4. **Configuration Management**: Structured configuration with validation prevents many deployment issues
5. **Async Processing**: AsyncOpenAI clients in labeling and promotion stages dramatically improve pipeline throughput by eliminating blocking API calls
6. **Prompt Engineering**: Centralizing prompts in dedicated modules enables safer iteration and better collaboration between engineers and prompt designers

This architecture balances simplicity with functionality, enabling both development efficiency and production robustness while maintaining the flexibility to adapt to changing requirements.
import os
import logging
import asyncio
import warnings
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from typing import List, Dict, Any, Optional

# Suppress ALL warnings globally
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

from .config import Config
from .utils import load_json, save_json

logger = logging.getLogger(__name__)

# Suppress LightRAG and nano-vectordb logging
logging.getLogger('lightrag').setLevel(logging.CRITICAL)
logging.getLogger('nano-vectordb').setLevel(logging.CRITICAL)

class LightRAGClient:
    """
    LightRAG integration client using custom KG insertion.
    
    This implementation bypasses LightRAG's text processing pipeline and directly
    inserts our pre-processed knowledge graph (chunks, canonical patterns, relationships).
    """
    
    def __init__(self, config: Config):
        """Initialize the LightRAG client.
        
        Args:
            config: Pipeline configuration with API keys and settings
        """
        self.config = config
        self.rag: Optional[LightRAG] = None
        self._initialized = False
        
    async def _get_llm_model_func(self):
        """Create OpenAI-compatible LLM function for LightRAG.
        
        Returns:
            Async function for LLM model completion
        """
        async def llm_model_func(
            prompt, 
            system_prompt=None, 
            history_messages=[], 
            **kwargs
        ):
            return await openai_complete_if_cache(
                model=self.config.llm_model,
                prompt=prompt,
                system_prompt=system_prompt,
                api_key=self.config.openai_api_key,
                base_url="https://api.openai.com/v1",
            )
        return llm_model_func
    
    def _get_embedding_func(self):
        """Create OpenAI embedding function for LightRAG.
        
        Returns:
            EmbeddingFunc instance configured for text-embedding-3-small
        """
        async def embedding_func(texts: List[str]) -> List[List[float]]:
            return await openai_embed(
                texts,
                model=self.config.embedding_model,
                api_key=self.config.openai_api_key,
                base_url="https://api.openai.com/v1"
            )
        
        return EmbeddingFunc(
            embedding_dim=1536,  # text-embedding-3-small dimension
            max_token_size=8191,  # text-embedding-3-small max tokens
            func=embedding_func
        )
    
    async def initialize(self):
        """Initialize LightRAG instance.
        
        Creates working directory and initializes storage backends.
        """
        if self._initialized:
            return
            
        # Suppress all warnings
        warnings.filterwarnings("ignore")
            
        logger.info("Initializing LightRAG client...")
        
        # Create working directory
        working_dir = Path(self.config.lightrag_working_dir)
        working_dir.mkdir(parents=True, exist_ok=True)
        
        # Create LightRAG instance
        self.rag = LightRAG(
            working_dir=str(working_dir),
            llm_model_func=await self._get_llm_model_func(),
            embedding_func=self._get_embedding_func(),
            # Configure chunking - though we'll bypass this with custom KG
            chunk_token_size=self.config.chunk_target_tokens,
            chunk_overlap_token_size=20,
        )
        
        # Initialize storage backends
        await self.rag.initialize_storages()
        
        
        self._initialized = True
        
        logger.info(f"LightRAG initialized with working directory: {working_dir}")
    
    def _format_chunks_for_custom_kg(self, chunks: List[Dict]) -> List[Dict]:
        """Format our chunks for LightRAG's custom KG insertion.
        
        Args:
            chunks: List of chunk dictionaries from pipeline
            
        Returns:
            List of formatted chunks for LightRAG
            
        Expected format: [{"content": str, "source_id": str, "file_path": str, "chunk_order_index": int}]
        """
        formatted_chunks = []
        
        for chunk in chunks:
            # Get source path from first document source (standardized structure)
            doc_sources = chunk["metadata"]["document_sources"]
            source_path = doc_sources[0]["source_path"] if doc_sources else "unknown"
            
            formatted_chunks.append({
                "content": chunk["text"],
                "source_id": chunk["id"],
                "file_path": source_path,
                "chunk_order_index": chunk["metadata"]["order"]
            })
        
        return formatted_chunks
    
    def _format_entities_for_custom_kg(self, pattern_bank: List[Dict]) -> List[Dict]:
        """Format canonical patterns as entities for custom KG insertion.
        
        Args:
            pattern_bank: List of canonical pattern dictionaries
            
        Returns:
            List of formatted entities for LightRAG
            
        Expected format: [{"entity_name": str, "entity_type": str, "description": str, "source_id": str}]
        """
        formatted_entities = []
        
        for pattern in pattern_bank:
            # Create meaningful source_id from filename and first 10 chars of description
            desc_prefix = pattern["desc"][:10].replace(" ", "_").replace("\n", "_")
            
            # Use first support doc as source file, fallback to pattern label
            support_docs = pattern.get("support_docs", [])
            if support_docs:
                source_file = support_docs[0].replace(".txt", "")
            else:
                source_file = pattern["label"]
            
            source_id = f"{source_file}_{desc_prefix}"
            
            formatted_entities.append({
                "entity_name": pattern["label"],
                "entity_type": "canonical_pattern",
                "description": pattern["desc"],
                "source_id": source_id
            })
        
        return formatted_entities
    
    def _format_relationships_for_custom_kg(self, kg_edges: List[Dict], chunks: List[Dict], pattern_bank: List[Dict]) -> List[Dict]:
        """Format relationships for custom KG insertion.
        
        Args:
            kg_edges: List of edge dictionaries from knowledge graph
            chunks: List of chunk dictionaries for lookup
            pattern_bank: List of pattern dictionaries for lookup
            
        Returns:
            List of formatted relationships for LightRAG
            
        Expected format: [{"src_id": str, "tgt_id": str, "description": str, "keywords": str, "weight": float, "source_id": str}]
        """
        formatted_relationships = []
        
        # Create lookup maps
        chunk_lookup = {chunk["id"]: chunk for chunk in chunks}
        pattern_lookup = {pattern["pattern_id"]: pattern for pattern in pattern_bank}
        
        for edge in kg_edges:
            src_id = edge["src"]
            dst_id = edge["dst"]
            edge_type = edge["type"]
            
            # Determine description based on relationship type
            if src_id in chunk_lookup and dst_id.startswith("doc:"):
                description = f"Chunk belongs to document"
            elif src_id in pattern_lookup:
                if dst_id in chunk_lookup:
                    description = f"Pattern appears in chunk"
                else:
                    description = f"Pattern relationship"
            else:
                description = f"Relationship of type {edge_type}"
            
            formatted_relationships.append({
                "src_id": src_id,
                "tgt_id": dst_id,
                "description": description,
                "keywords": edge_type,
                "weight": 1.0,
                "source_id": src_id  # Use src_id as source for the relationship
            })
        
        return formatted_relationships
    
    async def insert_custom_knowledge_graph(self, chunks: List[Dict], pattern_bank: List[Dict], kg_edges: List[Dict]) -> bool:
        """Insert our pre-processed knowledge graph directly into LightRAG.
        
        Args:
            chunks: List of text chunks with metadata
            pattern_bank: List of canonical patterns
            kg_edges: List of knowledge graph edges
            
        Returns:
            True if insertion successful, False otherwise
        """
        try:
            await self.initialize()
            
            # Format data for LightRAG's custom KG insertion
            custom_chunks = self._format_chunks_for_custom_kg(chunks)
            custom_entities = self._format_entities_for_custom_kg(pattern_bank)
            custom_relationships = self._format_relationships_for_custom_kg(kg_edges, chunks, pattern_bank)
            
            logger.info(f"Formatted {len(custom_chunks)} chunks, {len(custom_entities)} entities, {len(custom_relationships)} relationships")
            
            # Create custom KG structure
            custom_kg = {
                "chunks": custom_chunks,
                "entities": custom_entities,
                "relationships": custom_relationships
            }
            
            # Insert using LightRAG's custom KG method (suppress stderr for UNKNOWN warnings)
            logger.info("Inserting custom knowledge graph into LightRAG...")
            
            # Redirect stderr to suppress UNKNOWN warnings from LightRAG
            stderr_buffer = StringIO()
            with redirect_stderr(stderr_buffer):
                await self.rag.ainsert_custom_kg(custom_kg)
            
            logger.info("✓ Custom knowledge graph successfully inserted into LightRAG")
            return True
            
        except Exception as e:
            logger.error(f"Error inserting custom KG into LightRAG: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def query(self, query_text: str, mode: str = "hybrid") -> str:
        """Query the LightRAG knowledge graph with reference format post-processing.
        
        Args:
            query_text: Natural language query
            mode: Query mode ("local", "global", "hybrid", "naive")
            
        Returns:
            Query response with improved reference formatting
        """
        try:
            await self.initialize()
            
            result = await self.rag.aquery(
                query_text,
                param=QueryParam(mode=mode, enable_rerank=False)
            )
            
            # Post-process to fix reference format
            result = self._fix_reference_format(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error querying LightRAG: {e}")
            return f"Query error: {e}"
    
    def _fix_reference_format(self, response: str) -> str:
        """Simple post-processing to fix reference format.
        
        Args:
            response: Raw response from LightRAG
            
        Returns:
            Response with improved reference formatting
        """
        # Replace LightRAG's default reference format with our preferred format
        response = response.replace("[KG]", "[Knowledge Graph]")
        response = response.replace("[DC]", "[Document Chunk]") 
        response = response.replace("[KG/DC]", "[Knowledge Graph/Document Chunk]")
        
        # Replace generic custom_kg with actual source information if possible
        if "custom_kg" in response:
            response = response.replace("[Knowledge Graph] custom_kg", "[Knowledge Graph] (entities from pattern analysis)")
            
        return response
    
    def run(self, chunks_file: str, pattern_bank_file: str, kg_edges_file: str) -> str:
        """Main entry point for LightRAG integration (sync wrapper).
        
        Args:
            chunks_file: Path to chunks JSON file
            pattern_bank_file: Path to pattern bank JSON file
            kg_edges_file: Path to KG edges JSON file
            
        Returns:
            Path to integration manifest file
        """
        return asyncio.run(self._run_async(chunks_file, pattern_bank_file, kg_edges_file))
    
    async def _run_async(self, chunks_file: str, pattern_bank_file: str, kg_edges_file: str) -> str:
        """Async implementation of the main LightRAG integration.
        
        Args:
            chunks_file: Path to chunks JSON file
            pattern_bank_file: Path to pattern bank JSON file
            kg_edges_file: Path to KG edges JSON file
            
        Returns:
            Path to integration manifest file
        """
        logger.info("Step 7: Starting LightRAG custom KG integration")
        
        # Load data
        chunks = load_json(Path(chunks_file)) or []
        pattern_bank = load_json(Path(pattern_bank_file)) or []
        kg_edges = load_json(Path(kg_edges_file)) or []
        
        logger.info(f"Loaded {len(chunks)} chunks, {len(pattern_bank)} patterns, {len(kg_edges)} edges")
        
        # Load previous manifest
        manifest_file = self.config.output_dir / "07_upsert_manifest.json"
        manifest = load_json(manifest_file) or {"doc_chunks": {}, "pattern_ids": []}
        
        # Group chunks by document for tracking
        doc_chunks = {}
        for chunk in chunks:
            # Get doc_ids from document_sources (standardized structure)
            doc_sources = chunk["metadata"]["document_sources"]
            for doc_source in doc_sources:
                doc_id = doc_source["doc_id"]
                if doc_id not in doc_chunks:
                    doc_chunks[doc_id] = []
                doc_chunks[doc_id].append(chunk["id"])
        
        # Insert custom knowledge graph
        success = await self.insert_custom_knowledge_graph(chunks, pattern_bank, kg_edges)
        
        if success:
            # Update manifest
            manifest["doc_chunks"] = doc_chunks
            manifest["pattern_ids"] = [p["pattern_id"] for p in pattern_bank]
            manifest["lightrag_working_dir"] = self.config.lightrag_working_dir
            manifest["total_chunks"] = len(chunks)
            manifest["total_patterns"] = len(pattern_bank)
            manifest["integration_method"] = "custom_kg_insertion"
            
            save_json(manifest, manifest_file)
            
            logger.info(f"✓ Successfully integrated {len(chunks)} chunks and {len(pattern_bank)} patterns")
            logger.info(f"✓ LightRAG storage: {self.config.lightrag_working_dir}")
            logger.info(f"✓ Manifest saved to {manifest_file}")
        else:
            logger.error("✗ Failed to integrate data with LightRAG")
        
        return str(manifest_file)


# Convenience function for testing queries
async def test_query(config: Config, query: str, mode: str = "hybrid") -> str:
    """Test query function for standalone testing.
    
    Args:
        config: Pipeline configuration
        query: Query text to test
        mode: Query mode to use
        
    Returns:
        Query response
    """
    client = LightRAGClient(config)
    return await client.query(query, mode)
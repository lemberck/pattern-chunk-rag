import logging
from pathlib import Path
from typing import List, Dict, Any, Set

from .config import Config
from .utils import load_json, save_json

logger = logging.getLogger(__name__)


class KnowledgeGraphBuilder:
    """Builds knowledge graph nodes and edges from patterns and chunks.
    
    Creates a structured graph representation with patterns, chunks, and documents
    as nodes, connected by typed edges for LightRAG insertion.
    """
    def __init__(self, config: Config):
        """Initialize the knowledge graph builder.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
    
    def create_pattern_nodes(self, pattern_bank: List[Dict]) -> List[Dict[str, Any]]:
        """Create nodes for canonical patterns.
        
        Args:
            pattern_bank: List of canonical pattern dictionaries
            
        Returns:
            List of pattern node dictionaries
        """
        nodes = []
        for pattern in pattern_bank:
            node = {
                "id": pattern["pattern_id"],
                "type": "pattern",
                "label": pattern["label"],
                "desc": pattern["desc"]
            }
            nodes.append(node)
        return nodes
    
    def create_chunk_nodes(self, chunks: List[Dict]) -> List[Dict[str, Any]]:
        """Create nodes for text chunks.
        
        Args:
            chunks: List of chunk dictionaries with metadata
            
        Returns:
            List of chunk node dictionaries with descriptive labels
        """
        nodes = []
        for chunk in chunks:
            metadata = chunk["metadata"]
            
            # All chunks now have standardized structure
            if metadata.get("canonical_pattern"):
                # Canonical pattern chunk
                pattern_info = metadata["canonical_pattern"]
                pattern_label = pattern_info.get("label", "unknown_pattern")
                doc_count = len(metadata.get("document_sources", []))
                label = f"{pattern_label} ({doc_count} docs)"
            else:
                # Document chunk (no pattern)
                doc_source = metadata["document_sources"][0]
                doc_name = doc_source["doc_name"]
                start_id = metadata.get("start_sentence_id", 0)
                end_id = metadata.get("end_sentence_id", 0)
                label = f"{doc_name}:{start_id:04d}-{end_id:04d}"
            
            node = {
                "id": chunk["id"],
                "type": "chunk",
                "label": label
            }
            nodes.append(node)
        return nodes
    
    def create_document_nodes(self, chunks: List[Dict]) -> List[Dict[str, Any]]:
        """Create nodes for source documents.
        
        Args:
            chunks: List of chunk dictionaries containing document sources
            
        Returns:
            List of unique document node dictionaries
        """
        # Get unique documents from standardized document_sources
        documents = {}
        for chunk in chunks:
            doc_sources = chunk["metadata"]["document_sources"]
            for doc_source in doc_sources:
                doc_id = doc_source["doc_id"]
                if doc_id not in documents:
                    documents[doc_id] = {
                        "id": f"doc:{doc_id}",
                        "type": "document",
                        "label": doc_source["doc_name"]
                    }
        
        return list(documents.values())
    
    def create_pattern_chunk_edges(self, chunks: List[Dict]) -> List[Dict[str, Any]]:
        """Create edges between patterns and their supporting chunks.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            List of pattern-to-chunk edge dictionaries
        """
        edges = []
        for chunk in chunks:
            canonical_pattern = chunk["metadata"].get("canonical_pattern")
            if canonical_pattern:  # Only canonical pattern chunks have this
                edge = {
                    "src": canonical_pattern["id"],
                    "dst": chunk["id"],
                    "type": "supports"
                }
                edges.append(edge)
        return edges
    
    def create_chunk_document_edges(self, chunks: List[Dict]) -> List[Dict[str, Any]]:
        """Create edges between chunks and their source documents.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            List of chunk-to-document edge dictionaries
        """
        edges = []
        for chunk in chunks:
            doc_sources = chunk["metadata"]["document_sources"]
            for doc_source in doc_sources:
                edge = {
                    "src": chunk["id"],
                    "dst": f"doc:{doc_source['doc_id']}",
                    "type": "part_of"
                }
                edges.append(edge)
        return edges
    
    def run(self, pattern_bank_file: str, chunks_file: str) -> tuple[str, str]:
        """Build knowledge graph from patterns and chunks.
        
        Args:
            pattern_bank_file: Path to pattern bank file
            chunks_file: Path to chunks file
            
        Returns:
            Tuple of (nodes_file, edges_file) paths
        """
        logger.info("Step 6: Building knowledge graph")
        
        # Load data
        pattern_bank = load_json(Path(pattern_bank_file)) or []
        chunks = load_json(Path(chunks_file)) or []
        
        # Create nodes
        pattern_nodes = self.create_pattern_nodes(pattern_bank)
        chunk_nodes = self.create_chunk_nodes(chunks)
        document_nodes = self.create_document_nodes(chunks)
        
        all_nodes = pattern_nodes + chunk_nodes + document_nodes
        
        # Create edges
        pattern_chunk_edges = self.create_pattern_chunk_edges(chunks)
        chunk_document_edges = self.create_chunk_document_edges(chunks)
        
        all_edges = pattern_chunk_edges + chunk_document_edges
        
        # Save nodes and edges
        nodes_file = self.config.output_dir / "06_kg_nodes.json"
        edges_file = self.config.output_dir / "06_kg_edges.json"
        
        save_json(all_nodes, nodes_file)
        save_json(all_edges, edges_file)
        
        logger.info(f"Created {len(all_nodes)} nodes ({len(pattern_nodes)} patterns, {len(chunk_nodes)} chunks, {len(document_nodes)} documents)")
        logger.info(f"Created {len(all_edges)} edges ({len(pattern_chunk_edges)} pattern-chunk, {len(chunk_document_edges)} chunk-document)")
        logger.info(f"Saved nodes to {nodes_file}")
        logger.info(f"Saved edges to {edges_file}")
        
        return str(nodes_file), str(edges_file)
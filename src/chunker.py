import logging
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

from .config import Config
from .utils import load_jsonl, save_json, generate_chunk_id, load_json

logger = logging.getLogger(__name__)


class SmartChunker:
    """Creates intelligent chunks based on canonical patterns and documents.
    
    Groups sentences by their canonical topics and creates chunks optimized
    for knowledge graph insertion, with fallback document-based chunking.
    """
    def __init__(self, config: Config):
        """Initialize the smart chunker.
        
        Args:
            config: Pipeline configuration with chunking parameters
        """
        self.config = config
    
    def load_data(self, sentences_file: str, mapping_file: str, pattern_bank_file: str) -> tuple[Dict[str, List], Dict[str, str], Dict[str, Dict]]:
        """Load and organize data for chunking.
        
        Args:
            sentences_file: Path to sentences file
            mapping_file: Path to sentence-to-canonical mapping file
            pattern_bank_file: Path to pattern bank file
            
        Returns:
            Tuple of (doc_sentences, sentence_to_canonical, pattern_labels)
        """
        # Load sentences grouped by document
        sentences = load_jsonl(Path(sentences_file))
        doc_sentences = defaultdict(list)
        
        for sentence in sentences:
            doc_sentences[sentence["doc_id"]].append(sentence)
        
        # Sort sentences by sentence_id within each document
        for doc_id in doc_sentences:
            doc_sentences[doc_id].sort(key=lambda x: x["sentence_id"])
        
        # Load sentence-to-canonical mapping
        mappings = load_jsonl(Path(mapping_file))
        sentence_to_canonical = {}
        
        for mapping in mappings:
            key = f"{mapping['doc_id']}:{mapping['sentence_id']}"
            sentence_to_canonical[key] = mapping["canonical_topic_id"]
        
        # Load pattern bank for human-readable labels
        pattern_bank = load_json(Path(pattern_bank_file))
        pattern_labels = {}
        for pattern in pattern_bank:
            pattern_labels[pattern["pattern_id"]] = {
                "label": pattern["label"],
                "desc": pattern["desc"]
            }
        
        return doc_sentences, sentence_to_canonical, pattern_labels
    
    def group_sentences_by_canonical_pattern(self, doc_sentences: Dict[str, List], sentence_to_canonical: Dict[str, str]) -> Dict[str, List]:
        """Group all sentences by their canonical pattern.
        
        Args:
            doc_sentences: Dictionary mapping doc_id to list of sentences
            sentence_to_canonical: Dictionary mapping sentence keys to canonical IDs
            
        Returns:
            Dictionary mapping canonical IDs to lists of sentence info dictionaries
        """
        canonical_groups = defaultdict(list)
        
        for doc_id, sentences in doc_sentences.items():
            for sentence in sentences:
                key = f"{doc_id}:{sentence['sentence_id']}"
                canonical_id = sentence_to_canonical.get(key)
                
                if canonical_id:
                    # Add sentence info including doc metadata
                    sentence_info = {
                        "sentence": sentence,
                        "doc_id": doc_id,
                        "sentence_id": sentence["sentence_id"]
                    }
                    canonical_groups[canonical_id].append(sentence_info)
        
        return canonical_groups
    
    def create_canonical_chunk(self, canonical_id: str, sentence_infos: List[Dict], pattern_labels: Dict[str, Dict], chunk_order: int) -> Dict[str, Any]:
        """Create one chunk per canonical pattern containing all relevant sentences.
        
        Args:
            canonical_id: ID of the canonical pattern
            sentence_infos: List of sentence info dictionaries
            pattern_labels: Dictionary of pattern metadata
            chunk_order: Order index for the chunk
            
        Returns:
            Chunk dictionary with combined text and metadata
        """
        
        # Combine all text from sentences in this canonical group
        all_text_parts = []
        total_tokens = 0
        doc_sources = set()
        
        for info in sentence_infos:
            sentence = info["sentence"]
            all_text_parts.append(sentence["text"])
            total_tokens += sentence["token_count"]
            doc_sources.add((info["doc_id"], sentence["doc_name"], sentence["source_path"]))
        
        # Create combined text
        combined_text = " ".join(all_text_parts)
        
        # Get pattern info
        pattern_info = {
            "id": canonical_id,
        }
        if canonical_id in pattern_labels:
            pattern_info["label"] = pattern_labels[canonical_id]["label"]
            pattern_info["desc"] = pattern_labels[canonical_id]["desc"]
        
        # Create unique chunk ID based on canonical pattern
        chunk_id = f"canonical_chunk_{canonical_id}"
        
        # Create document sources list
        doc_sources_list = []
        for doc_id, doc_name, source_path in doc_sources:
            doc_sources_list.append({
                "doc_id": doc_id,
                "doc_name": doc_name,
                "source_path": source_path
            })
        
        return {
            "id": chunk_id,
            "text": combined_text,
            "metadata": {
                "canonical_pattern": pattern_info,
                "document_sources": doc_sources_list,
                "sentence_count": len(sentence_infos),
                "token_count": total_tokens,
                "order": chunk_order,
                "chunk_type": "canonical_pattern_chunk"
            }
        }
    
    def create_document_chunks(self, doc_id: str, sentences: List[Dict]) -> List[Dict[str, Any]]:
        """Create document chunks for documents with no patterns.
        
        Uses standardized metadata structure and respects token limits
        with overlapping for optimal chunking.
        
        Args:
            doc_id: Document identifier
            sentences: List of sentence dictionaries for this document
            
        Returns:
            List of document chunk dictionaries
        """
        chunks = []
        doc_name = sentences[0]["doc_name"]
        source_path = sentences[0]["source_path"]
        
        i = 0
        chunk_order = 0
        
        while i < len(sentences):
            chunk_start = i
            current_tokens = 0
            
            # Build chunk up to target size
            while i < len(sentences) and current_tokens < self.config.chunk_target_tokens:
                new_tokens = current_tokens + sentences[i]["token_count"]
                if new_tokens > self.config.chunk_max_tokens:
                    break
                current_tokens = new_tokens
                i += 1
            
            # Only create chunk if it meets minimum size or is the last chunk
            if current_tokens >= self.config.chunk_min_tokens or chunk_start == len(sentences) - 1:
                chunk_end = i - 1
                chunk_sentences = sentences[chunk_start:chunk_end + 1]
                text = " ".join(s["text"] for s in chunk_sentences)
                
                chunk_id = generate_chunk_id(doc_id, sentences[chunk_start]["sentence_id"], sentences[chunk_end]["sentence_id"])
                
                # Use standardized metadata structure
                chunk = {
                    "id": chunk_id,
                    "text": text,
                    "metadata": {
                        "canonical_pattern": None,  # No pattern for document chunks
                        "document_sources": [
                            {
                                "doc_id": doc_id,
                                "doc_name": doc_name,
                                "source_path": source_path
                            }
                        ],
                        "sentence_count": len(chunk_sentences),
                        "token_count": current_tokens,
                        "order": chunk_order,
                        "chunk_type": "document_chunk",
                        # Keep these for document chunk specific info
                        "start_sentence_id": sentences[chunk_start]["sentence_id"],
                        "end_sentence_id": sentences[chunk_end]["sentence_id"]
                    }
                }
                chunks.append(chunk)
                chunk_order += 1
                
                # Add overlap for next chunk
                if i < len(sentences):
                    overlap_start = max(chunk_start, chunk_end - self.config.chunk_overlap_sents + 1)
                    i = overlap_start
        
        return chunks
    
    def run(self, sentences_file: str, mapping_file: str) -> str:
        """Create smart chunks from sentences and patterns.
        
        Args:
            sentences_file: Path to sentences file
            mapping_file: Path to sentence-to-canonical mapping file
            
        Returns:
            Path to output chunks file
        """
        logger.info("Step 5: Starting smart chunking")
        
        pattern_bank_file = self.config.output_dir / "04_pattern_bank.json"
        doc_sentences, sentence_to_canonical, pattern_labels = self.load_data(sentences_file, mapping_file, pattern_bank_file)
        
        # Group sentences by canonical pattern
        canonical_groups = self.group_sentences_by_canonical_pattern(doc_sentences, sentence_to_canonical)
        logger.info(f"Found {len(canonical_groups)} canonical patterns with sentences")
        
        all_chunks = []
        chunk_order = 0
        
        # Create one chunk per canonical pattern
        for canonical_id, sentence_infos in canonical_groups.items():
            pattern_label = pattern_labels.get(canonical_id, {}).get("label", canonical_id)
            logger.info(f"Creating chunk for pattern: {pattern_label} ({len(sentence_infos)} sentences)")
            
            chunk = self.create_canonical_chunk(canonical_id, sentence_infos, pattern_labels, chunk_order)
            all_chunks.append(chunk)
            chunk_order += 1
        
        # Create chunks for documents with no patterns (fallback) - if enabled
        if self.config.include_non_pattern_chunks:
            docs_with_patterns = set()
            for sentence_infos in canonical_groups.values():
                for info in sentence_infos:
                    docs_with_patterns.add(info["doc_id"])
            
            for doc_id, sentences in doc_sentences.items():
                if doc_id not in docs_with_patterns:
                    doc_chunks = self.create_document_chunks(doc_id, sentences)
                    all_chunks.extend(doc_chunks)
                    chunk_order += len(doc_chunks)
                    
                    doc_name = sentences[0]["doc_name"] if sentences else "unknown"
                    logger.info(f"Created {len(doc_chunks)} document chunks for {doc_name} (no patterns)")
        else:
            logger.info("Skipping non-pattern chunks (include_non_pattern_chunks=False)")
        
        # Log summary
        canonical_chunks = [c for c in all_chunks if c["metadata"]["chunk_type"] == "canonical_pattern_chunk"]
        doc_chunks = [c for c in all_chunks if c["metadata"]["chunk_type"] == "document_chunk"]
        
        logger.info(f"Created {len(canonical_chunks)} canonical pattern chunks")
        logger.info(f"Created {len(doc_chunks)} document chunks")
        
        output_file = self.config.output_dir / "05_chunks.json"
        save_json(all_chunks, output_file)
        
        logger.info(f"Created {len(all_chunks)} total chunks to {output_file}")
        
        return str(output_file)
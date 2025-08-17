import logging
from pathlib import Path
from typing import List, Dict, Any, Set
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

from .config import Config
from .utils import load_jsonl, save_json, load_json, generate_group_id

logger = logging.getLogger(__name__)


class TopicGrouper:
    """Groups similar topic labels using embedding-based similarity.
    
    Uses OpenAI embeddings and cosine similarity to group semantically
    related topic labels into candidate groups for canonical promotion.
    """
    def __init__(self, config: Config):
        """Initialize the topic grouper.
        
        Args:
            config: Pipeline configuration with API keys and similarity threshold
        """
        self.config = config
        self.client = OpenAI(api_key=config.openai_api_key)
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding vector for text using OpenAI API.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        try:
            response = self.client.embeddings.create(
                model=self.config.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding for '{text}': {e}")
            # Return zero embedding as fallback
            return [0.0] * 1536  # text-embedding-3-small dimension
    
    def find_similar_group(self, label_embedding: List[float], existing_groups: List[Dict]) -> Dict[str, Any]:
        """Find the most similar existing group for a label.
        
        Args:
            label_embedding: Embedding vector of the label
            existing_groups: List of existing candidate groups
            
        Returns:
            Most similar group or None if no group exceeds threshold
        """
        best_group = None
        best_similarity = -1
        
        for group in existing_groups:
            if "anchor_embedding" not in group:
                continue
                
            similarity = cosine_similarity(
                [label_embedding], 
                [group["anchor_embedding"]]
            )[0][0]
            
            if similarity >= self.config.label_sim_threshold and similarity > best_similarity:
                best_similarity = similarity
                best_group = group
        
        return best_group, best_similarity
    
    def create_group(self, label: str, desc: str, doc_name: str, embedding: List[float]) -> Dict[str, Any]:
        """Create a new candidate group.
        
        Args:
            label: Anchor label for the group
            desc: Description of the label
            doc_name: Source document name
            embedding: Embedding vector of the label
            
        Returns:
            New group dictionary
        """
        group_id = generate_group_id(label)
        return {
            "group_id": group_id,
            "anchor_label": label,
            "anchor_embedding": embedding,
            "labels": [label],
            "descriptions": [desc],
            "support_docs": [doc_name],
            "support_docs_count": 1
        }
    
    def add_to_group(self, group: Dict[str, Any], label: str, desc: str, doc_name: str) -> None:
        """Add a label to an existing group.
        
        Args:
            group: Existing group to modify
            label: Label to add
            desc: Description of the label
            doc_name: Source document name
        """
        if label not in group["labels"]:
            group["labels"].append(label)
        if desc not in group["descriptions"]:
            group["descriptions"].append(desc)
        if doc_name not in group["support_docs"]:
            group["support_docs"].append(doc_name)
            group["support_docs_count"] = len(group["support_docs"])
    
    def run(self, topics_file: str) -> str:
        """Group similar topics into candidate groups.
        
        Args:
            topics_file: Path to input topics file
            
        Returns:
            Path to output candidate groups file
        """
        logger.info("Step 3: Starting similarity grouping")
        
        # Load previous groups if they exist
        existing_groups_file = self.config.output_dir / "03_candidate_groups.json"
        existing_groups = load_json(existing_groups_file) or []
        
        # Load sentence topics
        sentence_topics = load_jsonl(Path(topics_file))
        
        # Get unique labels with their contexts
        unique_labels = {}
        for item in sentence_topics:
            label = item["topic"]
            desc = item["desc"]
            doc_name = item["doc_name"]
            
            if label not in unique_labels:
                unique_labels[label] = {
                    "desc": desc,
                    "docs": set([doc_name])
                }
            else:
                unique_labels[label]["docs"].add(doc_name)
        
        # Process each unique label
        for i, (label, info) in enumerate(unique_labels.items()):
            if i % 10 == 0:
                logger.info(f"Processing label {i+1}/{len(unique_labels)}: {label}")
            
            # Create label text for embedding
            label_text = f"{label} :: {info['desc']}"
            
            # Check if this label is already in existing groups
            label_found = False
            for group in existing_groups:
                if label in group["labels"]:
                    # Update support docs for existing label
                    for doc_name in info["docs"]:
                        if doc_name not in group["support_docs"]:
                            group["support_docs"].append(doc_name)
                    group["support_docs_count"] = len(group["support_docs"])
                    label_found = True
                    break
            
            if label_found:
                continue
            
            # Get embedding for new label
            embedding = self.get_embedding(label_text)
            
            # Find similar group
            similar_group, similarity = self.find_similar_group(embedding, existing_groups)
            
            if similar_group:
                # Add to existing group
                for doc_name in info["docs"]:
                    self.add_to_group(similar_group, label, info["desc"], doc_name)
                logger.debug(f"Added '{label}' to group {similar_group['group_id']} (similarity: {similarity:.3f})")
            else:
                # Create new group
                new_group = self.create_group(label, info["desc"], list(info["docs"])[0], embedding)
                for doc_name in list(info["docs"])[1:]:
                    self.add_to_group(new_group, label, info["desc"], doc_name)
                existing_groups.append(new_group)
                logger.debug(f"Created new group {new_group['group_id']} for '{label}'")
        
        # Clean up embeddings before saving (too large for JSON)
        for group in existing_groups:
            if "anchor_embedding" in group:
                del group["anchor_embedding"]
        
        # Save updated groups
        output_file = self.config.output_dir / "03_candidate_groups.json"
        save_json(existing_groups, output_file)
        
        logger.info(f"Created/updated {len(existing_groups)} candidate groups to {output_file}")
        
        return str(output_file)
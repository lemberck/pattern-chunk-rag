import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from openai import AsyncOpenAI

from .config import Config
from .utils import load_json, save_json, save_jsonl, load_jsonl, generate_pattern_id
from .prompts.prompts import CANONICAL_PROMOTION_PROMPT

logger = logging.getLogger(__name__)


class TopicPromoter:
    """Agent 2: Promotes qualified topic groups to canonical status.
    
    Analyzes candidate groups and creates canonical topic descriptions
    for patterns that appear across multiple documents.
    """
    def __init__(self, config: Config):
        """Initialize the topic promoter.
        
        Args:
            config: Pipeline configuration with API keys and thresholds
        """
        self.config = config
        self.client = AsyncOpenAI(api_key=config.openai_api_key)
    
    
    async def create_canonical_topic(self, group: Dict[str, Any]) -> Dict[str, Any]:
        """Create a canonical topic from a candidate group.
        
        Args:
            group: Candidate group with labels and metadata
            
        Returns:
            Canonical topic dictionary with pattern ID and description
        """
        labels_str = ", ".join(group["labels"])
        descs_str = "; ".join(group["descriptions"])
        prompt = CANONICAL_PROMOTION_PROMPT.format(
            labels=labels_str,
            descriptions=descs_str
        )
        
        try:
            response = await self.client.chat.completions.create(
                model=self.config.chat_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=150
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse the response
            if " :: " in content:
                canonical_label, desc = content.split(" :: ", 1)
                canonical_label = canonical_label.strip()
                desc = desc.strip()
            else:
                # Fallback
                canonical_label = group["anchor_label"]
                desc = f"Canonical topic derived from: {', '.join(group['labels'])}"
            
            pattern_id = generate_pattern_id(canonical_label)
            
            return {
                "pattern_id": pattern_id,
                "label": canonical_label,
                "desc": desc,
                "topic_type": "canonical",
                "support_docs": group["support_docs"],
                "support_docs_count": group["support_docs_count"],
                "source_group_id": group["group_id"]
            }
            
        except Exception as e:
            logger.error(f"Error creating canonical topic: {e}")
            # Fallback canonical topic
            pattern_id = generate_pattern_id(group["anchor_label"])
            return {
                "pattern_id": pattern_id,
                "label": group["anchor_label"],
                "desc": f"Canonical topic derived from: {', '.join(group['labels'])}",
                "topic_type": "canonical",
                "support_docs": group["support_docs"],
                "support_docs_count": group["support_docs_count"],
                "source_group_id": group["group_id"]
            }
    
    def create_sentence_mapping(self, pattern_bank: List[Dict], candidate_groups: List[Dict], sentence_topics: List[Dict]) -> List[Dict]:
        """Map sentences to their canonical topics.
        
        Args:
            pattern_bank: List of canonical patterns
            candidate_groups: List of candidate groups
            sentence_topics: List of sentence-topic pairs
            
        Returns:
            List of sentence-to-canonical mappings
        """
        # Create mapping from labels to pattern_ids
        label_to_pattern = {}
        
        for pattern in pattern_bank:
            source_group_id = pattern["source_group_id"]
            pattern_id = pattern["pattern_id"]
            
            # Find the source group
            for group in candidate_groups:
                if group["group_id"] == source_group_id:
                    for label in group["labels"]:
                        label_to_pattern[label] = pattern_id
                    break
        
        # Map sentences to canonical topics
        sentence_mapping = []
        for sentence_topic in sentence_topics:
            label = sentence_topic["topic"]
            canonical_topic_id = label_to_pattern.get(label)
            
            if canonical_topic_id:
                mapping = {
                    "doc_id": sentence_topic["doc_id"],
                    "doc_name": sentence_topic["doc_name"],
                    "sentence_id": sentence_topic["sentence_id"],
                    "canonical_topic_id": canonical_topic_id,
                    "raw_topic": label
                }
                sentence_mapping.append(mapping)
        
        return sentence_mapping
    
    async def _run_async(self, groups_file: str, topics_file: str) -> tuple[str, str]:
        """Process candidate groups asynchronously for promotion.
        
        Args:
            groups_file: Path to candidate groups file
            topics_file: Path to sentence topics file
            
        Returns:
            Tuple of (pattern_bank_file, mapping_file) paths
        """
        logger.info("Step 4: Starting canonical topic promotion (async)")
        
        # Load candidate groups and sentence topics
        candidate_groups = load_json(Path(groups_file))
        sentence_topics = load_jsonl(Path(topics_file))
        
        # Count total documents for threshold calculation from candidate groups
        # This gives us the complete picture of all documents in the system
        all_doc_names = set()
        for group in candidate_groups:
            for doc_name in group.get("support_docs", []):
                all_doc_names.add(doc_name)
        total_docs = len(all_doc_names)
        
        threshold = self.config.get_canonical_threshold(total_docs)
        logger.info(f"Canonical threshold: {threshold} docs (total: {total_docs})")
        
        # Load existing pattern bank
        pattern_bank_file = self.config.output_dir / "04_pattern_bank.json"
        existing_patterns = load_json(pattern_bank_file) or []
        
        # Track which groups are already canonical
        existing_pattern_groups = {p["source_group_id"] for p in existing_patterns}
        
        # Process qualified groups concurrently
        qualified_groups = [
            group for group in candidate_groups 
            if group["group_id"] not in existing_pattern_groups 
            and group["support_docs_count"] >= threshold
        ]
        
        promoted_count = 0
        if qualified_groups:
            logger.info(f"Processing {len(qualified_groups)} groups concurrently for canonical promotion...")
            tasks = [self.create_canonical_topic(group) for group in qualified_groups]
            canonical_topics = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, canonical_topic in enumerate(canonical_topics):
                if isinstance(canonical_topic, Exception):
                    logger.error(f"Error processing group {qualified_groups[i]['group_id']}: {canonical_topic}")
                    continue
                
                # Check if pattern already exists (by pattern_id)
                pattern_exists = False
                for j, existing in enumerate(existing_patterns):
                    if existing["pattern_id"] == canonical_topic["pattern_id"]:
                        # Update existing pattern
                        existing_patterns[j] = canonical_topic
                        pattern_exists = True
                        break
                
                if not pattern_exists:
                    existing_patterns.append(canonical_topic)
                
                promoted_count += 1
                logger.info(f"Promoted group {qualified_groups[i]['group_id']} to canonical: {canonical_topic['label']}")
        
        # Save pattern bank
        save_json(existing_patterns, pattern_bank_file)
        
        # Create sentence-to-canonical mapping
        sentence_mapping = self.create_sentence_mapping(existing_patterns, candidate_groups, sentence_topics)
        
        mapping_file = self.config.output_dir / "04_sentence_to_canonical.jsonl"
        save_jsonl(sentence_mapping, mapping_file)
        
        logger.info(f"Promoted {promoted_count} new canonical topics")
        logger.info(f"Pattern bank saved to {pattern_bank_file}")
        logger.info(f"Sentence mapping saved to {mapping_file}")
        
        return str(pattern_bank_file), str(mapping_file)
    
    def run(self, groups_file: str, topics_file: str) -> tuple[str, str]:
        """Promote qualified groups to canonical topics (sync wrapper).
        
        Args:
            groups_file: Path to candidate groups file
            topics_file: Path to sentence topics file
            
        Returns:
            Tuple of (pattern_bank_file, mapping_file) paths
        """
        """Synchronous wrapper for async processing"""
        return asyncio.run(self._run_async(groups_file, topics_file))
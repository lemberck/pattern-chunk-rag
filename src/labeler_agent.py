import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from openai import AsyncOpenAI

from .config import Config
from .utils import load_jsonl, save_jsonl
from .prompts.prompts import TOPIC_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)


class TopicLabeler:
    """Agent 1: Extracts candidate topics from sentences using LLM.
    
    Processes sentences asynchronously in batches to extract topic labels
    and descriptions from document content.
    """
    def __init__(self, config: Config):
        """Initialize the topic labeler.
        
        Args:
            config: Pipeline configuration containing API keys and model settings
        """
        self.config = config
        self.client = AsyncOpenAI(api_key=config.openai_api_key)
    
    
    async def label_sentence(self, sentence_text: str) -> Dict[str, str]:
        """Extract topic label and description from a sentence.
        
        Args:
            sentence_text: Text of the sentence to analyze
            
        Returns:
            Dictionary with topic, description, and topic_type
        """
        prompt = TOPIC_EXTRACTION_PROMPT.format(sentence=sentence_text)
        
        try:
            response = await self.client.chat.completions.create(
                model=self.config.chat_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=100
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse the response: "topic_name :: description"
            if " :: " in content:
                topic, desc = content.split(" :: ", 1)
                return {
                    "topic": topic.strip(),
                    "desc": desc.strip(),
                    "topic_type": "candidate"
                }
            else:
                # Fallback if format is wrong
                logger.warning(f"Unexpected format from Agent 1: {content}")
                topic = content.split()[0] if content else "unknown_topic"
                return {
                    "topic": topic.strip(),
                    "desc": "extracted topic",
                    "topic_type": "candidate"
                }
                
        except Exception as e:
            logger.error(f"Error labeling sentence: {e}")
            return {
                "topic": "error_topic",
                "desc": "failed to extract topic",
                "topic_type": "candidate"
            }
    
    async def _label_sentence_with_metadata(self, sentence_data: Dict[str, Any]) -> Dict[str, Any]:
        """Label a sentence and preserve metadata.
        
        Args:
            sentence_data: Sentence dictionary with metadata
            
        Returns:
            Enhanced sentence data with topic information
        """
        """Label a single sentence and return with metadata"""
        topic_data = await self.label_sentence(sentence_data["text"])
        
        return {
            "doc_id": sentence_data["doc_id"],
            "doc_name": sentence_data["doc_name"],
            "sentence_id": sentence_data["sentence_id"],
            "topic": topic_data["topic"],
            "desc": topic_data["desc"],
            "topic_type": topic_data["topic_type"]
        }
    
    async def _run_async(self, sentences_file: str) -> str:
        """Process sentences asynchronously in batches.
        
        Args:
            sentences_file: Path to input sentences file
            
        Returns:
            Path to output topics file
        """
        logger.info("Step 2: Starting topic labeling (async)")
        
        sentences = load_jsonl(Path(sentences_file))
        total = len(sentences)
        logger.info(f"Processing {total} sentences concurrently...")
        
        # Process sentences concurrently in batches to avoid overwhelming the API
        batch_size = 10  # can be adjusted based on API rate limits
        labeled_sentences = []
        
        for i in range(0, total, batch_size):
            batch = sentences[i:i + batch_size]
            batch_end = min(i + batch_size, total)
            logger.info(f"Processing batch {i+1}-{batch_end}/{total}")
            
            # Process batch concurrently
            tasks = [self._label_sentence_with_metadata(sentence_data) for sentence_data in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and log errors
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Error processing sentence {i+j+1}: {result}")
                    # Create fallback entry
                    result = {
                        "doc_id": batch[j]["doc_id"],
                        "doc_name": batch[j]["doc_name"],
                        "sentence_id": batch[j]["sentence_id"],
                        "topic": "error_topic",
                        "desc": "failed to extract topic",
                        "topic_type": "candidate"
                    }
                labeled_sentences.append(result)
        
        output_file = self.config.output_dir / "02_sentence_topics.jsonl"
        save_jsonl(labeled_sentences, output_file)
        
        logger.info(f"Labeled {len(labeled_sentences)} sentences to {output_file}")
        return str(output_file)
    
    def run(self, sentences_file: str) -> str:
        """Process sentences and extract topics (sync wrapper).
        
        Args:
            sentences_file: Path to input sentences file
            
        Returns:
            Path to output topics file
        """
        """Synchronous wrapper for async processing"""
        return asyncio.run(self._run_async(sentences_file))
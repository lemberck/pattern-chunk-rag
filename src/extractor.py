import logging
from pathlib import Path
from typing import List, Dict, Any

from .config import Config
from .utils import (
    get_tokenizer, 
    count_tokens, 
    generate_doc_id, 
    split_sentences,
    save_jsonl
)

logger = logging.getLogger(__name__)


class DocumentExtractor:
    """Extracts and tokenizes sentences from text documents.
    
    Processes .txt files and splits them into sentences with metadata
    including token counts and document provenance.
    """
    def __init__(self, config: Config):
        """Initialize the document extractor.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.tokenizer = get_tokenizer(config.tokenizer)
    
    def extract_documents(self, specific_files: List[Path] = None) -> List[Dict[str, Any]]:
        """Extract sentences from documents with metadata.
        
        Args:
            specific_files: Optional list of specific files to process.
                          If None, processes all .txt files in data directory.
                          
        Returns:
            List of sentence dictionaries with metadata
        """
        sentences = []
        
        if specific_files is not None:
            txt_files = specific_files
        else:
            txt_files = list(Path(self.config.data_dir).glob("*.txt"))
        
        if not txt_files:
            if specific_files is not None:
                logger.info("No files specified for processing")
                return []
            else:
                raise ValueError(f"No .txt files found in {self.config.data_dir}")
        
        logger.info(f"Processing {len(txt_files)} documents")
        
        for txt_file in txt_files:
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                doc_id = generate_doc_id(content)
                doc_name = txt_file.name
                source_path = str(txt_file)
                
                doc_sentences = split_sentences(content)
                
                for sentence_id, sentence_text in enumerate(doc_sentences):
                    if sentence_text.strip():
                        token_count = count_tokens(sentence_text, self.tokenizer)
                        
                        sentence_data = {
                            "doc_id": doc_id,
                            "doc_name": doc_name,
                            "source_path": source_path,
                            "sentence_id": sentence_id,
                            "text": sentence_text,
                            "token_count": token_count
                        }
                        sentences.append(sentence_data)
                
                logger.info(f"Extracted {len(doc_sentences)} sentences from {doc_name}")
                
            except Exception as e:
                logger.error(f"Error processing {txt_file}: {e}")
                continue
        
        return sentences
    
    def run(self, specific_files: List[Path] = None) -> str:
        """Extract documents and save to output file.
        
        Args:
            specific_files: Optional list of specific files to process
            
        Returns:
            Path to the saved sentences file
        """
        logger.info("Step 1: Starting document extraction")
        
        sentences = self.extract_documents(specific_files)
        
        output_file = self.config.output_dir / "01_sentences.jsonl"
        save_jsonl(sentences, output_file)
        
        logger.info(f"Extracted {len(sentences)} sentences to {output_file}")
        
        return str(output_file)
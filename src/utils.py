import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List
import tiktoken


def get_tokenizer(tokenizer_name: str):
    """Get a tokenizer instance by name.
    
    Args:
        tokenizer_name: Name of the tokenizer (e.g., 'tiktoken:cl100k_base')
        
    Returns:
        Tokenizer instance
        
    Raises:
        ValueError: If tokenizer name is not supported
    """
    if tokenizer_name.startswith("tiktoken:"):
        encoding_name = tokenizer_name.split(":", 1)[1]
        return tiktoken.get_encoding(encoding_name)
    else:
        raise ValueError(f"Unsupported tokenizer: {tokenizer_name}")


def count_tokens(text: str, tokenizer) -> int:
    """Count the number of tokens in text using the given tokenizer.
    
    Args:
        text: Text to tokenize
        tokenizer: Tokenizer instance
        
    Returns:
        Number of tokens
    """
    return len(tokenizer.encode(text))


def normalize_text(text: str) -> str:
    """Normalize text by collapsing whitespace.
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text with single spaces
    """
    return re.sub(r'\s+', ' ', text.strip())


def generate_doc_id(text: str) -> str:
    """Generate a deterministic document ID from text content.
    
    Args:
        text: Document text content
        
    Returns:
        12-character SHA1 hash of normalized text
    """
    normalized = normalize_text(text)
    return hashlib.sha1(normalized.encode('utf-8')).hexdigest()[:12]


def generate_sentence_id(doc_id: str, sentence_idx: int) -> str:
    """Generate a unique sentence ID within a document.
    
    Args:
        doc_id: Document identifier
        sentence_idx: Sentence index within document
        
    Returns:
        Formatted sentence ID
    """
    return f"{doc_id}:{sentence_idx}"


def generate_chunk_id(doc_id: str, start_sentence_id: int, end_sentence_id: int) -> str:
    """Generate a unique chunk ID from sentence range.
    
    Args:
        doc_id: Document identifier
        start_sentence_id: Starting sentence index
        end_sentence_id: Ending sentence index
        
    Returns:
        Formatted chunk ID
    """
    return f"{doc_id}:{start_sentence_id:04d}-{end_sentence_id:04d}"


def generate_group_id(anchor_label: str) -> str:
    """Generate a deterministic group ID from anchor label.
    
    Args:
        anchor_label: Primary label for the group
        
    Returns:
        Group ID with 'cand_' prefix and 8-character hash
    """
    return "cand_" + hashlib.sha1(anchor_label.encode('utf-8')).hexdigest()[:8]


def generate_pattern_id(canonical_label: str) -> str:
    """Generate a deterministic pattern ID from canonical label.
    
    Args:
        canonical_label: Canonical topic label
        
    Returns:
        Pattern ID with 'canonical_topic_' prefix and 8-character hash
    """
    return "canonical_topic_" + hashlib.sha1(canonical_label.lower().encode('utf-8')).hexdigest()[:8]


def split_sentences(text: str) -> List[str]:
    """Split text into sentences using punctuation patterns.
    
    Args:
        text: Text to split
        
    Returns:
        List of sentence strings
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def save_jsonl(data: List[Dict[str, Any]], filepath: Path) -> None:
    """Save data as JSON Lines format.
    
    Args:
        data: List of dictionaries to save
        filepath: Path to output file
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def load_jsonl(filepath: Path) -> List[Dict[str, Any]]:
    """Load data from JSON Lines format.
    
    Args:
        filepath: Path to input file
        
    Returns:
        List of dictionaries, empty list if file doesn't exist
    """
    if not filepath.exists():
        return []
    
    items = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def save_json(data: Any, filepath: Path) -> None:
    """Save data as pretty-printed JSON.
    
    Args:
        data: Data to save
        filepath: Path to output file
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(filepath: Path) -> Any:
    """Load data from JSON file.
    
    Args:
        filepath: Path to input file
        
    Returns:
        Loaded data, or None if file doesn't exist
    """
    if not filepath.exists():
        return None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)
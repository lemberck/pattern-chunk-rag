import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from datetime import datetime

from .utils import load_jsonl, save_jsonl, load_json, save_json

logger = logging.getLogger(__name__)


class ChangeTracker:
    """Tracks file modifications to enable incremental processing"""
    
    def __init__(self, data_dir: Path, cache_dir: Path):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.manifest_file = self.cache_dir / "file_manifest.json"
        self.intermediate_dir = self.cache_dir / "intermediate"
        
        # Ensure cache directories exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.intermediate_dir.mkdir(parents=True, exist_ok=True)
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA1 hash of file content.
        
        Args:
            file_path: Path to file to hash
            
        Returns:
            SHA1 hash as hexadecimal string
        """
        hash_sha1 = hashlib.sha1()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha1.update(chunk)
        return hash_sha1.hexdigest()
    
    def _get_file_info(self, file_path: Path) -> Dict:
        """Get file metadata including hash and modification time.
        
        Args:
            file_path: Path to file to analyze
            
        Returns:
            Dictionary with file metadata
        """
        stat = file_path.stat()
        return {
            "path": str(file_path),
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "hash": self._calculate_file_hash(file_path),
            "last_processed": datetime.now().isoformat()
        }
    
    def load_manifest(self) -> Dict[str, Dict]:
        """Load the file manifest from cache.
        
        Returns:
            Dictionary mapping file paths to metadata, empty if not found
        """
        if not self.manifest_file.exists():
            return {}
        
        try:
            with open(self.manifest_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            logger.warning(f"Could not load manifest from {self.manifest_file}, starting fresh")
            return {}
    
    def save_manifest(self, manifest: Dict[str, Dict]) -> None:
        """Save the file manifest to cache.
        
        Args:
            manifest: Dictionary of file metadata to save
        """
        with open(self.manifest_file, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
    
    def detect_changes(self) -> Tuple[List[Path], List[Path], List[Path]]:
        """
        Detect changes in data directory
        
        Returns:
            Tuple of (new_files, modified_files, deleted_files)
        """
        # Get current files
        current_files = list(self.data_dir.glob("*.txt"))
        current_file_map = {str(f): f for f in current_files}
        
        # Load previous manifest
        manifest = self.load_manifest()
        
        new_files = []
        modified_files = []
        deleted_files = []
        
        # Check for new and modified files
        for file_path in current_files:
            file_str = str(file_path)
            
            if file_str not in manifest:
                # New file
                new_files.append(file_path)
                logger.info(f"New file detected: {file_path.name}")
            else:
                # Check if modified
                current_info = self._get_file_info(file_path)
                previous_info = manifest[file_str]
                
                if current_info["hash"] != previous_info.get("hash"):
                    modified_files.append(file_path)
                    logger.info(f"Modified file detected: {file_path.name}")
        
        # Check for deleted files
        for file_str in manifest:
            if file_str not in current_file_map:
                deleted_files.append(Path(file_str))
                logger.info(f"Deleted file detected: {Path(file_str).name}")
        
        return new_files, modified_files, deleted_files
    
    def get_files_to_process(self) -> List[Path]:
        """Get list of files that need processing (new or modified).
        
        Returns:
            List of file paths requiring processing
        """
        new_files, modified_files, _ = self.detect_changes()
        return new_files + modified_files
    
    def update_manifest(self, processed_files: List[Path]) -> None:
        """Update manifest with newly processed files.
        
        Args:
            processed_files: List of files that were successfully processed
        """
        manifest = self.load_manifest()
        
        for file_path in processed_files:
            file_info = self._get_file_info(file_path)
            manifest[str(file_path)] = file_info
            logger.debug(f"Updated manifest for: {file_path.name}")
        
        self.save_manifest(manifest)
    
    def reset_cache(self) -> None:
        """Remove all cache data for fresh start.
        
        Cleans cache, output, and LightRAG storage directories.
        """
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            logger.info(f"Removed cache directory: {self.cache_dir}")
        
        # Also remove output and lightrag storage directories
        output_dir = self.cache_dir.parent / "output"
        if output_dir.exists():
            shutil.rmtree(output_dir)
            logger.info(f"Removed output directory: {output_dir}")
        
        lightrag_dir = self.cache_dir.parent / "lightrag_storage"
        if lightrag_dir.exists():
            shutil.rmtree(lightrag_dir)
            logger.info(f"Removed LightRAG storage: {lightrag_dir}")
    
    def should_process(self) -> bool:
        """Check if pipeline should run (has changes or no previous run).
        
        Returns:
            True if processing is needed, False otherwise
        """
        if not self.manifest_file.exists():
            logger.info("No previous run detected, full processing required")
            return True
        
        files_to_process = self.get_files_to_process()
        if files_to_process:
            logger.info(f"Changes detected in {len(files_to_process)} files")
            return True
        else:
            logger.info("No changes detected, skipping pipeline")
            return False
    
    def get_status_summary(self) -> Dict:
        """Get summary of current status for logging.
        
        Returns:
            Dictionary with file count statistics
        """
        new_files, modified_files, deleted_files = self.detect_changes()
        total_files = len(list(self.data_dir.glob("*.txt")))
        
        return {
            "total_files": total_files,
            "new_files": len(new_files),
            "modified_files": len(modified_files),
            "deleted_files": len(deleted_files),
            "files_to_process": len(new_files) + len(modified_files)
        }
    
    def has_deleted_files(self) -> bool:
        """Check if any files were deleted.
        
        Returns:
            True if files were deleted since last run
        """
        _, _, deleted_files = self.detect_changes()
        return len(deleted_files) > 0
    
    def get_remaining_files(self) -> List[Path]:
        """Get all current files that exist (excluding deleted).
        
        Returns:
            List of all existing .txt files in data directory
        """
        return list(self.data_dir.glob("*.txt"))
    
    def clean_deleted_from_manifest(self) -> None:
        """Remove deleted files from manifest.
        
        Updates manifest to reflect current file state.
        """
        manifest = self.load_manifest()
        current_files = {str(f) for f in self.get_remaining_files()}
        
        # Remove entries for deleted files
        deleted_count = 0
        to_remove = []
        for file_path in manifest:
            if file_path not in current_files:
                to_remove.append(file_path)
                deleted_count += 1
        
        for file_path in to_remove:
            del manifest[file_path]
            logger.info(f"Removed deleted file from manifest: {Path(file_path).name}")
        
        if deleted_count > 0:
            self.save_manifest(manifest)
            logger.info(f"Cleaned {deleted_count} deleted files from manifest")
    
    def cache_intermediate_result(self, step_name: str, data: Any) -> None:
        """Cache intermediate pipeline result.
        
        Args:
            step_name: Name of the pipeline step
            data: Data to cache (list or dict)
        """
        cache_file = self.intermediate_dir / f"{step_name}.json"
        if isinstance(data, list):
            save_jsonl(data, cache_file.with_suffix('.jsonl'))
        else:
            save_json(data, cache_file)
        logger.debug(f"Cached {step_name} to {cache_file}")
    
    def load_cached_result(self, step_name: str) -> Any:
        """Load cached intermediate result.
        
        Args:
            step_name: Name of the pipeline step
            
        Returns:
            Cached data or None if not found
        """
        cache_file_json = self.intermediate_dir / f"{step_name}.json"
        cache_file_jsonl = self.intermediate_dir / f"{step_name}.jsonl"
        
        if cache_file_jsonl.exists():
            return load_jsonl(cache_file_jsonl)
        elif cache_file_json.exists():
            return load_json(cache_file_json)
        else:
            return None
    
    def has_cached_result(self, step_name: str) -> bool:
        """Check if cached result exists for step.
        
        Args:
            step_name: Name of the pipeline step
            
        Returns:
            True if cached result exists
        """
        cache_file_json = self.intermediate_dir / f"{step_name}.json"
        cache_file_jsonl = self.intermediate_dir / f"{step_name}.jsonl"
        return cache_file_json.exists() or cache_file_jsonl.exists()
    
    def remove_deleted_files_from_outputs(self, deleted_files: List[Path], output_dir: Path) -> None:
        """Remove entries for deleted files from all pipeline outputs.
        
        Args:
            deleted_files: List of deleted file paths
            output_dir: Directory containing pipeline output files
        """
        if not deleted_files:
            return
            
        deleted_file_names = {f.name for f in deleted_files}
        logger.info(f"Surgically removing {len(deleted_file_names)} deleted files from pipeline outputs")
        
        # Files to clean
        files_to_clean = [
            output_dir / "01_sentences.jsonl",
            output_dir / "02_sentence_topics.jsonl", 
            output_dir / "04_sentence_to_canonical.jsonl"
        ]
        
        # Clean JSONL files (remove lines where doc_name matches deleted files)
        for file_path in files_to_clean:
            if file_path.exists():
                self._clean_jsonl_file(file_path, deleted_file_names)
        
        # Clean candidate groups (remove groups that only have deleted files as support_docs)
        groups_file = output_dir / "03_candidate_groups.json"
        if groups_file.exists():
            self._clean_candidate_groups(groups_file, deleted_file_names)
        
        # Clean chunks (remove chunks from deleted files)
        chunks_file = output_dir / "05_chunks.json"
        if chunks_file.exists():
            self._clean_chunks_file(chunks_file, deleted_file_names)
    
    def _clean_jsonl_file(self, file_path: Path, deleted_file_names: set) -> None:
        """Remove lines from JSONL file where doc_name matches deleted files.
        
        Args:
            file_path: Path to JSONL file to clean
            deleted_file_names: Set of deleted file names to remove
        """
        lines_kept = []
        lines_removed = 0
        
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line.strip())
                    doc_name = data.get('doc_name', '')
                    if doc_name not in deleted_file_names:
                        lines_kept.append(line.strip())
                    else:
                        lines_removed += 1
        
        # Write back the cleaned lines
        with open(file_path, 'w') as f:
            for line in lines_kept:
                f.write(line + '\n')
        
        if lines_removed > 0:
            logger.info(f"Removed {lines_removed} entries from {file_path.name}")
    
    def _clean_candidate_groups(self, file_path: Path, deleted_file_names: set) -> None:
        """Remove candidate groups that only reference deleted files.
        
        Args:
            file_path: Path to candidate groups JSON file
            deleted_file_names: Set of deleted file names
        """
        with open(file_path, 'r') as f:
            groups = json.load(f)
        
        original_count = len(groups)
        
        # Filter out groups that only have deleted files as support docs
        cleaned_groups = []
        for group in groups:
            support_docs = group.get('support_docs', [])
            remaining_docs = [doc for doc in support_docs if doc not in deleted_file_names]
            
            if remaining_docs:  # Keep groups that still have non-deleted support docs
                group['support_docs'] = remaining_docs
                group['support_docs_count'] = len(remaining_docs)
                cleaned_groups.append(group)
        
        # Write back cleaned groups
        with open(file_path, 'w') as f:
            json.dump(cleaned_groups, f, indent=2)
        
        removed_count = original_count - len(cleaned_groups)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} candidate groups from {file_path.name}")
    
    def _clean_chunks_file(self, file_path: Path, deleted_file_names: set) -> None:
        """Remove chunks from deleted files.
        
        Args:
            file_path: Path to chunks JSON file
            deleted_file_names: Set of deleted file names
        """
        with open(file_path, 'r') as f:
            chunks = json.load(f)
        
        original_count = len(chunks)
        
        # Filter out chunks from deleted files
        cleaned_chunks = []
        for chunk in chunks:
            doc_name = chunk.get('metadata', {}).get('doc_name', '')
            if doc_name not in deleted_file_names:
                cleaned_chunks.append(chunk)
        
        # Write back cleaned chunks
        with open(file_path, 'w') as f:
            json.dump(cleaned_chunks, f, indent=2)
        
        removed_count = original_count - len(cleaned_chunks)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} chunks from {file_path.name}")
    
    def recalculate_canonical_patterns(self, output_dir: Path, min_support_docs: int) -> None:
        """Recalculate which patterns remain canonical after deletions.
        
        Args:
            output_dir: Directory containing pipeline outputs
            min_support_docs: Minimum number of supporting documents for canonical status
        """
        groups_file = output_dir / "03_candidate_groups.json"
        pattern_bank_file = output_dir / "04_pattern_bank.json"
        
        if not groups_file.exists():
            logger.warning("No candidate groups file found, cannot recalculate patterns")
            return
        
        # Load current groups and pattern bank
        with open(groups_file, 'r') as f:
            groups = json.load(f)
        
        # Check which groups still meet canonical threshold
        canonical_groups = [g for g in groups if g.get('support_docs_count', 0) >= min_support_docs]
        
        logger.info(f"After deletion: {len(canonical_groups)} groups still meet canonical threshold (≥{min_support_docs} docs)")
        
        # If pattern bank exists, update it
        if pattern_bank_file.exists():
            # Keep only patterns whose source groups still meet threshold
            with open(pattern_bank_file, 'r') as f:
                pattern_bank = json.load(f)
            
            canonical_group_ids = {g['group_id'] for g in canonical_groups}
            updated_patterns = []
            
            for pattern in pattern_bank:
                source_group_id = pattern.get('source_group_id')
                if source_group_id in canonical_group_ids:
                    # Update support docs to match cleaned group
                    matching_group = next(g for g in canonical_groups if g['group_id'] == source_group_id)
                    pattern['support_docs'] = matching_group['support_docs']
                    pattern['support_docs_count'] = matching_group['support_docs_count']
                    updated_patterns.append(pattern)
                else:
                    logger.info(f"Pattern '{pattern['label']}' no longer canonical (insufficient support docs)")
            
            # Write updated pattern bank
            with open(pattern_bank_file, 'w') as f:
                json.dump(updated_patterns, f, indent=2)
            
            removed_patterns = len(pattern_bank) - len(updated_patterns)
            if removed_patterns > 0:
                logger.info(f"Removed {removed_patterns} patterns that fell below threshold")
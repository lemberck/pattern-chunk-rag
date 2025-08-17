#!/usr/bin/env python3
"""
Pareto RAG Pipeline - Main Entry Point

A chunking pipeline for identifying & representing abstract multi-document patterns.
Follows the 7-step process: Extract → Label → Group → Promote → Chunk → KG → LightRAG

Usage:
    python main.py [--config .env]
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path
import warnings
import os

# Suppress all warnings
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'

from src.config import get_config
from src.change_tracker import ChangeTracker
from src.extractor import DocumentExtractor
from src.labeler_agent import TopicLabeler
from src.grouper import TopicGrouper
from src.promoter_agent import TopicPromoter
from src.chunker import SmartChunker
from src.kg_builder import KnowledgeGraphBuilder
from src.lightrag_client import LightRAGClient


def setup_logging():
    """Configure logging for the pipeline"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('pipeline.log')
        ]
    )


def main():
    """Main pipeline execution"""
    parser = argparse.ArgumentParser(description="Pareto RAG Pipeline")
    parser.add_argument("--config", default=".env", help="Path to config file")
    parser.add_argument("--reset", action="store_true", help="Reset all cache and storage, reprocess all files")
    parser.add_argument("--verbose", action="store_true", help="Verbose output showing processing details")
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = get_config()
        
        # Validate required directories
        if not Path(config.data_dir).exists():
            raise ValueError(f"Data directory not found: {config.data_dir}")
        
        # Create output directory
        config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 0: Change Detection and Cache Management
        logger.info("=" * 50)
        cache_dir = Path(".pipeline_cache")
        change_tracker = ChangeTracker(config.data_dir, cache_dir)
        
        # Handle reset flag
        if args.reset:
            logger.info("Reset flag detected - clearing all cache and storage")
            change_tracker.reset_cache()
            force_full_process = True
        else:
            force_full_process = False
        
        # Handle deletions first - use surgical removal approach
        _, _, deleted_files = change_tracker.detect_changes()
        has_deletions = len(deleted_files) > 0
        
        if has_deletions and not force_full_process:
            logger.info("Deleted files detected. Using surgical removal approach...")
            
            # Check if we have existing pipeline outputs to work with
            required_files = [
                config.output_dir / "01_sentences.jsonl",
                config.output_dir / "02_sentence_topics.jsonl", 
                config.output_dir / "03_candidate_groups.json"
            ]
            
            if all(f.exists() for f in required_files):
                # Use surgical removal approach
                logger.info("Existing pipeline outputs found. Removing deleted file entries...")
                
                # Remove deleted files from all pipeline outputs
                change_tracker.remove_deleted_files_from_outputs(deleted_files, config.output_dir)
                
                # Recalculate canonical patterns (some may fall below threshold)
                canonical_threshold = config.get_canonical_threshold(len(change_tracker.get_remaining_files()))
                change_tracker.recalculate_canonical_patterns(config.output_dir, canonical_threshold)
                
                # Clean deleted files from manifest
                change_tracker.clean_deleted_from_manifest()
                
                # Need to regenerate chunks after cleaning sentence data (Steps 5-7)
                logger.info("Surgical removal complete. Regenerating chunks and rebuilding KG...")
                
                # Must regenerate chunks since they group by canonical patterns
                run_surgical_rebuild = True
                files_to_process = change_tracker.get_files_to_process()  # Get new/modified files if any
            else:
                # Fall back to full rebuild if no existing outputs
                logger.info("No existing pipeline outputs found. Falling back to full rebuild...")
                if config.output_dir.exists():
                    shutil.rmtree(config.output_dir)
                    config.output_dir.mkdir(parents=True, exist_ok=True)
                
                lightrag_dir = Path("lightrag_storage")
                if lightrag_dir.exists():
                    shutil.rmtree(lightrag_dir)
                    logger.info("Cleared LightRAG storage due to file deletions")
                
                # Clean deleted files from manifest
                change_tracker.clean_deleted_from_manifest()
                
                # Process all remaining files
                run_surgical_rebuild = False
                files_to_process = change_tracker.get_remaining_files()
        else:
            run_surgical_rebuild = False
            
            # Check if processing is needed
            if not force_full_process and not change_tracker.should_process():
                logger.info("No changes detected. Pipeline execution skipped.")
                logger.info("Use --reset to force full reprocessing.")
                return
            
            # Get files to process
            if force_full_process:
                files_to_process = list(Path(config.data_dir).glob("*.txt"))
            else:
                files_to_process = change_tracker.get_files_to_process()
        
        # Get status summary (using already calculated values to avoid duplicate detection)
        total_files = len(list(Path(config.data_dir).glob("*.txt")))
        if args.verbose or force_full_process or has_deletions:
            logger.info(f"File status: {total_files} total files")
            if run_surgical_rebuild:
                logger.info(f"Using surgical removal (deleted files: {len(deleted_files)}, new/modified: {len(files_to_process)})")
            elif has_deletions and not run_surgical_rebuild:
                logger.info(f"Rebuilding from all {total_files} remaining files")
            elif force_full_process:
                logger.info(f"Processing all {len(files_to_process)} files (reset)")
            else:
                logger.info(f"Processing {len(files_to_process)} changed files")
        
        logger.info(f"Starting Pareto RAG Pipeline")
        logger.info(f"Data directory: {config.data_dir}")
        logger.info(f"Output directory: {config.output_dir}")
        
        if run_surgical_rebuild:
            # For surgical rebuild, run Steps 5-7 (Chunking + KG + LightRAG)
            # But first process any new/modified files if they exist
            if files_to_process:
                logger.info("Processing new/modified files first...")
                
                # Step 1: Document Extraction (for new/modified files only)
                logger.info("=" * 50)
                extractor = DocumentExtractor(config)
                sentences_file = extractor.run(files_to_process)
                
                # Step 2: Topic Labeling (Agent 1)
                logger.info("=" * 50)
                labeler = TopicLabeler(config)
                topics_file = labeler.run(sentences_file)
                
                # Step 3: Similarity Grouping
                logger.info("=" * 50)
                grouper = TopicGrouper(config)
                groups_file = grouper.run(topics_file)
                
                # Step 4: Canonical Topic Promotion (Agent 2)
                logger.info("=" * 50)
                promoter = TopicPromoter(config)
                pattern_bank_file, mapping_file = promoter.run(groups_file, topics_file)
                
                # Step 5: Smart Chunking
                logger.info("=" * 50)
                chunker = SmartChunker(config)
                chunks_file = chunker.run(sentences_file, mapping_file)
            else:
                # No new files, but still need to regenerate chunks from cleaned data
                pattern_bank_file = config.output_dir / "04_pattern_bank.json"
                mapping_file = config.output_dir / "04_sentence_to_canonical.jsonl"
                sentences_file = config.output_dir / "01_sentences.jsonl"
                
                # Step 5: Smart Chunking (regenerate from cleaned data)
                logger.info("=" * 50)
                chunker = SmartChunker(config)
                chunks_file = chunker.run(sentences_file, mapping_file)
            
            # Clear LightRAG storage for clean rebuild
            lightrag_dir = Path("lightrag_storage")
            if lightrag_dir.exists():
                shutil.rmtree(lightrag_dir)
                logger.info("Cleared LightRAG storage for rebuild")
        else:
            # Full pipeline processing
            # Step 1: Document Extraction
            logger.info("=" * 50)
            extractor = DocumentExtractor(config)
            sentences_file = extractor.run(files_to_process)
            
            # Step 2: Topic Labeling (Agent 1)
            logger.info("=" * 50)
            labeler = TopicLabeler(config)
            topics_file = labeler.run(sentences_file)
            
            # Step 3: Similarity Grouping
            logger.info("=" * 50)
            grouper = TopicGrouper(config)
            groups_file = grouper.run(topics_file)
            
            # Step 4: Canonical Topic Promotion (Agent 2)
            logger.info("=" * 50)
            promoter = TopicPromoter(config)
            pattern_bank_file, mapping_file = promoter.run(groups_file, topics_file)
            
            # Step 5: Smart Chunking
            logger.info("=" * 50)
            chunker = SmartChunker(config)
            chunks_file = chunker.run(sentences_file, mapping_file)
        
        # Step 6: Knowledge Graph Construction (always run)
        logger.info("=" * 50)
        kg_builder = KnowledgeGraphBuilder(config)
        nodes_file, edges_file = kg_builder.run(pattern_bank_file, chunks_file)
        
        # Step 7: LightRAG Upsert (always run)
        logger.info("=" * 50)
        lightrag_client = LightRAGClient(config)
        manifest_file = lightrag_client.run(chunks_file, pattern_bank_file, edges_file)
        
        # Update manifest with processed files (ensure cache directory exists)
        change_tracker.cache_dir.mkdir(parents=True, exist_ok=True)
        if not run_surgical_rebuild or files_to_process:
            if force_full_process:
                # For reset runs, track all current .txt files to ensure deletions are detected
                all_current_files = list(Path(config.data_dir).glob("*.txt"))
                change_tracker.update_manifest(all_current_files)
            else:
                change_tracker.update_manifest(files_to_process)
        
        # Pipeline Complete
        logger.info("=" * 50)
        logger.info("Pipeline completed successfully!")
        logger.info(f"Generated artifacts in: {config.output_dir}")
        
        # Summary
        logger.info("\nGenerated files:")
        for file_path in sorted(config.output_dir.glob("*")):
            logger.info(f"  {file_path.name}")
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

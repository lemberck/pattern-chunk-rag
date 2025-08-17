#!/usr/bin/env python3
"""
Simple test to show LightRAG integration Q&A
"""
import asyncio
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.lightrag_client import LightRAGClient

# Set up logging to only show errors
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


async def main():
    """Test LightRAG integration and show Q&A results"""
    config = get_config()
    client = LightRAGClient(config)
    
    print("=" * 58)
    print("🎯 LIGHTRAG KNOWLEDGE GRAPH Q&A DEMO 🎯")
    print("=" * 58)
    
    try:
        await client.initialize()
        
        # Test queries to demonstrate our canonical patterns in action
        test_queries = [
            "What are the main patterns causing user abandonment during verification?",
            "How does verification timing impact user activation and retention?",
            "Explain the benefits of deferred verification strategies",
            "What are the key SMS delivery issues affecting verification?",
            "How can we optimize verification timing for better conversion rates?",
            "What is the relationship between verification delays and user experience?",
            "How do different user segments respond to verification timing changes?",
            "What solutions exist for reducing verification abandonment on mobile?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{'='*80}")
            print(f"❓ QUESTION {i}: {query}")
            print('='*80)
            
            try:
                start_time = time.time()
                result = await client.query(query, mode="hybrid")
                end_time = time.time()
                query_time = end_time - start_time
                
                print(f"💡 ANSWER ({query_time:.2f}s):\n{result}")
                    
            except Exception as e:
                print(f"❌ ERROR: {e}")
        
        print("🏆 LightRAG Knowledge Graph Demo Complete! 🏆")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
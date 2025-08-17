"""
All LLM prompts used in the RAG pipeline.

This module centralizes all prompts to enable easy modification without 
touching the core pipeline logic.
"""

# Agent 1: Topic Extraction Prompt
TOPIC_EXTRACTION_PROMPT = """You are a topic extraction specialist. Your task is to analyze a single sentence and extract exactly ONE candidate topic from it.

Rules:
1. Output exactly one topic in snake_case format (e.g., "activation_friction_step2")
2. Keep topics concise but descriptive (2-4 words max)
3. Focus on the main concept, problem, or action in the sentence
4. Include a brief description of the topic (1 sentence max)
5. Output format: topic_name :: description

Sentence to analyze: "{sentence}"

Output the topic and description:"""

# Agent 2: Canonical Topic Promotion Prompt
CANONICAL_PROMOTION_PROMPT = """You are a canonical topic naming specialist. Your task is to analyze a group of similar candidate topics and their respective descriptions and create one canonical label with a clear description.

Given similar labels: {labels}
Descriptions: {descriptions}

Rules:
1. Create ONE canonical label in snake_case (e.g., "activation_friction_verification")
2. The label should capture the common essence of all input labels
3. Write a 1-2 sentence description that explains the pattern without inventing details
4. Focus on what the labels actually represent, not what you think they might mean
5. Output format: canonical_label :: description

Output the canonical label and description:"""
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict


class Config(BaseSettings):
    """Configuration settings for the Pareto RAG pipeline.
    
    Manages all configurable parameters including tokenization, similarity thresholds,
    chunking parameters, and LLM model settings.
    """
    data_dir: Path = Field(default="./data", description="Directory containing .txt files")
    tokenizer: str = Field(default="tiktoken:cl100k_base", description="Tokenizer to use")
    label_sim_threshold: float = Field(default=0.6, description="Similarity threshold for grouping labels")
    canonical_support_fraction: float = Field(default=0.10, description="Fraction of docs needed for canonical status")
    canonical_min_docs: int = Field(default=2, description="Minimum docs for canonical status")
    chunk_min_tokens: int = Field(default=180, description="Minimum tokens per chunk")
    chunk_target_tokens: int = Field(default=400, description="Target tokens per chunk")
    chunk_max_tokens: int = Field(default=500, description="Maximum tokens per chunk")
    chunk_overlap_sents: int = Field(default=1, description="Sentence overlap between chunks")
    include_non_pattern_chunks: bool = Field(default=False, description="Include chunks for documents without canonical patterns")
    lightrag_endpoint: str = Field(default="http://localhost:8020", description="LightRAG endpoint URL")
    lightrag_index: str = Field(default="canonical-topics-demo", description="LightRAG index name")
    lightrag_working_dir: str = Field(default="./lightrag_storage", description="LightRAG working directory")
    lightrag_max_async: int = Field(default=4, description="Maximum async operations for LightRAG")
    lightrag_cache_size: int = Field(default=1000, description="LightRAG cache size")
    openai_api_key: str = Field(description="OpenAI API key")
    embedding_model: str = Field(default="text-embedding-3-small", description="OpenAI embedding model")
    chat_model: str = Field(default="gpt-4o-mini", description="OpenAI chat model for agents")
    llm_model: str = Field(default="gpt-4o-mini", description="LLM model for LightRAG")
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )
    
    @property
    def output_dir(self) -> Path:
        """Get the output directory for pipeline artifacts.
        
        Returns:
            Path to the output directory
        """
        return Path("./output")
    
    def get_canonical_threshold(self, total_docs: int) -> int:
        """Calculate the minimum document count for canonical topic promotion.
        
        Args:
            total_docs: Total number of documents in the dataset
            
        Returns:
            Minimum number of documents required for canonical status
        """
        return max(
            int(self.canonical_support_fraction * total_docs),
            self.canonical_min_docs
        )


def get_config() -> Config:
    """Create and return a configuration instance.
    
    Returns:
        Configured Config instance with environment variables loaded
    """
    return Config()
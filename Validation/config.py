"""
Configuration file for validation system.

Store your API keys and system settings here.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Configuration for validation system."""
    
    # API Keys
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "AIzaSyCsmuWPmGN2lwYtsyA3IluDl2OgsJ1dJPY")
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "pcsk_37ezP5_RxhCU8Ce2V1VUZHMTbs1TSxdYqPPMwLq7AKSDwChbmRhtocWVr1DQ7uq5eqQ7hb")
    
    REAL_DATA_PATH: str = "../data/raw/creditcard.csv"
    SYNTHETIC_DATA_PATH: str = "../data/synthetic/day3_synthetic_1k.csv"
    OUTPUT_DIR: str = "../reports/validation"
    
    # Layer 1: Rule Validator
    LAYER1_ENABLE_GEMINI_RULES: bool = True
    LAYER1_MAX_GEMINI_RULES: int = 10
    LAYER1_GEMINI_MODEL: str = "gemini-2.0-flash-exp"
    
    # Layer 2: Statistical Validator
    LAYER2_KS_THRESHOLD: float = 0.05
    LAYER2_CORRELATION_THRESHOLD: float = 0.1
    LAYER2_OUTLIER_ZSCORE_THRESHOLD: float = 3.0
    
    # Layer 3: Semantic Validator
    LAYER3_SAMPLE_SIZE: int = 1000
    LAYER3_GEMINI_MODEL: str = "gemini-2.0-flash-exp"
    LAYER3_TEMPERATURE: float = 0.0
    LAYER3_BATCH_SIZE: int = 10
    
    # Layer 4: RAG Validator
    LAYER4_EMBEDDING_MODEL: str = "models/text-embedding-004"
    LAYER4_PINECONE_INDEX: str = "synthetic-validation"
    LAYER4_TOP_K_NEIGHBORS: int = 10
    LAYER4_DISTANCE_THRESHOLD: str = "auto"
    
    # Meta Validator
    META_CONSISTENCY_RUNS: int = 3
    META_AGREEMENT_THRESHOLD: float = 0.85
    
    # Performance
    BATCH_SIZE: int = 10000
    PARALLEL_WORKERS: int = 2
    
    # Success criteria
    TARGET_PASS_RATE: float = 0.90
    TARGET_DETECTION_RATE: float = 0.98
    TARGET_FALSE_POSITIVE_RATE: float = 0.05


# Global config instance
config = Config()


def load_config_from_env():
    """
    Load configuration from environment variables.
    
    Set environment variables before running:
        export GEMINI_API_KEY="your-key"
        export PINECONE_API_KEY="your-key"
    """
    if os.getenv("GEMINI_API_KEY"):
        config.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    if os.getenv("PINECONE_API_KEY"):
        config.PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    
    if os.getenv("REAL_DATA_PATH"):
        config.REAL_DATA_PATH = os.getenv("REAL_DATA_PATH")
    
    if os.getenv("SYNTHETIC_DATA_PATH"):
        config.SYNTHETIC_DATA_PATH = os.getenv("SYNTHETIC_DATA_PATH")
    
    return config


def validate_config():
    """
    Validate that required configuration is set.
    
    Returns:
        bool: True if config is valid, False otherwise
    """
    errors = []
    
    # Check API keys
    if "your-gemini-api-key-here" in config.GEMINI_API_KEY:
        errors.append("GEMINI_API_KEY not set")
    
    # Check file paths
    if not os.path.exists(config.REAL_DATA_PATH):
        errors.append(f"Real data file not found: {config.REAL_DATA_PATH}")
    
    if not os.path.exists(config.SYNTHETIC_DATA_PATH):
        errors.append(f"Synthetic data file not found: {config.SYNTHETIC_DATA_PATH}")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)
        print(f"Created output directory: {config.OUTPUT_DIR}")
    
    if errors:
        print("\nConfiguration errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print("\nConfiguration validated successfully!")
    print(f"  Gemini API key: {config.GEMINI_API_KEY[:20]}...")
    print(f"  Real data: {config.REAL_DATA_PATH}")
    print(f"  Synthetic data: {config.SYNTHETIC_DATA_PATH}")
    print(f"  Output directory: {config.OUTPUT_DIR}")
    
    return True


if __name__ == "__main__":
    """Test configuration loading."""
    print("="*60)
    print("Configuration Test")
    print("="*60)
    
    load_config_from_env()
    
    print("\nCurrent configuration:")
    print(f"  Gemini API Key: {config.GEMINI_API_KEY[:20] if len(config.GEMINI_API_KEY) > 20 else 'NOT SET'}...")
    print(f"  Real data path: {config.REAL_DATA_PATH}")
    print(f"  Synthetic data path: {config.SYNTHETIC_DATA_PATH}")
    print(f"  Output directory: {config.OUTPUT_DIR}")
    print(f"  Layer 1 Gemini model: {config.LAYER1_GEMINI_MODEL}")
    print(f"  Layer 2 KS threshold: {config.LAYER2_KS_THRESHOLD}")
    print(f"  Layer 3 sample size: {config.LAYER3_SAMPLE_SIZE}")
    
    print("\n" + "="*60)
    validate_config()
    print("="*60)

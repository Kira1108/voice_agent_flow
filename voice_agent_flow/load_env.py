"""Configuration management for AI Toolkits."""
import os
from pathlib import Path
from functools import lru_cache
from dotenv import load_dotenv

@lru_cache(maxsize=1)
def load_environment(env_path: str = None) -> bool:
    """Load environment variables from .env file (cached to run only once)."""
    if env_path is None:
        env_path = Path.home() / ".env"
    load_dotenv(str(env_path))
    return True 

def get_env_var(key: str, default: str = None) -> str:
    """Get environment variable with optional default."""
    load_environment()  # Ensures env is loaded before accessing
    return os.getenv(key, default)

def get_required_env_var(key: str) -> str:
    """Get required environment variable, raise error if not found."""
    load_environment()
    value = os.getenv(key)
    if value is None:
        raise ValueError(f"Required environment variable '{key}' not found")
    return value
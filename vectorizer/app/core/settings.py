from os import environ
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parents[3]
load_dotenv(project_root / ".env")
load_dotenv(project_root / ".env.local", override=True)


class Config:
    OPENAI_API_KEY: str = environ.get("OPENAI_API_KEY")
    OPENAI_BASE_URL: str = environ.get("OPENAI_BASE_URL", "")

    # Separate embedding API configuration
    EMBEDDING_API_KEY: str = environ.get("EMBEDDING_API_KEY", environ.get("OPENAI_API_KEY"))
    EMBEDDING_BASE_URL: str = environ.get("EMBEDDING_BASE_URL", "https://api.openai.com/v1")
    EMBEDDING_MODEL: str = environ.get("EMBEDDING_MODEL", "text-embedding-3-small")

    # Local embedding configuration
    USE_LOCAL_EMBEDDINGS: bool = environ.get("USE_LOCAL_EMBEDDINGS", "false").lower() == "true"
    LOCAL_EMBEDDING_MODEL: str = environ.get("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    DATABASE_HOST: str = environ.get("DATABASE_HOST", "localhost")
    DATABASE_PORT: int = int(environ.get("DATABASE_PORT", "5432"))
    DATABASE_NAME: str = environ.get("DATABASE_NAME", "multi_agent")
    DATABASE_USER: str = environ.get("DATABASE_USER", "postgres")
    DATABASE_PASSWORD: str = environ.get("DATABASE_PASSWORD", "")
    QDRANT_URL: str = environ.get("QDRANT_URL", "http://localhost:6333")
    QDRANT_KEY: str = environ.get("QDRANT_KEY", "")
    RECREATE_COLLECTIONS: bool = environ.get("RECREATE_COLLECTIONS", "False")


def get_settings():
    return Config()

import os
from dataclasses import dataclass


@dataclass
class Config:
    """Configuration via environment variables"""

    # Server
    HOST: str = os.getenv("API_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("API_PORT", "8000"))

    # Worker pool
    MAX_CONCURRENT_JOBS: int = int(os.getenv("MAX_CONCURRENT_JOBS", "4"))
    MAX_QUEUE_SIZE: int = int(os.getenv("MAX_QUEUE_SIZE", "50"))

    # Storage paths
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "/app/uploads")
    RESULTS_DIR: str = os.getenv("RESULTS_DIR", "/app/results")
    DATA_DIR: str = os.getenv("DATA_DIR", "/app/data")

    # Database
    DB_PATH: str = os.getenv("DB_PATH", "/app/data/jobs.db")

    # TTL
    TTL_HOURS: int = int(os.getenv("TTL_HOURS", "24"))


config = Config()

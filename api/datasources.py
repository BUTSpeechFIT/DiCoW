import os
import glob
from pathlib import Path
from fastapi import UploadFile
from typing import List

from .config import config
from .logger import setup_logger

logger = setup_logger(__name__)


class HTTPUploadSource:
    """Handle file uploads via HTTP multipart/form-data"""

    async def save_uploaded_file(self, file: UploadFile, job_id: str) -> str:
        """
        Save uploaded file with job_id prefix to avoid collisions.

        Args:
            file: Uploaded file from FastAPI
            job_id: UUID of the job (used as prefix)

        Returns:
            Path to saved file
        """
        os.makedirs(config.UPLOAD_DIR, exist_ok=True)

        safe_filename = f"{job_id}_{file.filename}"
        file_path = os.path.join(config.UPLOAD_DIR, safe_filename)

        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        logger.debug(f"Saved uploaded file to {file_path}")
        return file_path

    async def cleanup_uploaded_file(self, file_path: str) -> None:
        """Delete uploaded file"""
        try:
            os.remove(file_path)
            logger.debug(f"Deleted uploaded file: {file_path}")
        except FileNotFoundError:
            logger.debug(f"Uploaded file not found: {file_path}")
        except Exception as e:
            logger.error(f"Error deleting uploaded file {file_path}: {e}")


class LocalVolumeSource:
    """Handle files from mounted volume"""

    async def get_files(
        self, folder_path: str, file_pattern: str = "*.wav"
    ) -> List[str]:
        """Get all matching files from folder"""
        search_pattern = os.path.join(folder_path, file_pattern)
        files = glob.glob(search_pattern)
        logger.debug(
            f"Found {len(files)} files matching {file_pattern} in {folder_path}"
        )
        return sorted(files)

    async def validate_folder_access(self, folder_path: str) -> bool:
        """Check if folder exists and is accessible"""
        exists = os.path.exists(folder_path) and os.path.isdir(folder_path)
        if not exists:
            logger.warning(f"Folder not accessible: {folder_path}")
        return exists

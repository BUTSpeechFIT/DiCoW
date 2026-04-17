import asyncio
import os
from .store import JobStore
from .config import config
from .logger import setup_logger

logger = setup_logger(__name__)


async def cleanup_task(store: JobStore):
    """
    Background task to cleanup expired jobs.
    Runs every hour, deletes jobs past TTL.
    """
    while True:
        try:
            await asyncio.sleep(3600)
            logger.info("Running TTL cleanup...")

            expired_jobs = await store.cleanup_expired_jobs()

            if not expired_jobs:
                logger.debug("No expired jobs to cleanup")
                continue

            logger.info(f"Cleaning up {len(expired_jobs)} expired jobs")

            for job in expired_jobs:
                job_id = job["id"]

                result_path = job.get("result_path")
                if result_path and os.path.exists(result_path):
                    try:
                        os.remove(result_path)
                        logger.debug(f"Deleted result file: {result_path}")
                    except FileNotFoundError:
                        logger.debug(f"Result file not found: {result_path}")
                    except Exception as e:
                        logger.error(f"Error deleting result file {result_path}: {e}")

                upload_path = job.get("upload_path")
                if upload_path and os.path.exists(upload_path):
                    try:
                        os.remove(upload_path)
                        logger.debug(f"Deleted uploaded file: {upload_path}")
                    except FileNotFoundError:
                        logger.debug(f"Uploaded file not found: {upload_path}")
                    except Exception as e:
                        logger.error(f"Error deleting uploaded file {upload_path}: {e}")

            logger.info(f"TTL cleanup complete: {len(expired_jobs)} jobs removed")

        except asyncio.CancelledError:
            logger.info("Cleanup task cancelled")
            break
        except Exception as e:
            logger.error(f"Cleanup task error: {e}", exc_info=True)

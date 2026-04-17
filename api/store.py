import aiosqlite
import json
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any

from .config import config
from .logger import setup_logger

logger = setup_logger(__name__)


class JobStore:
    def __init__(self, db_path: str):
        self.db_path = db_path

    async def init_db(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT,
                    input_json TEXT NOT NULL,
                    result_path TEXT,
                    error TEXT,
                    retry_count INTEGER DEFAULT 0,
                    ttl_expires_at TEXT NOT NULL,
                    upload_path TEXT
                )
            """)
            await db.commit()
        logger.info(f"Database initialized at {self.db_path}")

    async def create_job(
        self,
        job_id: str,
        input_data: Dict[str, Any],
        ttl_hours: int,
        upload_path: Optional[str] = None,
    ) -> None:
        now = datetime.utcnow().replace(tzinfo=timezone.utc)
        ttl_expires = now + timedelta(hours=ttl_hours)

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT INTO jobs 
                   (id, status, created_at, updated_at, input_json, ttl_expires_at, upload_path)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    job_id,
                    "pending",
                    now.isoformat(),
                    now.isoformat(),
                    json.dumps(input_data),
                    ttl_expires.isoformat(),
                    upload_path,
                ),
            )
            await db.commit()
        logger.debug(f"Job {job_id} created in database")

    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM jobs WHERE id = ?", (job_id,)
            ) as cursor:
                row = await cursor.fetchone()
                return dict(row) if row else None

    async def update_job_status(
        self,
        job_id: str,
        status: str,
        error: Optional[str] = None,
        result_path: Optional[str] = None,
        increment_retry: bool = False,
    ) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            now = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()

            if increment_retry:
                await db.execute(
                    """UPDATE jobs 
                       SET status = ?, updated_at = ?, error = ?, result_path = ?,
                           retry_count = retry_count + 1
                       WHERE id = ?""",
                    (status, now, error, result_path, job_id),
                )
            else:
                fields = ["status = ?", "updated_at = ?"]
                values = [status, now]

                if error is not None:
                    fields.append("error = ?")
                    values.append(error)
                if result_path is not None:
                    fields.append("result_path = ?")
                    values.append(result_path)
                if status == "completed":
                    fields.append("completed_at = ?")
                    values.append(now)

                values.append(job_id)

                await db.execute(
                    f"UPDATE jobs SET {', '.join(fields)} WHERE id = ?", values
                )

            await db.commit()
        logger.debug(f"Job {job_id} status updated to {status}")

    async def get_jobs_by_status(self, status: str) -> List[Dict[str, Any]]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM jobs WHERE status = ?", (status,)
            ) as cursor:
                return [dict(row) for row in await cursor.fetchall()]

    async def get_all_jobs(self) -> List[Dict[str, Any]]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM jobs") as cursor:
                return [dict(row) for row in await cursor.fetchall()]

    async def delete_job(self, job_id: str) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            await db.commit()
        logger.debug(f"Job {job_id} deleted from database")

    async def cleanup_expired_jobs(self) -> List[Dict[str, Any]]:
        """
        Delete jobs past TTL, return full job data for cleanup.
        Returns full job dicts so caller can cleanup files.
        """
        now = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            async with db.execute(
                "SELECT * FROM jobs WHERE ttl_expires_at < ?", (now,)
            ) as cursor:
                expired = await cursor.fetchall()

            expired_jobs = [dict(job) for job in expired]
            expired_ids = [job["id"] for job in expired_jobs]

            for job_id in expired_ids:
                await db.execute("DELETE FROM jobs WHERE id = ?", (job_id,))

            await db.commit()

        logger.info(f"Cleaned up {len(expired_jobs)} expired jobs from database")
        return expired_jobs

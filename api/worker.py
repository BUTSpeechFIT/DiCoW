import asyncio
import uuid
import json
import os
import re
import glob
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from librosa import get_duration as librosa_get_duration

from .store import JobStore
from .config import config
from .logger import setup_logger

logger = setup_logger(__name__)


class Worker:
    def __init__(self, store: JobStore, dicow_pipeline):
        self.store = store
        self.pipeline = dicow_pipeline
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=config.MAX_QUEUE_SIZE)
        self.semaphore: asyncio.Semaphore = asyncio.Semaphore(
            config.MAX_CONCURRENT_JOBS
        )
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the worker background task"""
        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.info("Worker started")

    async def stop(self):
        """Stop worker gracefully"""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("Worker stopped")

    async def submit_job(self, job_id: str) -> None:
        """Add job to queue"""
        await self.queue.put(job_id)
        logger.debug(f"Job {job_id} submitted to queue")

    async def _worker_loop(self):
        """Main worker loop - processes jobs from queue"""
        while self._running:
            try:
                job_id = await asyncio.wait_for(self.queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            async with self.semaphore:
                try:
                    await self._process_job(job_id)
                except asyncio.CancelledError:
                    logger.warning(f"Job {job_id} cancelled")
                    break
                except Exception as e:
                    logger.error(
                        f"Job {job_id} failed catastrophically: {e}", exc_info=True
                    )
                    await self.store.update_job_status(job_id, "failed", error=str(e))
                finally:
                    self.queue.task_done()

    async def _process_job(self, job_id: str):
        """Process a single job"""
        logger.info(f"Processing job {job_id}")

        await self.store.update_job_status(job_id, "processing")

        job_data = await self.store.get_job(job_id)
        if not job_data:
            logger.error(f"Job {job_id} not found in database")
            return

        input_data = json.loads(job_data["input_json"])

        def run_pipeline():
            return self._run_pipeline_sync(input_data, job_id)

        try:
            result = await asyncio.to_thread(run_pipeline)

            result_path = os.path.join(config.RESULTS_DIR, f"{job_id}.json")
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            await self.store.update_job_status(
                job_id, "completed", result_path=result_path
            )

            logger.info(f"Job {job_id} completed successfully")

        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}", exc_info=True)

            retry_count = job_data.get("retry_count", 0) + 1
            if retry_count <= 2:
                logger.info(f"Retrying job {job_id} (attempt {retry_count}/2)")
                await self.store.update_job_status(
                    job_id, "pending", increment_retry=True
                )
                await self.queue.put(job_id)
            else:
                logger.error(f"Job {job_id} exceeded max retries, marking as FAILED")
                await self.store.update_job_status(job_id, "failed", error=str(e))

    def _run_pipeline_sync(
        self, input_data: Dict[str, Any], job_id: str
    ) -> Dict[str, Any]:
        """
        Run the DiCoW pipeline synchronously (called from thread).
        Returns ONLY structured segments, no formatted text.
        """
        start_time = datetime.utcnow()

        if input_data["mode"] == "single_file":
            audio_path = input_data["file_path"]

            result = self.pipeline(audio_path, return_timestamps=True)

            duration_seconds = librosa_get_duration(filename=audio_path)

            segments = self._parse_segments(result["text"])

            speakers = set(seg["speaker"] for seg in segments)

            completed_at = datetime.utcnow()
            processing_time = (completed_at - start_time).total_seconds()

            return {
                "job_id": job_id,
                "status": "completed",
                "created_at": start_time.isoformat(),
                "completed_at": completed_at.isoformat(),
                "input_info": {
                    "mode": "single_file",
                    "filename": os.path.basename(audio_path),
                    "size_bytes": os.path.getsize(audio_path),
                },
                "result": {
                    "segments": segments,
                    "speakers_count": len(speakers),
                    "duration_seconds": duration_seconds,
                    "metadata": {
                        "model": "BUT-FIT/SE-DiCoW",
                        "diarization_model": "BUT-FIT/diarizen-wavlm-large-s80-md",
                        "processing_time_seconds": processing_time,
                    },
                },
            }

        elif input_data["mode"] == "batch_folder":
            folder_path = input_data["folder_path"]
            file_pattern = input_data["file_pattern"]

            files = self._get_wav_files(folder_path, file_pattern)

            if not files:
                raise ValueError(
                    f"No WAV files found matching '{file_pattern}' in '{folder_path}'"
                )

            all_segments = []
            total_duration = 0.0
            all_speakers = set()

            for audio_path in files:
                logger.info(f"Processing batch file: {audio_path}")

                result = self.pipeline(audio_path, return_timestamps=True)
                duration = librosa_get_duration(filename=audio_path)
                total_duration += duration

                segments = self._parse_segments(result["text"])

                for seg in segments:
                    seg["filename"] = os.path.basename(audio_path)

                all_segments.extend(segments)
                all_speakers.update(seg["speaker"] for seg in segments)

            completed_at = datetime.utcnow()
            processing_time = (completed_at - start_time).total_seconds()

            return {
                "job_id": job_id,
                "status": "completed",
                "created_at": start_time.isoformat(),
                "completed_at": completed_at.isoformat(),
                "input_info": {
                    "mode": "batch_folder",
                    "folder_path": folder_path,
                    "file_pattern": file_pattern,
                    "files_count": len(files),
                },
                "result": {
                    "segments": all_segments,
                    "speakers_count": len(all_speakers),
                    "duration_seconds": total_duration,
                    "metadata": {
                        "model": "BUT-FIT/SE-DiCoW",
                        "diarization_model": "BUT-FIT/diarizen-wavlm-large-s80-md",
                        "files_processed": len(files),
                        "processing_time_seconds": processing_time,
                    },
                },
            }

        else:
            raise ValueError(f"Unknown mode: {input_data.get('mode')}")

    def _get_wav_files(self, input_folder: str, file_pattern: str) -> List[str]:
        """Find all WAV files in the input folder matching the pattern."""
        search_pattern = os.path.join(input_folder, file_pattern)
        wav_files = glob.glob(search_pattern)
        return sorted(wav_files)

    def _parse_segments(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse formatted text to extract structured segments.

        Input example:
        "🗣️ Speaker 0:\n<|0.00|>Hello<|2.50|> world<|5.00|>\n\n🗣️ Speaker 1:\n<|3.00|>Hi<|6.00|>"

        Output:
        [
            {"speaker": 0, "start": 0.00, "end": 2.50, "text": "Hello"},
            {"speaker": 0, "start": 2.50, "end": 5.00, "text": "world"},
            {"speaker": 1, "start": 3.00, "end": 6.00, "text": "Hi"}
        ]
        """
        segments = []

        speaker_pattern = r"🗣️ Speaker (\d+):"
        speaker_sections = re.split(speaker_pattern, text)

        for i in range(1, len(speaker_sections), 2):
            speaker_id = int(speaker_sections[i])
            content = speaker_sections[i + 1] if i + 1 < len(speaker_sections) else ""

            segment_pattern = r"<\|(\d+\.\d+)\|>(.+?)<\|(\d+\.\d+)\|>"
            matches = re.findall(segment_pattern, content, re.DOTALL)

            for start_str, text_content, end_str in matches:
                text_content = " ".join(text_content.split())

                if text_content.strip():
                    segments.append(
                        {
                            "speaker": speaker_id,
                            "start": float(start_str),
                            "end": float(end_str),
                            "text": text_content,
                        }
                    )

        return segments

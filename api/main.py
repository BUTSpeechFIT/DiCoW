from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uuid
import os
import json
import asyncio
from datetime import datetime, timezone
from typing import List, Optional

from .config import config
from .store import JobStore
from .worker import Worker
from .datasources import HTTPUploadSource, LocalVolumeSource
from .cleanup import cleanup_task
from .models import (
    JobResponse,
    JobDetailResponse,
    JobResultResponse,
    JobStatus,
    TranscribeMode,
    TranscriptionResult,
    Segment,
)
from .logger import setup_logger

logger = setup_logger(__name__)

store: Optional[JobStore] = None
worker: Optional[Worker] = None
http_source: Optional[HTTPUploadSource] = None
local_source: Optional[LocalVolumeSource] = None
cleanup_bg_task: Optional[asyncio.Task] = None
dicow_pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle - startup and shutdown"""
    global store, worker, http_source, local_source, cleanup_bg_task, dicow_pipeline

    logger.info("Starting DiCoW API...")

    store = JobStore(config.DB_PATH)
    await store.init_db()

    http_source = HTTPUploadSource()
    local_source = LocalVolumeSource()

    logger.info("Loading DiCoW models...")

    from pipeline import DiCoWPipeline
    from transformers import (
        AutoTokenizer,
        AutoFeatureExtractor,
        AutoModelForSpeechSeq2Seq,
    )
    from diarizen.pipelines.inference import DiariZenPipeline
    import torch

    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    logger.info(f"Using device: {device}")

    dicow = AutoModelForSpeechSeq2Seq.from_pretrained(
        "BUT-FIT/SE-DiCoW", trust_remote_code=True, local_files_only=False
    )
    feature_extractor = AutoFeatureExtractor.from_pretrained("BUT-FIT/SE-DiCoW")
    tokenizer = AutoTokenizer.from_pretrained("BUT-FIT/SE-DiCoW")

    vocab = tokenizer.get_vocab()
    tokenizer.upper_cased_tokens = {}
    for token, index in vocab.items():
        if len(token) < 1:
            continue
        if token[0] == "Ġ" and len(token) > 1:
            lower_cased_token = (
                token[0] + token[1].lower() + (token[2:] if len(token) > 2 else "")
            )
        else:
            lower_cased_token = token[0].lower() + token[1:]
        if lower_cased_token != token:
            lower_index = vocab.get(lower_cased_token, None)
            if lower_index is not None:
                tokenizer.upper_cased_tokens[lower_index] = index

    dicow.set_tokenizer(tokenizer)
    dicow.config.model_type = "whisper"

    diar_pipeline = DiariZenPipeline.from_pretrained(
        "BUT-FIT/diarizen-wavlm-large-s80-md"
    ).to(device)
    diar_pipeline.embedding_batch_size = 16
    diar_pipeline.segmentation_batch_size = 16

    dicow_pipeline = DiCoWPipeline(
        dicow,
        diarization_pipeline=diar_pipeline,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        device=device,
    )

    logger.info("Models loaded successfully")

    worker = Worker(store, dicow_pipeline)
    await worker.start()

    pending_jobs = await store.get_jobs_by_status("pending")
    processing_jobs = await store.get_jobs_by_status("processing")

    logger.info(f"Recovery: {len(pending_jobs)} PENDING jobs to re-queue")
    logger.info(f"Recovery: {len(processing_jobs)} PROCESSING jobs to retry")

    for job in pending_jobs:
        await worker.submit_job(job["id"])

    for job in processing_jobs:
        retry_count = job.get("retry_count", 0) + 1
        if retry_count <= 2:
            logger.info(f"Retrying job {job['id']} (attempt {retry_count}/2)")
            await store.update_job_status(job["id"], "pending", increment_retry=True)
            await worker.submit_job(job["id"])
        else:
            logger.warning(f"Job {job['id']} exceeded max retries, marking as FAILED")
            await store.update_job_status(
                job["id"], "failed", error="Max retries exceeded"
            )

    loop = asyncio.get_event_loop()
    cleanup_bg_task = loop.create_task(cleanup_task(store))

    os.makedirs(config.UPLOAD_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.DATA_DIR, exist_ok=True)

    logger.info("DiCoW API started successfully")

    yield

    logger.info("Shutting down DiCoW API...")

    if worker:
        await worker.stop()

    if cleanup_bg_task:
        cleanup_bg_task.cancel()
        try:
            await cleanup_bg_task
        except asyncio.CancelledError:
            pass

    processing_jobs = await store.get_jobs_by_status("processing")
    for job in processing_jobs:
        logger.warning(f"Job {job['id']} interrupted by shutdown, marking as FAILED")
        await store.update_job_status(job["id"], "failed", error="Container shutdown")

    logger.info("DiCoW API shutdown complete")


app = FastAPI(
    title="DiCoW Transcription API",
    description="API for asynchronous audio transcription using DiCoW",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "worker_running": worker._running if worker else False,
        "queue_size": worker.queue.qsize() if worker else 0,
    }


@app.post("/transcribe", response_model=JobResponse)
async def transcribe(
    file: Optional[UploadFile] = File(None),
    mode: TranscribeMode = Form(TranscribeMode.SINGLE_FILE),
    folder_path: Optional[str] = Form(None),
    file_pattern: str = Form("*.wav"),
):
    """
    Submit a transcription job.

    - **mode**: `single_file` (upload via HTTP) or `batch_folder` (process files from mounted volume)
    - **file**: Audio file (.wav) for single_file mode
    - **folder_path**: Path in mounted volume for batch_folder mode
    - **file_pattern**: Glob pattern for batch mode (default: *.wav)
    """
    job_id = str(uuid.uuid4())

    if mode == TranscribeMode.SINGLE_FILE:
        if not file:
            raise HTTPException(
                status_code=400, detail="No file uploaded for single_file mode"
            )

        file_path = await http_source.save_uploaded_file(file, job_id)

        input_data = {
            "mode": "single_file",
            "file_path": file_path,
            "filename": file.filename,
        }

        await store.create_job(
            job_id, input_data, config.TTL_HOURS, upload_path=file_path
        )

    else:
        if not folder_path:
            raise HTTPException(
                status_code=400, detail="folder_path required for batch_folder mode"
            )

        if not await local_source.validate_folder_access(folder_path):
            raise HTTPException(
                status_code=400, detail=f"Folder not accessible: {folder_path}"
            )

        files = await local_source.get_files(folder_path, file_pattern)
        if not files:
            raise HTTPException(
                status_code=400,
                detail=f"No files found matching '{file_pattern}' in {folder_path}",
            )

        input_data = {
            "mode": "batch_folder",
            "folder_path": folder_path,
            "file_pattern": file_pattern,
            "files_count": len(files),
        }

        await store.create_job(job_id, input_data, config.TTL_HOURS)

    await worker.submit_job(job_id)

    logger.info(f"Job {job_id} created with mode {mode.value}")

    return JobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        created_at=datetime.utcnow().replace(tzinfo=timezone.utc),
        updated_at=datetime.utcnow().replace(tzinfo=timezone.utc),
        mode=mode,
        input_info=input_data,
    )


@app.get("/jobs", response_model=List[JobResponse])
async def list_jobs(status: Optional[JobStatus] = None):
    """List all jobs, optionally filtered by status"""
    if status:
        jobs = await store.get_jobs_by_status(status.value)
    else:
        jobs = await store.get_all_jobs()

    return [
        JobResponse(
            job_id=job["id"],
            status=JobStatus(job["status"]),
            created_at=datetime.fromisoformat(job["created_at"].replace("Z", "+00:00")),
            updated_at=datetime.fromisoformat(job["updated_at"].replace("Z", "+00:00")),
            mode=json.loads(job["input_json"])["mode"],
            input_info=json.loads(job["input_json"]),
            retry_count=job.get("retry_count", 0),
        )
        for job in jobs
    ]


@app.get("/jobs/{job_id}", response_model=JobDetailResponse)
async def get_job(job_id: str):
    """Get job details and status"""
    job_data = await store.get_job(job_id)

    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")

    input_data = json.loads(job_data["input_json"])

    return JobDetailResponse(
        job_id=job_data["id"],
        status=JobStatus(job_data["status"]),
        created_at=datetime.fromisoformat(
            job_data["created_at"].replace("Z", "+00:00")
        ),
        updated_at=datetime.fromisoformat(
            job_data["updated_at"].replace("Z", "+00:00")
        ),
        mode=input_data["mode"],
        input_info=input_data,
        completed_at=datetime.fromisoformat(
            job_data["completed_at"].replace("Z", "+00:00")
        )
        if job_data.get("completed_at")
        else None,
        error=job_data.get("error"),
        result_path=job_data.get("result_path"),
    )


@app.get("/jobs/{job_id}/result", response_model=JobResultResponse)
async def get_job_result(job_id: str):
    """Get transcription result for completed job"""
    job_data = await store.get_job(job_id)

    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")

    if job_data["status"] != "completed":
        raise HTTPException(
            status_code=400, detail=f"Job not completed. Status: {job_data['status']}"
        )

    result_path = job_data.get("result_path")
    if not result_path or not os.path.exists(result_path):
        raise HTTPException(status_code=500, detail="Result file not found")

    with open(result_path, "r", encoding="utf-8") as f:
        result = json.load(f)

    return JobResultResponse(**result)


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its result file"""
    job_data = await store.get_job(job_id)

    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")

    result_path = job_data.get("result_path")
    if result_path and os.path.exists(result_path):
        os.remove(result_path)
        logger.debug(f"Deleted result file: {result_path}")

    upload_path = job_data.get("upload_path")
    if upload_path and os.path.exists(upload_path):
        os.remove(upload_path)
        logger.debug(f"Deleted uploaded file: {upload_path}")

    await store.delete_job(job_id)

    return {"message": "Job deleted successfully"}

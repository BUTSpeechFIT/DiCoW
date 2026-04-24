from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Header
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from contextlib import asynccontextmanager
import uuid
import os
import json
import asyncio
import tempfile
from datetime import datetime, timezone
from typing import List, Optional
import time

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
from .formatters import format_openai_response

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


@app.post("/v1/audio/transcriptions")
async def transcribe_openai(
    # Required parameters
    file: UploadFile = File(..., description="Audio file to transcribe (max 25MB). Supported formats: mp3, mp4, mpeg, mpga, m4a, wav, webm."),
    model: str = Form("BUT-FIT/SE-DiCoW", description="Model identifier. Use 'BUT-FIT/SE-DiCoW' for diarization-capable transcription."),
    
    # Optional parameters
    language: Optional[str] = Form(None, description="Language code (e.g., 'en', 'fr'). Auto-detected if not provided."),
    prompt: Optional[str] = Form(None, description="Optional prompt to guide context. Note: Currently not supported by SE-DiCoW model."),
    response_format: str = Form("json", description="Output format: 'json' | 'text' | 'verbose_json' | 'diarized_json'. Default: 'json'."),
    stream: bool = Form(False, description="Enable streaming response. Default: false."),
    temperature: Optional[float] = Form(None, description="Sampling temperature (0.0-1.0). Higher values increase randomness. Default: 0.0."),
    timestamp_granularities: Optional[str] = Form(None, description="Timestamp granularity: 'segment' or 'word'. For OpenAI compatibility, use 'timestamp_granularities[]=word'."),
    diarize: bool = Form(True, description="Enable speaker diarization. When false, merges all speakers. Default: true."),
    
    # Headers (not enforced - gateway handles auth)
    x_api_key: Optional[str] = Header(None, alias="X-API-Key", include_in_schema=False)
):
    """
    Transcribe audio to text with optional speaker diarization.
    
    OpenAI-compatible endpoint using SE-DiCoW model for multi-speaker transcription.
    
    ## Features
    - **Multi-speaker diarization**: Automatically identifies and separates speakers
    - **Word-level timestamps**: Precise timing for each word
    - **Multiple output formats**: JSON, plain text, verbose JSON with metadata
    - **Language auto-detection**: Automatically detects spoken language
    
    ## Response Formats
    
    ### json (default)
    Simple text output: `{"text": "transcription..."}`
    
    ### text
    Plain text transcription without JSON wrapper
    
    ### verbose_json
    Full metadata including segments, word timestamps, and confidence scores
    
    ### diarized_json
    Speaker-labeled segments with speaker identifiers
    
    ## Examples
    
    **Basic transcription:**
    ```bash
    curl -X POST /v1/audio/transcriptions -F "file=@meeting.wav"
    ```
    
    **Diarized transcription with word timestamps:**
    ```bash
    curl -X POST /v1/audio/transcriptions \\
      -F "file=@meeting.wav" \\
      -F "response_format=diarized_json" \\
      -F "timestamp_granularities=word" \\
      -F "diarize=true"
    ```
    """
    
    # Validate file
    if not file or not file.filename:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "No file uploaded",
                "message": "Please provide an audio file in the 'file' field"
            }
        )
    
    # Validate file size (25MB limit like OpenAI)
    file_content = await file.read()
    MAX_FILE_SIZE_MB = 25
    if len(file_content) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail={
                "error": "File too large",
                "message": f"File size exceeds {MAX_FILE_SIZE_MB}MB limit"
            }
        )
    
    # Validate response_format
    valid_formats = ["json", "text", "verbose_json", "diarized_json"]
    if response_format not in valid_formats:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid response_format",
                "message": f"Must be one of: {', '.join(valid_formats)}"
            }
        )
    
    # Validate timestamp_granularities
    if timestamp_granularities and timestamp_granularities not in ["segment", "word"]:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid timestamp_granularities",
                "message": "Must be 'segment' or 'word'"
            }
        )
    
    # Log warning if prompt is provided (not supported)
    if prompt:
        logger.warning(f"Prompt parameter provided but not supported: {prompt[:50]}...")
    
    # Save to temp file
    file_ext = os.path.splitext(file.filename)[1] if file.filename else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        tmp.write(file_content)
        temp_path = tmp.name
    
    try:
        # Run pipeline in thread pool (non-blocking)
        start_time = time.time()
        
        result = await asyncio.to_thread(
            lambda: dicow_pipeline.transcribe_openai(
                audio_path=temp_path,
                language=language,
                temperature=temperature if temperature is not None else 0.0,
                return_word_timestamps=(timestamp_granularities == "word"),
                diarize=diarize,
                # Pass thresholds for hallucination detection
                compression_ratio_threshold=2.0,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6
            ),
            timeout=300  # 5 minute timeout
        )
        
        # Format response
        response_data = format_openai_response(
            result,
            response_format=response_format,
            diarize=diarize,
            timestamp_granularities=timestamp_granularities  # Pass to formatter
        )
        
        processing_time = time.time() - start_time
        
        # Return response
        if stream:
            return StreamingResponse(
                _stream_json_response(response_data),
                media_type="application/json" if "json" in response_format else "text/plain",
                headers={
                    "X-Processing-Time": f"{processing_time:.2f}s"
                }
            )
        elif response_format == "text":
            return PlainTextResponse(
                response_data,
                headers={
                    "X-Processing-Time": f"{processing_time:.2f}s"
                }
            )
        else:
            return JSONResponse(
                response_data,
                headers={
                    "X-Processing-Time": f"{processing_time:.2f}s"
                }
            )
    
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=408,
            detail={
                "error": "Timeout",
                "message": "Transcription exceeded 300 second timeout. Consider splitting long audio files."
            }
        )
    except Exception as e:
        logger.error(f"Transcription failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": f"Transcription failed: {str(e)}"
            }
        )
    finally:
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


async def _stream_json_response(data):
    """Stream JSON response in chunks."""
    import json
    yield json.dumps(data).encode()

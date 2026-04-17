from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TranscribeMode(str, Enum):
    SINGLE_FILE = "single_file"
    BATCH_FOLDER = "batch_folder"


class Segment(BaseModel):
    speaker: int
    start: float
    end: float
    text: str


class TranscriptionResult(BaseModel):
    segments: List[Segment]
    speakers_count: int
    duration_seconds: float
    metadata: Dict[str, Any]


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    mode: TranscribeMode
    input_info: Dict[str, Any]
    retry_count: int = 0


class JobDetailResponse(JobResponse):
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    result_path: Optional[str] = None


class JobResultResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: datetime
    completed_at: datetime
    input_info: Dict[str, Any]
    result: TranscriptionResult

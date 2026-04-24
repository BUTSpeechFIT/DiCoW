"""
OpenAI-compatible response formatters for SE-DiCoW transcription output.

Converts internal model output to OpenAI API response formats:
- json: Simple text response
- text: Plain text string
- verbose_json: Full metadata with segments and word timestamps
- diarized_json: Speaker-labeled segments
"""

from typing import List, Dict, Any, Literal, Union, Optional
import math


def format_openai_response(
    model_output: Dict[str, Any],
    response_format: Literal["json", "text", "verbose_json", "diarized_json"],
    diarize: bool = True,
    timestamp_granularities: Optional[str] = None
) -> Union[str, Dict[str, Any]]:
    """
    Convert pipeline output to OpenAI-compatible format.
    
    Args:
        model_output: Dict from pipeline.transcribe_openai() with keys:
            - text: str (full transcription)
            - segments: List[Dict] (segment-level data)
            - word_segments: List[Dict] or None (word-level data)
            - speakers_count: int
            - duration: float
            - language: str
            - metadata: Dict (compression_ratio, avg_logprob, no_speech_prob)
        response_format: Target format (json, text, verbose_json, diarized_json)
        diarize: Whether to keep speaker separation
        timestamp_granularities: "segment", "word", or None (for verbose_json)
    
    Returns:
        Formatted response according to response_format
    """
    if response_format == "json":
        return _format_json(model_output, diarize)
    
    elif response_format == "text":
        return _format_text(model_output, diarize)
    
    elif response_format == "verbose_json":
        return _format_verbose_json(model_output, timestamp_granularities)
    
    elif response_format == "diarized_json":
        return _format_diarized_json(model_output)
    
    else:
        raise ValueError(f"Unknown response_format: {response_format}")


def _format_json(model_output: Dict[str, Any], diarize: bool) -> Dict[str, str]:
    """Format as simple JSON with text field only."""
    return {"text": model_output["text"]}


def _format_text(model_output: Dict[str, Any], diarize: bool) -> str:
    """Format as plain text string."""
    return model_output["text"]


def _format_verbose_json(
    model_output: Dict[str, Any],
    timestamp_granularities: Optional[str] = None
) -> Dict[str, Any]:
    """
    Format as verbose JSON with full metadata.
    
    Matches OpenAI's verbose_json format:
    - task: "transcribe"
    - language: Detected language
    - duration: Audio duration in seconds
    - text: Full transcription
    - segments: Only when timestamp_granularities="segment" (or None/default)
    - words: Only when timestamp_granularities="word"
    
    Args:
        model_output: Dict from pipeline
        timestamp_granularities: "segment", "word", or None
    """
    segments = model_output.get("segments", [])
    word_segments = model_output.get("word_segments")
    
    # Base response structure matching OpenAI
    response = {
        "task": "transcribe",
        "language": model_output.get("language", "en"),
        "duration": model_output.get("duration", 0.0),
        "text": model_output["text"]
    }
    
    # Add segments only when requested or by default
    if timestamp_granularities is None or timestamp_granularities == "segment":
        response["segments"] = _format_segments_verbose_openai(segments)
    
    # Add words only when explicitly requested
    if timestamp_granularities == "word" and word_segments:
        response["words"] = _format_word_segments_openai(word_segments)
    
    return response


def _format_diarized_json(model_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format as diarized JSON with speaker labels.
    
    Matches OpenAI's diarized_json format:
    - text: Transcription with speaker labels
    - segments: Speaker-separated segments with speaker field
    - duration: Audio duration
    
    OpenAI segment format:
    {
        "type": "transcript.text.segment",
        "id": "seg_001",
        "start": 0.0,
        "end": 4.7,
        "text": "Thanks for calling...",
        "speaker": "agent"
    }
    """
    segments = model_output.get("segments", [])
    
    return {
        "text": model_output["text"],
        "segments": _format_segments_diarized_openai(segments),
        "duration": model_output.get("duration", 0.0)
    }


def _format_segments_diarized_openai(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Format segments matching OpenAI's diarized_json structure.
    
    Includes type and id fields to match OpenAI spec.
    """
    formatted_segments = []
    
    for i, seg in enumerate(segments):
        speaker_id = seg.get("speaker", 0)
        formatted_seg = {
            "type": "transcript.text.segment",
            "id": f"seg_{i+1:03d}",  # Format: seg_001, seg_002, etc.
            "start": round(seg.get("start", 0.0), 2),
            "end": round(seg.get("end", 0.0), 2),
            "text": seg.get("text", ""),
            "speaker": f"speaker_{speaker_id}"  # Could be customized if speaker names provided
        }
        formatted_segments.append(formatted_seg)
    
    return formatted_segments


def _format_segments_verbose_openai(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Format segments for verbose_json matching OpenAI's exact structure.
    
    OpenAI format:
    {
        "id": 0,
        "seek": 0,
        "start": 0.0,
        "end": 3.32,
        "text": " The beach was a popular spot...",
        "tokens": [50364, 440, 7534, ...],
        "temperature": 0.0,
        "avg_logprob": -0.286,
        "compression_ratio": 1.236,
        "no_speech_prob": 0.009
    }
    """
    formatted_segments = []
    
    for i, seg in enumerate(segments):
        formatted_seg = {
            "id": i,
            "seek": 0,  # Seek position (0 for single-segment processing)
            "start": round(seg.get("start", 0.0), 2),
            "end": round(seg.get("end", 0.0), 2),
            "text": seg.get("text", ""),
            "tokens": seg.get("tokens", []),  # Token IDs if available
            "temperature": 0.0,  # Default temperature
            "avg_logprob": seg.get("avg_logprob", -0.5),
            "compression_ratio": seg.get("compression_ratio", 1.0),
            "no_speech_prob": seg.get("no_speech_prob", 0.0)
        }
        formatted_segments.append(formatted_seg)
    
    return formatted_segments


def _format_word_segments_openai(word_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Format word-level segments matching OpenAI's exact structure.
    
    OpenAI format:
    {
        "word": "The",
        "start": 0.0,
        "end": 0.24
    }
    
    Note: Uses "word" field (not "text") to match OpenAI spec.
    """
    formatted_words = []
    
    for word in word_segments:
        formatted_word = {
            "word": word.get("text", "").strip(),  # OpenAI uses "word" not "text"
            "start": round(word.get("start", 0.0), 2),
            "end": round(word.get("end", 0.0), 2)
        }
        
        # Only include non-empty words
        if formatted_word["word"]:
            formatted_words.append(formatted_word)
    
    return formatted_words


def _format_segments_diarized(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Format segments with speaker labels for diarized_json.
    
    Converts speaker integers to string labels:
    - speaker: "speaker_0", "speaker_1", etc.
    - start: Start time in seconds
    - end: End time in seconds
    - text: Segment text
    """
    formatted_segments = []
    
    for seg in segments:
        speaker_id = seg.get("speaker", 0)
        formatted_seg = {
            "speaker": f"speaker_{speaker_id}",
            "start": round(seg.get("start", 0.0), 2),
            "end": round(seg.get("end", 0.0), 2),
            "text": seg.get("text", "")
        }
        formatted_segments.append(formatted_seg)
    
    return formatted_segments
    """
    Merge all speaker segments chronologically into single transcript.
    
    Sorts segments by start time and concatenates text.
    Used when diarize=False to produce unified transcript.
    
    Args:
        segments: List of segment dicts with 'start' and 'text' keys
    
    Returns:
        Merged transcript text
    """
    # Sort by start time
    sorted_segments = sorted(segments, key=lambda x: x.get("start", 0.0))
    
    # Concatenate text with spaces
    texts = [seg.get("text", "") for seg in sorted_segments]
    return " ".join(texts)


def format_diarized_text(segments: List[Dict[str, Any]]) -> str:
    """
    Format segments with speaker labels as readable text.
    
    Groups segments by speaker and formats as:
    "🗣️ Speaker 0:\n[text]\n\n🗣️ Speaker 1:\n[text]"
    
    Args:
        segments: List of segment dicts with 'speaker' and 'text' keys
    
    Returns:
        Formatted text with speaker labels
    """
    # Group by speaker
    by_speaker: Dict[int, List[str]] = {}
    for seg in segments:
        speaker = seg.get("speaker", 0)
        if speaker not in by_speaker:
            by_speaker[speaker] = []
        by_speaker[speaker].append(seg.get("text", ""))
    
    # Format with speaker labels
    lines = []
    for speaker_id in sorted(by_speaker.keys()):
        texts = by_speaker[speaker_id]
        lines.append(f"🗣️ Speaker {speaker_id}:")
        lines.append(" ".join(texts))
    
    return "\n\n".join(lines)


def extract_metadata_from_generation(
    segments: List[Dict[str, Any]],
    compression_ratio_threshold: float = 2.0,
    logprob_threshold: float = -1.0
) -> Dict[str, float]:
    """
    Extract metadata fields from generation output.
    
    Calculates aggregate statistics for verbose_json:
    - avg_logprob: Average log probability across segments
    - compression_ratio: Average compression ratio
    - no_speech_prob: Average no-speech probability
    
    Args:
        segments: List of segment dicts from model output
        compression_ratio_threshold: Threshold for hallucination detection
        logprob_threshold: Minimum acceptable log probability
    
    Returns:
        Dict with metadata fields
    """
    if not segments:
        return {
            "avg_logprob": 0.0,
            "compression_ratio": 1.0,
            "no_speech_prob": 0.0
        }
    
    # Extract available metadata
    logprobs = [seg.get("avg_logprob") for seg in segments if "avg_logprob" in seg]
    compression_ratios = [seg.get("compression_ratio") for seg in segments if "compression_ratio" in seg]
    no_speech_probs = [seg.get("no_speech_prob") for seg in segments if "no_speech_prob" in seg]
    
    # Calculate averages or use defaults
    avg_logprob = sum(logprobs) / len(logprobs) if logprobs else -0.5
    avg_compression = sum(compression_ratios) / len(compression_ratios) if compression_ratios else 1.0
    avg_no_speech = sum(no_speech_probs) / len(no_speech_probs) if no_speech_probs else 0.0
    
    return {
        "avg_logprob": round(avg_logprob, 4),
        "compression_ratio": round(avg_compression, 2),
        "no_speech_prob": round(avg_no_speech, 4)
    }

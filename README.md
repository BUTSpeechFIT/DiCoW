# DiCoW Docker API

Asynchronous REST API for multi-speaker audio transcription using SE-DiCoW (Diarization-Conditioned Whisper).

## 🚀 Quick Start

```bash
# Build and start
docker-compose build
docker-compose up
```

API available at `http://localhost:8000`.

Interactive Swagger documentation: `http://localhost:8000/docs`

## 📡 Usage

### Transcribe an audio file

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "mode=single_file" \
  -F "file=@meeting.wav"
```

Response:
```json
{
  "job_id": "uuid-1234",
  "status": "pending"
}
```

### Check job status

```bash
curl http://localhost:8000/jobs/{job_id}
```

### Get transcription result

```bash
curl http://localhost:8000/jobs/{job_id}/result
```

Response:
```json
{
  "result": {
    "segments": [
      {"speaker": 0, "start": 0.00, "end": 2.50, "text": "Hello"},
      {"speaker": 1, "start": 3.00, "end": 6.00, "text": "Hi there"}
    ],
    "speakers_count": 2,
    "duration_seconds": 180.5
  }
}
```

### Batch mode (folder)

```bash
# Copy files to mounted volume
mkdir -p ./data/batch_input
cp *.wav ./data/batch_input/

# Submit batch job
curl -X POST http://localhost:8000/transcribe \
  -F "mode=batch_folder" \
  -F "folder_path=/app/data/batch_input" \
  -F "file_pattern=*.wav"
```

## ⚙️ Configuration

Edit `docker-compose.yml`:

| Variable              | Description                | Default |
| --------------------- | -------------------------- | ------- |
| `MAX_CONCURRENT_JOBS` | Max concurrent jobs        | `4`     |
| `TTL_HOURS`           | Job retention time (hours) | `24`    |

## 📚 Full Documentation

See [API_README.md](API_README.md)

## 📝 License

- **DiCoW**: Apache License 2.0
- **DiCoW Model**: CC BY 4.0
- **Diarizen**: CC BY-NC 4.0 (research/non-commercial only)

## 📞 Contact

[ipoloka@fit.vut.cz](mailto:ipoloka@fit.vut.cz), [xkleme15@vutbr.cz](mailto:xkleme15@vutbr.cz)

# DiCoW-v1: Diarization-Conditioned Whisper for Target Speaker Automatic Speech Recognition

DiCoW (Diarization-Conditioned Whisper) enhances OpenAI’s Whisper ASR model by integrating **speaker diarization** for multi-speaker transcription. The app leverages `pyannote/speaker-diarization-3.1` to segment speakers and provides diarization-conditioned transcription for long-form audio inputs.

## Features

- **Multi-Speaker ASR**: Handles multi-speaker audio using diarization-aware transcription.
- **Flexible Input Sources**:
  - **Microphone**: Record and transcribe live audio.
  - **Audio File Upload**: Upload pre-recorded audio files for transcription.
- **Diarization Support**: Powered by `pyannote/speaker-diarization-3.1` for accurate speaker segmentation.
- **Built with 🤗 Transformers**: Uses the latest Whisper checkpoints for robust transcription.

---

## Demo
Run the app directly in your browser with the following link: https://b6c05a6fdfd3cd4dee.gradio.live

![DiCoW-v1 Demo](img.png)
---

## Installation

### Requirements
- Python 3.11+
- Required Python Libraries:
  - `gradio`
  - `transformers`
  - `pyannote.audio`
  - `torch`
  - 
### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/DiCoW-v1.git
   cd DiCoW-v1

2. Setup dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Export your Hugging Face API token:
   ```bash
   export HF_HOME=''
   ```
4. Run the app:
   ```bash
    python app.py
    ```
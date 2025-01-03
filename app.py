import torch

import gradio as gr
from transformers import AutoTokenizer, AutoFeatureExtractor
from modeling_dicow import DiCoWForConditionalGeneration
from pyannote.audio import Pipeline
from pipeline import DiCoWPipeline
import os

BATCH_SIZE = 8
FILE_LIMIT_MB = 1000

device = 0 if torch.cuda.is_available() else "cpu"

MODEL_NAME = "BUT-FIT/DiCoW_v1"
dicow = DiCoWForConditionalGeneration.from_pretrained(MODEL_NAME)
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
diar_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=os.environ['HF_TOKEN'])
pipeline = DiCoWPipeline(dicow, diarization_pipeline=diar_pipeline, feature_extractor=feature_extractor,
                         tokenizer=tokenizer)


def transcribe(inputs):
    if inputs is None:
        raise gr.Error("No audio file submitted! Please upload or record an audio file before submitting your request.")

    text = pipeline(inputs, return_timestamps=True)["text"]
    return text


demo = gr.Blocks(theme=gr.themes.Ocean())

mf_transcribe = gr.Interface(
    fn=transcribe,
    inputs=[
        gr.Audio(sources="microphone", type="filepath"),
        # gr.Radio(["transcribe", "translate"], label="Task", value="transcribe"),
    ],
    outputs="text",
    title="DiCoW-v1: Diarization-Conditioned Whisper",
    description=(
        "DiCoW (Diarization-Conditioned Whisper) enhances Whisper with diarization-aware transcription, enabling it to handle multi-speaker audio effectively. "
        "Use your microphone to transcribe audio with speaker-aware precision! This demo uses the"
        f" checkpoint [{MODEL_NAME}](https://huggingface.co/{MODEL_NAME}) and 🤗 Transformers for diarization-conditioned transcription. "
        "Speaker diarization is powered by the `pyannote/speaker-diarization-3.1`."
    ),
    allow_flagging="never",
)

file_transcribe = gr.Interface(
    fn=transcribe,
    inputs=[
        gr.Audio(sources="upload", type="filepath", label="Audio file"),
    ],
    outputs="text",
    title="DiCoW-v1: Diarization-Conditioned Whisper",
    description=(
        "DiCoW (Diarization-Conditioned Whisper) supports diarization-aware transcription for multi-speaker audio files. "
        f"Upload an audio file to experience state-of-the-art multi-speaker transcription. Demo uses the checkpoint "
        f"[{MODEL_NAME}](https://huggingface.co/{MODEL_NAME}) and 🤗 Transformers. "
        "Speaker diarization is powered by the `pyannote/speaker-diarization-3.1`."
    ),
    allow_flagging="never",
)

# if __name__ == "__main__":
with demo:
    gr.TabbedInterface([mf_transcribe, file_transcribe], ["Microphone", "Audio file"])

demo.queue().launch(ssr_mode=False, share=True)

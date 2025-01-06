import torch

import gradio as gr
from transformers import AutoTokenizer, AutoFeatureExtractor
from modeling_dicow import DiCoWForConditionalGeneration
from pyannote.audio import Pipeline
from pipeline import DiCoWPipeline
import os


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

MODEL_NAME = "BUT-FIT/DiCoW_v1"
dicow = DiCoWForConditionalGeneration.from_pretrained(MODEL_NAME)
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
diar_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=os.environ['HF_TOKEN']).to(device)
pipeline = DiCoWPipeline(dicow, diarization_pipeline=diar_pipeline, feature_extractor=feature_extractor,
                         tokenizer=tokenizer, device=device)

def transcribe(inputs):
    if inputs is None:
        raise gr.Error(
            "No audio file submitted! Please upload or record an audio file before submitting your request.")

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

    gr.Markdown(
        """
        ## Features

        - **Multi-Speaker ASR**: Handles multi-speaker audio using diarization-aware transcription.  
        - **Flexible Input Sources**:  
          - **Microphone**: Record and transcribe live audio.  
          - **Audio File Upload**: Upload pre-recorded audio files for transcription.  
        - **Diarization Support**: Powered by `pyannote/speaker-diarization-3.1` for accurate speaker segmentation.  
        - **Built with 🤗 Transformers**: Uses the latest Whisper checkpoints for robust transcription.  

        **Disclaimer**: This version of DiCoW currently supports **English only** and is still under **active development**. Expect frequent updates and feature improvements.

        ## Citation
        If you use our model or code, please, cite:
        ```bibtex
        @misc{polok2024dicowdiarizationconditionedwhispertarget,
              title={DiCoW: Diarization-Conditioned Whisper for Target Speaker Automatic Speech Recognition}, 
              author={Alexander Polok and Dominik Klement and Martin Kocour and Jiangyu Han and Federico Landini and Bolaji Yusuf and Matthew Wiesner and Sanjeev Khudanpur and Jan Černocký and Lukáš Burget},
              year={2024},
              eprint={2501.00114},
              archivePrefix={arXiv},
              primaryClass={eess.AS},
              url={https://arxiv.org/abs/2501.00114}, 
        }
        @misc{polok2024targetspeakerasrwhisper,
              title={Target Speaker ASR with Whisper}, 
              author={Alexander Polok and Dominik Klement and Matthew Wiesner and Sanjeev Khudanpur and Jan Černocký and Lukáš Burget},
              year={2024},
              eprint={2409.09543},
              archivePrefix={arXiv},
              primaryClass={eess.AS},
              url={https://arxiv.org/abs/2409.09543}, 
        }
        ```

        ## Contributing
        We welcome contributions! If you’d like to add features or improve our pipeline, please open an issue or submit a pull request.
        """
    )
demo.queue().launch(ssr_mode=False, share=False)

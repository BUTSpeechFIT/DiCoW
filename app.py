import torch

import gradio as gr
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModelForSpeechSeq2Seq
from pipeline import DiCoWPipeline
from diarizen.pipelines.inference import DiariZenPipeline


def create_lower_uppercase_mapping(tokenizer):
    tokenizer.upper_cased_tokens = {}
    vocab = tokenizer.get_vocab()
    for token, index in vocab.items():
        if len(token) < 1:
            continue
        if token[0] == 'Ġ' and len(token) > 1:
            lower_cased_token = token[0] + token[1].lower() + (token[2:] if len(token) > 2 else '')
        else:
            lower_cased_token = token[0].lower() + token[1:]
        if lower_cased_token != token:
            lower_index = vocab.get(lower_cased_token, None)
            if lower_index is not None:
                tokenizer.upper_cased_tokens[lower_index] = index
            else:
                pass


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

MODEL_NAME = "BUT-FIT/DiCoW_v3_2"
DIARIZATION_MODEL="BUT-FIT/diarizen-wavlm-large-s80-md"
dicow = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_NAME, trust_remote_code=True)
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
create_lower_uppercase_mapping(tokenizer)
dicow.set_tokenizer(tokenizer)
diar_pipeline = DiariZenPipeline.from_pretrained(DIARIZATION_MODEL).to(device)
diar_pipeline.embedding_batch_size = 16

pipeline = DiCoWPipeline(dicow, diarization_pipeline=diar_pipeline, feature_extractor=feature_extractor,
                         tokenizer=tokenizer, device=device)

def transcribe(inputs):
    if inputs is None:
        raise gr.Error(
            "No audio file submitted! Please upload or record an audio file before submitting your request. "
            "Note: You might have pressed the 'Submit' button before the audio was displayed.")

    text = pipeline(inputs, return_timestamps=True)["text"]
    torch.cuda.empty_cache()
    return text


demo = gr.Blocks(theme=gr.themes.Ocean())

mf_audio = gr.Audio(sources="microphone", type="filepath", format="wav")


mf_transcribe = gr.Interface(
    fn=transcribe,
    inputs=[mf_audio
    ],
    outputs="text",
    title="DiCoW: Diarization-Conditioned Whisper",
    description=(
        "DiCoW (Diarization-Conditioned Whisper) enhances Whisper with diarization-aware transcription, enabling it to handle multi-speaker audio effectively. "
        "Use your microphone to transcribe audio with speaker-aware precision! This demo uses the"
        f" checkpoint [{MODEL_NAME}](https://huggingface.co/{MODEL_NAME}) and 🤗 Transformers for diarization-conditioned transcription. "
        f"Speaker diarization is powered by the `{DIARIZATION_MODEL}`. **Note:** CTC joint decoding is disabled."
    ),
    allow_flagging="never",
)

file_transcribe = gr.Interface(
    fn=transcribe,
    inputs=[
        gr.Audio(sources="upload", type="filepath", label="Audio file"),
    ],
    outputs="text",
    title="DiCoW: Diarization-Conditioned Whisper",
    description=(
        "DiCoW (Diarization-Conditioned Whisper) supports diarization-aware transcription for multi-speaker audio files. "
        f"Upload an audio file to experience state-of-the-art multi-speaker transcription. Demo uses the checkpoint "
        f"[{MODEL_NAME}](https://huggingface.co/{MODEL_NAME}) and 🤗 Transformers. "
        f"Speaker diarization is powered by the `{DIARIZATION_MODEL}`. **Note:** CTC joint decoding is disabled."
    ),
    allow_flagging="never",
)

# if __name__ == "__main__":
with demo:
    gr.TabbedInterface([ file_transcribe, mf_transcribe], ["Audio file", "Microphone"])

    gr.Markdown(
        """
        ## Features

        - **Multi-Speaker ASR**: Handles multi-speaker audio using diarization-aware transcription.  
        - **Flexible Input Sources**:  
          - **Microphone**: Record and transcribe live audio.  
          - **Audio File Upload**: Upload pre-recorded audio files for transcription.  
        - **Diarization Support**: Powered by `BUT-FIT/diarizen-wavlm-large-s80-md` for accurate speaker segmentation.  
        - **Built with 🤗 Transformers**: Uses the latest Whisper checkpoints for robust transcription.  

        ## Citation
        If you use our model or code, please, cite:
        ```bibtex
        @article{POLOK2026101841,
            title = {DiCoW: Diarization-conditioned Whisper for target speaker automatic speech recognition},
            journal = {Computer Speech & Language},
            volume = {95},
            pages = {101841},
            year = {2026},
            issn = {0885-2308},
            doi = {https://doi.org/10.1016/j.csl.2025.101841},
            url = {https://www.sciencedirect.com/science/article/pii/S088523082500066X},
            author = {Alexander Polok and Dominik Klement and Martin Kocour and Jiangyu Han and Federico Landini and Bolaji Yusuf and Matthew Wiesner and Sanjeev Khudanpur and Jan Černocký and Lukáš Burget},
            keywords = {Diarization-conditioned Whisper, Target-speaker ASR, Speaker diarization, Long-form ASR, Whisper adaptation},
        }
        
        @INPROCEEDINGS{10887683,
          author={Polok, Alexander and Klement, Dominik and Wiesner, Matthew and Khudanpur, Sanjeev and Černocký, Jan and Burget, Lukáš},
          booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
          title={Target Speaker ASR with Whisper}, 
          year={2025},
          volume={},
          number={},
          pages={1-5},
          keywords={Transforms;Signal processing;Transformers;Acoustics;Speech processing;target-speaker ASR;diarization conditioning;multi-speaker ASR;Whisper},
          doi={10.1109/ICASSP49660.2025.10887683}
        }
        
        ```

        ## Contributing
        We welcome contributions! If you’d like to add features or improve our pipeline, please open an issue or submit a pull request.
        """
    )
    mf_audio.start_recording(lambda _: gr.Warning(f"Please wait for the audio to be displayed before submitting!"))

demo.queue(max_size=5).launch( share=False, root_path="/gradio-demo")

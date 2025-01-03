from transformers import AutoTokenizer, AutoFeatureExtractor
from modeling_dicow import DiCoWForConditionalGeneration
from pyannote.audio import Pipeline
from pipeline import DiCoWPipeline
import os

if __name__ == "__main__":
    MODEL_ID = "BUT-FIT/DiCoW_v1"
    dicow = DiCoWForConditionalGeneration.from_pretrained(MODEL_ID)
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    diar_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=os.environ['HF_TOKEN'])
    pipeline = DiCoWPipeline(dicow, diarization_pipeline=diar_pipeline, feature_extractor=feature_extractor,
                             tokenizer=tokenizer)
    transcript = pipeline("test.wav", return_timestamps=True)
    print(transcript["text"])

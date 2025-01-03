from transformers import AutoConfig, AutoModelForSpeechSeq2Seq
from .dicow_config import DiCoWConfig
from .modeling_dicow import DiCoWForConditionalGeneration

AutoConfig.register("dicow", DiCoWConfig)
AutoModelForSpeechSeq2Seq.register(DiCoWConfig, DiCoWForConditionalGeneration)

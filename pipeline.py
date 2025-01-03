from typing import Dict, Optional

import torch
from transformers.pipelines.automatic_speech_recognition import AutomaticSpeechRecognitionPipeline


class DiCoWPipeline(AutomaticSpeechRecognitionPipeline):
    def __init__(self, *args, diarization_pipeline, **kwargs):
        super().__init__(*args, **kwargs)
        self.diarization_pipeline = diarization_pipeline
        self.type = "seq2seq_whisper"

    def get_diarization_mask(self, per_speaker_samples, audio_length):
        diarization_mask = torch.zeros(len(per_speaker_samples), audio_length)
        for i, speaker_samples in enumerate(per_speaker_samples):
            for start, end in speaker_samples:
                diarization_mask[i, round(start * 50):round(end * 50)] = 1
        return diarization_mask

    @staticmethod
    def get_stno_mask(diar_mask, s_index):
        non_target_mask = torch.ones((diar_mask.shape[0],), dtype=torch.bool)
        non_target_mask[s_index] = False
        sil_frames = (1 - diar_mask).prod(axis=0)
        anyone_else = (1 - diar_mask[non_target_mask]).prod(axis=0)
        target_spk = diar_mask[s_index] * anyone_else
        non_target_spk = (1 - diar_mask[s_index]) * (1 - anyone_else)
        overlapping_speech = diar_mask[s_index] - target_spk
        stno_mask = torch.stack([sil_frames, target_spk, non_target_spk, overlapping_speech], axis=0)
        return stno_mask

    def preprocess(self, inputs, chunk_length_s=0, stride_length_s=None):
        if not isinstance(inputs, str):
            raise ValueError("For now input must be a string representing a path to an audio file")
        generator = super().preprocess(inputs, chunk_length_s=chunk_length_s, stride_length_s=stride_length_s)
        sample = next(generator)
        diariation_output = self.diarization_pipeline(inputs)
        per_speaker_samples = []
        for speaker in diariation_output.labels():
            per_speaker_samples.append(diariation_output.label_timeline(speaker))
        diarization_mask = self.get_diarization_mask(per_speaker_samples, sample['input_features'].shape[-1] // 2)
        stno_masks = []
        for i, speaker_samples in enumerate(per_speaker_samples):
            stno_mask = self.get_stno_mask(diarization_mask, i)
            stno_masks.append(stno_mask)
        sample['vad_mask'] = torch.stack(stno_masks, axis=0).to(sample['input_features'].device,
                                                                dtype=sample['input_features'].dtype)
        sample['input_features'] = sample['input_features'].repeat(len(per_speaker_samples), 1, 1)
        sample['attention_mask'] = torch.ones(sample['input_features'].shape[0], sample['input_features'].shape[2],
                                              dtype=torch.bool, device=sample['input_features'].device)
        yield sample

    def _forward(self, model_inputs, return_timestamps=False, **generate_kwargs):
        attention_mask = model_inputs.pop("attention_mask", None)
        stride = model_inputs.pop("stride", None)
        segment_size = model_inputs.pop("segment_size", None)
        is_last = model_inputs.pop("is_last")

        if stride is not None and segment_size is not None:
            raise ValueError("segment_size must be used only when stride is None")

        # Consume values so we can let extra information flow freely through
        # the pipeline (important for `partial` in microphone)
        if "input_features" in model_inputs:
            inputs = model_inputs.pop("input_features")
        elif "input_values" in model_inputs:
            inputs = model_inputs.pop("input_values")
        else:
            raise ValueError(
                "Seq2Seq speech recognition model requires either a "
                f"`input_features` or `input_values` key, but only has {model_inputs.keys()}"
            )

        # custom processing for Whisper timestamps and word-level timestamps
        if return_timestamps and self.type == "seq2seq_whisper":
            generate_kwargs["return_timestamps"] = return_timestamps
            if return_timestamps == "word":
                generate_kwargs["return_token_timestamps"] = True
                generate_kwargs["return_segments"] = True

                if stride is not None:
                    if isinstance(stride, tuple):
                        generate_kwargs["num_frames"] = stride[0] // self.feature_extractor.hop_length
                    else:
                        generate_kwargs["num_frames"] = [s[0] // self.feature_extractor.hop_length for s in stride]

                else:
                    if isinstance(segment_size, int):
                        generate_kwargs["num_frames"] = segment_size // self.feature_extractor.hop_length
                    else:
                        generate_kwargs["num_frames"] = segment_size[0] // self.feature_extractor.hop_length

            generate_kwargs["input_features"] = inputs

        tokens = self.model.generate(
            attention_mask=attention_mask,
            **generate_kwargs,
            **model_inputs,
        )
        # whisper longform generation stores timestamps in "segments"
        if return_timestamps == "word" and self.type == "seq2seq_whisper":
            if "segments" not in tokens:
                out = {"tokens": tokens["sequences"], "token_timestamps": tokens["token_timestamps"]}
            else:
                token_timestamps = [
                    torch.cat([segment["token_timestamps"] for segment in segment_list])
                    for segment_list in tokens["segments"]
                ]
                out = {"tokens": tokens["sequences"], "token_timestamps": token_timestamps}
        else:
            out = {"tokens": tokens}
        if self.type == "seq2seq_whisper":
            if stride is not None:
                out["stride"] = stride

        # Leftover
        extra = model_inputs
        return {"is_last": is_last, **out, **extra}

    def postprocess(
            self, model_outputs, decoder_kwargs: Optional[Dict] = None, return_timestamps=None, return_language=None
    ):
        per_spk_outputs = self.tokenizer.batch_decode(model_outputs[0]['tokens'], decode_with_timestamps=True, skip_special_tokens=True)
        full_text = "\n".join([f"|Speaker {spk}|: {text}" for spk, text in enumerate(per_spk_outputs)])
        return {"text": full_text, "per_spk_outputs": per_spk_outputs}

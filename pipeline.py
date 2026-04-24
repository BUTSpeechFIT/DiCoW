import os
import re
from typing import Dict, Optional, List  # type: ignore

import gradio as gr
import torch
from librosa import load as libr_load
from soundfile import write as sf_write
from transformers.pipelines.automatic_speech_recognition import AutomaticSpeechRecognitionPipeline


def max_ones_window(tensor: torch.Tensor, window_size: int = 30):
    # Create a 1D convolution kernel of ones
    kernel = torch.ones(window_size, dtype=torch.float32, device=tensor.device)

    # Use conv1d: reshape to [N, C, L] format
    # input: [1, 1, L], weight: [1, 1, K]
    input_tensor = tensor.view(1, 1, -1)
    kernel = kernel.view(1, 1, -1)

    # Convolution gives rolling sums (like np.convolve with 'valid')
    window_sums = torch.nn.functional.conv1d(input_tensor, kernel).flatten()

    # Find index of maximum sum
    max_start = torch.argmax(window_sums).item()
    max_sum = window_sums[max_start].item()

    # Extract the slice of the original tensor
    max_slice = tensor[max_start:max_start + window_size]

    return max_start, max_sum, max_slice


class DiCoWPipeline(AutomaticSpeechRecognitionPipeline):
    def __init__(self, *args, diarization_pipeline, **kwargs):
        super().__init__(*args, **kwargs)
        self.diarization_pipeline = diarization_pipeline

    def _sanitize_parameters(self, **kwargs):
        # DiCoW is Whisper-based; parent sets type="ctc" because model_type!="whisper"
        self.type = "seq2seq_whisper"
        # Don't pass Whisper-specific params to parent - they go directly to _forward via generate_kwargs
        # Return empty dicts to let all kwargs flow through to _forward
        return {}, kwargs, {}

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

    def _process_enrollment_sample(self, samples, idx, stno_mask, original_stno_length):
        """Process enrollment sample with padding to match original size."""
        # Find best 30s enrollment window
        enrollment_length = 30 * 50
        best_start, best_sum, _ = max_ones_window(stno_mask[1], window_size=30 * 50)

        # Extract enrollment features
        enrollment_features = samples['input_features'][idx][:, best_start * 2:best_start * 2 + enrollment_length * 2]
        enrollment_attention = samples['attention_mask'][idx][best_start * 2:best_start * 2 + enrollment_length * 2]
        enrollment_stno = stno_mask[:, best_start:best_start + enrollment_length]

        return enrollment_features, enrollment_attention, enrollment_stno

    def preprocess(self, inputs, chunk_length_s=0, stride_length_s=None):
        if not isinstance(inputs, str):
            raise ValueError("For now input must be a string representing a path to an audio file")

        input_dirname = os.path.dirname(inputs)
        resampled_path = f'{input_dirname}/resampled.wav'

        inp_aud, sr = libr_load(inputs, sr=16_000, mono=True)
        sf_write(resampled_path, inp_aud, sr, format='wav')
        inputs = resampled_path

        generator = super().preprocess(inputs, chunk_length_s=chunk_length_s, stride_length_s=stride_length_s)
        samples = next(generator)

        diarization_output = self.diarization_pipeline(inputs)
        per_speaker_samples = []
        for speaker in diarization_output.labels():
            per_speaker_samples.append(diarization_output.label_timeline(speaker))
        diarization_mask = self.get_diarization_mask(per_speaker_samples, samples['input_features'].shape[-1] // 2)
        stno_masks = []
        for i, speaker_samples in enumerate(per_speaker_samples):
            stno_mask = self.get_stno_mask(diarization_mask, i)
            stno_masks.append(stno_mask)
        samples['stno_mask'] = torch.stack(stno_masks, axis=0).to(samples['input_features'].device,
                                                                  dtype=samples['input_features'].dtype)
        samples['input_features'] = samples['input_features'].repeat(len(per_speaker_samples), 1, 1)
        samples['attention_mask'] = torch.ones(samples['input_features'].shape[0], samples['input_features'].shape[2],
                                               dtype=torch.bool, device=samples['input_features'].device)
        if "num_frames" in samples:
            del samples["num_frames"]

        if hasattr(self.model.config, "use_enrollments") and self.model.config.use_enrollments:
            if len(inp_aud) / sr <= 30.0:
                # We are in the shortform regime, we don't want to condition, deactivate enrollments
                gr.Info(
                    "If you are experiencing suboptimal performance, consider using a non–self-enrollment conditioned model (e.g., `BUT-FIT/DiCoW_v3_3`) for inputs shorter than 30s.")

            # Collect all samples (original + enrollment)
            all_input_features = []
            all_attention_masks = []
            all_stno_masks = []

            enroll_input_features = []
            enroll_attention_masks = []
            enroll_stno_masks = []

            original_stno_length = samples['stno_mask'].shape[-1]

            for idx, stno_mask in enumerate(samples['stno_mask']):
                # Add original sample
                all_input_features.append(samples['input_features'][idx])
                all_attention_masks.append(samples['attention_mask'][idx])
                all_stno_masks.append(stno_mask)

                # Add enrollment sample (padded to original size)
                enrollment_features, enrollment_attention, enrollment_stno = self._process_enrollment_sample(
                    samples, idx, stno_mask, original_stno_length
                )
                enroll_input_features.append(enrollment_features)
                enroll_attention_masks.append(enrollment_attention)
                enroll_stno_masks.append(enrollment_stno)

            # Stack all samples
            samples['input_features'] = torch.stack(all_input_features, dim=0)
            samples['attention_mask'] = torch.stack(all_attention_masks, dim=0)
            samples['stno_mask'] = torch.stack(all_stno_masks, dim=0)
            samples["enrollments"] = {
                "input_features": torch.stack(enroll_input_features, dim=0),
                "attention_mask": torch.stack(enroll_attention_masks, dim=0),
                "stno_mask": torch.stack(enroll_stno_masks, dim=0),
            }

        yield samples

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

    @staticmethod
    def postprocess_text(input_string):
        pattern = r"<\|([\d.]+)\|>"
        matches = re.finditer(pattern, input_string)
        timestamps = [(float(match.group(1)), match.start(), match.end()) for match in matches]
        if not timestamps or len(timestamps) <= 2:
            return input_string

        # The whole algorithm boils down to either removing the entire chain of timestamps - the case where all of them are the same (i.e. ...<a><a><a>... -> ......)
        # or removing all but the corner ones (i.e. <a><b><c><c><d> -> <a><d>) - the case where we have end and start timestamps and some rubbish in-between.

        processed_timestamps = []
        i = 0
        while i < len(timestamps):
            ts, st, et = timestamps[i]

            if i < len(timestamps) - 1 or processed_timestamps[-1][-1] != st:
                processed_timestamps.append((ts, st, et))

            if i == len(timestamps) - 1:
                break

            j = i + 1
            nts, nst, net = timestamps[j]
            all_equal_ts = nts == ts
            prev_et = et
            while nst - prev_et == 0:
                # Skip all but the last timestamp. If the last in the chain has the same TS as the processed_timestamps tail, pop processed_timestamps.
                # If not, append it while skipping all the previous ones.
                # In other words, keep appending (-2, X, X) as long as the next one is in the chain and then decide what to do with the last one if the next one is not in the chain.

                if j == len(timestamps) - 1:
                    if net == len(input_string) and prev_et != nst:
                        processed_timestamps.append((nts, nst, net))
                        j += 1
                    break
                else:
                    if timestamps[j + 1][1] - net == 0:
                        processed_timestamps.append((-2, nst, net))
                    else:
                        if all_equal_ts:
                            # If there's a chain of eq timestamps at the beginning, we need to keep at least one.
                            if i != 0:
                                processed_timestamps[i] = (-1, st, et)
                            processed_timestamps.append((-2, nst, net))
                        else:
                            # If there's a chain of tags at the beginning with all ts not being equal, we need to keep the last one.
                            if i == 0:
                                processed_timestamps[i] = (-2, st, et)
                            processed_timestamps.append((nts, nst, net))
                        j += 1
                        break

                j += 1
                prev_et = net
                nts, nst, net = timestamps[j]
                all_equal_ts = all_equal_ts and nts == ts

            i = j

        result = []
        prev_end = 0
        for i, (ts, st, et) in enumerate(processed_timestamps):
            result.append(f'{input_string[prev_end:st]}')
            if ts == -1:
                result.append(' ')
            elif ts == -2:
                # Empty string, so no need to append anything
                pass
            else:
                result.append(f'<|{ts:.2f}|>')
            prev_end = et

        return "".join(result)

    def postprocess(
            self, model_outputs, decoder_kwargs: Optional[Dict] = None, return_timestamps=None, return_language=None
    ):
        per_spk_outputs = self.tokenizer.batch_decode(
            model_outputs[0]['tokens'], decode_with_timestamps=True, skip_special_tokens=True
        )

        formatted_lines = []
        for spk, text in enumerate(per_spk_outputs):
            processed_text = self.postprocess_text(text)

            # Split on each timestamp pair
            # This regex finds "<|start|>...<|end|>" pairs with everything inside
            segments = re.findall(r"(<\|\d+\.\d+\|>.*?<\|\d+\.\d+\|>)", processed_text)

            # Build the output for this speaker
            speaker_header = f"🗣️ Speaker {spk}:\n"
            speaker_body = "\n".join(segments)
            formatted_lines.append(f"{speaker_header}{speaker_body}")

        full_text = "\n\n".join(formatted_lines)

        return {"text": full_text, "per_spk_outputs": per_spk_outputs}

    def transcribe_openai(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe",
        temperature: float = 0.0,
        return_word_timestamps: bool = False,
        diarize: bool = True,
        compression_ratio_threshold: Optional[float] = None,
        logprob_threshold: Optional[float] = None,
        no_speech_threshold: Optional[float] = None,
    ):
        """
        OpenAI-compatible transcription returning clean structured data.

        Uses the raw per-speaker decoded output (before emoji formatting)
        to build OpenAI-compatible segments with clean text.

        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'en'). Auto-detect if None.
            task: 'transcribe' or 'translate' (default: 'transcribe')
            temperature: Sampling temperature (0.0 = greedy)
            return_word_timestamps: Ignored (word-level timestamps not supported by DiCoW)
            diarize: Keep speaker separation (default True for SE-DiCoW)
            compression_ratio_threshold: Hallucination detection threshold
            logprob_threshold: Minimum average log probability
            no_speech_threshold: Silence detection threshold

        Returns:
            dict with text, segments, word_segments, speakers_count, duration, language, metadata
        """
        from librosa import get_duration as librosa_get_duration

        # Always use segment-level timestamps (word-level not supported by DiCoW)
        generate_kwargs = {
            "return_timestamps": True,
            "temperature": temperature,
            "task": task,
        }

        if language:
            generate_kwargs["language"] = language

        if compression_ratio_threshold is not None:
            generate_kwargs["compression_ratio_threshold"] = compression_ratio_threshold
        if logprob_threshold is not None:
            generate_kwargs["logprob_threshold"] = logprob_threshold
        if no_speech_threshold is not None:
            generate_kwargs["no_speech_threshold"] = no_speech_threshold

        # Run pipeline (preprocess → _forward → postprocess)
        result = super().__call__(audio_path, **generate_kwargs)

        duration = librosa_get_duration(filename=audio_path)

        # Use raw per-speaker decoded text (before emoji formatting)
        per_spk_outputs = result.get("per_spk_outputs", [])

        # Parse structured segments from raw decoded output
        segments = self._extract_segments_from_raw(per_spk_outputs)

        # Build clean text: all segments sorted by start time
        sorted_segments = sorted(segments, key=lambda s: s["start"])
        final_text = " ".join(seg["text"] for seg in sorted_segments)

        return {
            "text": final_text,
            "segments": segments,
            "word_segments": None,
            "speakers_count": len(set(seg.get("speaker", 0) for seg in segments)),
            "duration": duration,
            "language": language or "en",
            "metadata": {
                "avg_logprob": -0.5,
                "compression_ratio": 1.0,
                "no_speech_prob": 0.0,
            },
        }

    def _extract_segments_from_raw(self, per_spk_outputs: List[str]) -> List[Dict]:
        """
        Extract structured segments from raw per-speaker decoded text.

        Each per_spk_output is raw tokenizer output like:
        "<|0.00|>to get out of bed on fridays<|1.46|><|1.76|>it might be good...<|5.66|>"

        Returns list of segments with clean text (no timestamp tags).
        """
        segments = []
        segment_pattern = r"<\|(\d+\.\d+)\|>(.*?)<\|(\d+\.\d+)\|>"

        for spk_idx, raw_text in enumerate(per_spk_outputs):
            # Apply timestamp cleanup (removes duplicate/chained timestamps only)
            cleaned = self.postprocess_text(raw_text)

            # Parse <|start|>text<|end|> pairs
            matches = re.findall(segment_pattern, cleaned, re.DOTALL)

            for start_str, text_content, end_str in matches:
                text_clean = " ".join(text_content.split()).strip()
                if text_clean:
                    segments.append({
                        "speaker": spk_idx,
                        "start": float(start_str),
                        "end": float(end_str),
                        "text": text_clean,
                    })

        return segments

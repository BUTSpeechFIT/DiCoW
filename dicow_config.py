from typing import Optional
from transformers import WhisperConfig


class DiCoWConfig(WhisperConfig):
    """This is a modified version of the `WhisperEncoder` model from the `transformers` library.
    The model has been modified to support CTC loss computation in the forward pass."""
    model_type = "dicow"
    def __init__(
            self,
            ctc_loss_reduction: str = "mean",
            final_dropout: float = 0.0,
            ctc_zero_infinity: bool = False,
            ctc_weight: float = 0.0,
            blank_token_id: Optional[int] = None,
            additional_layer: bool = False,
            additional_self_attention_layer: bool = False,
            sub_sample: bool = False,
            use_target_amplifiers: bool = True,
            target_amp_is_diagonal: bool = True,
            target_amp_bias_only: bool = False,
            target_amp_use_silence: bool = True,
            target_amp_use_target: bool = True,
            target_amp_use_overlap: bool = True,
            target_amp_use_non_target: bool = True,
            remove_timestamps_from_ctc: bool = False,
            apply_target_amp_to_n_layers: int = -1,
            target_amp_init: str = 'non-disturbing',  # random, non-disturbing, dispargement
            n_soft_prompts: int = 16,
            mt_num_speakers: int = 1,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.ctc_loss_reduction = ctc_loss_reduction
        self.final_dropout = final_dropout
        self.ctc_zero_infinity = ctc_zero_infinity
        self.ctc_weight = ctc_weight
        self.blank_token_id = blank_token_id
        self.additional_layer = additional_layer
        self.additional_self_attention_layer = additional_self_attention_layer
        self.sub_sample = sub_sample
        self.use_target_amplifiers = use_target_amplifiers
        self.target_amp_is_diagonal = target_amp_is_diagonal
        self.target_amp_bias_only = target_amp_bias_only
        self.target_amp_use_silence = target_amp_use_silence
        self.target_amp_use_target = target_amp_use_target
        self.target_amp_use_overlap = target_amp_use_overlap
        self.target_amp_use_non_target = target_amp_use_non_target
        self.remove_timestamps_from_ctc = remove_timestamps_from_ctc
        self.apply_target_amp_to_n_layers = apply_target_amp_to_n_layers
        self.target_amp_init = target_amp_init
        self.n_soft_prompts = n_soft_prompts
        self.mt_num_speakers = mt_num_speakers

import torch
from torch import nn
from transformers import WhisperConfig
from transformers.activations import ACT2FN
from transformers.models.whisper.modeling_whisper import WHISPER_ATTENTION_CLASSES
import torch.nn.functional as F
from .coattention import CoAttention
from .layers import CustomLinear, CustomDiagonalLinear, Gate

class LowRankApproxSelectFirst(nn.Module):
    def __init__(self, d_in, d_out, rank):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.rank = rank
        self.proj_in = nn.Linear(d_in, rank)
        self.proj_out = nn.Linear(rank, d_out)

    def forward(self, x):
        return self.proj_out(self.proj_in(x))

    def _init_weights(self):
        # Create low-rank approximation of the identity projection from first d_out of input
        eye = torch.eye(self.d_out, self.d_in)  # (d_out x d_in)

        # Low-rank SVD of eye matrix
        U, S, Vh = torch.linalg.svd(eye, full_matrices=False)  # U: (d_out x d_out), Vh: (d_in x d_in)

        U_k = U[:, :self.rank]              # (d_out x rank)
        S_k = S[:self.rank]                 # (rank,)
        V_k = Vh[:self.rank, :]             # (rank x d_in)

        A = V_k                             # (rank x d_in)
        B = U_k @ torch.diag(S_k)           # (d_out x rank)

        # Set weights
        self.proj_in.weight.data.copy_(A)
        self.proj_in.bias.data.zero_()
        self.proj_out.weight.data.copy_(B)
        self.proj_out.bias.data.zero_()



class TACBlock(nn.Module):
    def __init__(self, config: WhisperConfig, d_int_factor: float = 1, num_speakers=2):
        super().__init__()
        d = config.d_model
        d_prime = int(d * d_int_factor)
        self.num_speakers = num_speakers
        self.proj_in_1 = nn.Linear(d, d_prime, bias=True)
        self.proj_in_2 = nn.Linear(d, d_prime, bias=True)
        self.proj_int = nn.Linear(d_prime, d_prime,bias=True)
        self.proj_out_1 = nn.Linear(d+d_prime, d,bias=True)
        self.proj_out_2 = nn.Linear(d+d_prime, d,bias=True)
        self.activation_fn = ACT2FN[config.activation_function]
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(self.num_speakers)])
        self.gate = Gate(self.num_speakers, 0.01)

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        # hidden_states: (B, self.num_speakers, T, F)

        x_proj = torch.stack([self.activation_fn(self.proj_in_1(hidden_states[:,0])), self.activation_fn(self.proj_in_2(hidden_states[:, 1]))], dim=1)  # (B, 2, T, d')
        x_mean = x_proj.mean(dim=1, keepdim=True)  # (B, 1, T, d')
        z = self.activation_fn(self.proj_int(x_mean))  # (B, 1, T, d')

        z_expand = z.expand(-1, self.num_speakers, -1, -1)  # (B, self.num_speakers, T, d')
        x_cat = torch.cat([hidden_states, z_expand], dim=-1)  # (B, self.num_speakers, T, d + d')
        x_out = torch.stack([self.norms[0](self.proj_out_1(x_cat[:, 0])), self.norms[1](self.proj_out_2(x_cat[:, 1]))], dim=1)  # (B, self.num_speakers, T, d)
        return hidden_states + self.gate(x_out, dim=1)


class CrossAttentionBlock(nn.Module):
    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.num_speakers = getattr(config, "mt_num_speakers", 2)
        if self.num_speakers != 2:
            raise ValueError("CrossAttentionBlock supports only 2 speakers.")

        # Separate attention block per speaker
        self.attn_blocks = nn.ModuleList([
            WHISPER_ATTENTION_CLASSES[config._attn_implementation](
                embed_dim=self.embed_dim,
                num_heads=config.encoder_attention_heads,
                dropout=config.attention_dropout,
                config=config,
            )
            for _ in range(self.num_speakers)
        ])

        self.norms = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(self.num_speakers)])
        self.gate = Gate(self.num_speakers, 0.01)

    def forward(self, hidden_states):
        # hidden_states: (B, 2, T, F)
        outputs = []
        for s in range(self.num_speakers):
            q = hidden_states[:, s]  # (B, T, F)
            other_s = 1 - s
            kv = hidden_states[:, other_s]  # (B, T, F)

            attn_out, _, _ = self.attn_blocks[s](hidden_states=q, key_value_states=kv)  # (B, T, F)
            outputs.append(self.norms[s](attn_out[:, None, :, :]))
        outputs =  torch.concat(outputs, dim=1)
        outputs_modulated = self.gate(outputs, dim=1) + hidden_states
        return outputs_modulated


class CompetitiveCrossAttentionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        assert (
                self.head_dim * self.num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.num_speakers = getattr(config, "mt_num_speakers", 2)
        if self.num_speakers != 2:
            raise ValueError("CompetitiveCrossAttentionBlock supports only 2 speakers.")

        # Separate projections for Q, K, V
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.norms = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(self.num_speakers)])
        self.eps = 1e-6
        self.gate = Gate(self.num_speakers, 0.01)

    def _shape(self, tensor, seq_len, batch_size):
        # reshape into (B, num_heads, T, head_dim)
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, hidden_states):
        # hidden_states: (B, 2, T, F)
        B, _, T, _ = hidden_states.shape

        h1, h2 = hidden_states[:, 0], hidden_states[:, 1]  # (B, T, F)

        # Project Q,K,V
        Q1 = self.q_proj(h1)  # (B, T, F)
        K2 = self.k_proj(h2)
        V2 = self.v_proj(h2)

        Q2 = self.q_proj(h2)
        K1 = self.k_proj(h1)
        V1 = self.v_proj(h1)

        # Reshape for multi-head attention
        Q1 = self._shape(Q1, T, B)  # (B, heads, T, head_dim)
        K2 = self._shape(K2, T, B)
        V2 = self._shape(V2, T, B)

        Q2 = self._shape(Q2, T, B)
        K1 = self._shape(K1, T, B)
        V1 = self._shape(V1, T, B)

        # Scaled dot-product attention logits
        scale = 1 / (self.head_dim ** 0.5)
        L_1to2 = torch.matmul(Q1, K2.transpose(-1, -2)) * scale  # (B, heads, T, T)
        L_2to1 = torch.matmul(Q2, K1.transpose(-1, -2)) * scale  # (B, heads, T, T)

        # Softmax over last dim (keys)
        S_1to2 = F.softmax(L_1to2, dim=-1)
        S_2to1 = F.softmax(L_2to1, dim=-1)

        # Competitive normalization (soft exclusivity)
        M_joint = S_1to2 + S_2to1 + self.eps
        A_1to2 = S_1to2 / M_joint
        A_2to1 = S_2to1 / M_joint

        # Weighted sum of values
        H1_attn = torch.matmul(A_1to2, V2)  # (B, heads, T, head_dim)
        H2_attn = torch.matmul(A_2to1, V1)

        # Concatenate heads back
        H1_attn = H1_attn.transpose(1, 2).contiguous().view(B, T, self.embed_dim)  # (B, T, F)
        H2_attn = H2_attn.transpose(1, 2).contiguous().view(B, T, self.embed_dim)

        # Output projection
        H1_attn = self.norms[0](self.out_proj(H1_attn))
        H2_attn = self.norms[1](self.out_proj(H2_attn))

        # Residuals
        out = hidden_states + self.gate(torch.concat([H1_attn[:, None, :, :], H2_attn[:, None, :, :]], dim=1), dim=1)

        return out # (B, 2, T, F)


class CoAttentionWrapper(nn.Module):
    def __init__(self, config, num_speakers=2):
        super().__init__()
        self.coa = CoAttention(embed_dim=config.d_model, single_dim=config.d_model//2, multi_dim=config.d_model // 4, n_heads=config.encoder_attention_heads, attn_dropout=config.attention_dropout)
        self.gate = Gate(num_speakers, 0.01)

    def forward(self, coa_input: torch.Tensor) -> torch.Tensor:
        # hidden_states: (B, 2, T, F)
        hidden_states = coa_input.permute(-2, 0, 1, -1)
        hidden_states = self.coa(hidden_states)
        out = coa_input + self.gate(hidden_states.permute(1, 2, 0, -1), dim=1)
        return out


class SpeakerCommunicationBlock(nn.Module):
    def __init__(self, config, scb_method):
        super().__init__()
        self.num_speakers = getattr(config, "mt_num_speakers", 2)
        self.embed_dim = config.d_model
        self.scb_method = scb_method
        self.config = config

        if self.scb_method == "tac":
            self.method = TACBlock(config)
        elif self.scb_method == "cross_attention":
            self.method = CrossAttentionBlock(config)
        elif self.scb_method == "competitive_cross_attention":
            self.method = CompetitiveCrossAttentionBlock(config)
        elif self.scb_method == "co_attention":
            self.method = CoAttentionWrapper(config)
        elif self.scb_method == "identity":
            self.method = (nn.Parameter(torch.zeros(self.embed_dim)) if config.fddt_bias_only else (
                CustomDiagonalLinear(self.embed_dim, bias=True, init_eye_val=1.0) if config.fddt_is_diagonal else CustomLinear(
                    self.embed_dim, self.embed_dim, bias=True, init_eye_val=1.0)))
        else:
            raise ValueError(f"Unsupported scb_method: {self.scb_method}")

    def forward(self, x):
        # x: (B, T, F)
        B, T, F = x.shape
        S = self.num_speakers

        # Reshape to (B//S, S, T, F)
        x_reshaped = x.view(B//S, S, T, F)

        # Call the selected method
        out = self.method(x_reshaped)

        # Reshape back (B, T, F)
        out_merged = out.view(B, T, F)
        return out_merged

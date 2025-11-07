import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings
from .components import MixtureOfExperts
from xformers.ops import memory_efficient_attention, LowerTriangularMask


class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        self.rope = RotaryPositionalEmbeddings(dim=dim, max_seq_len=max_seq_len, base=10000)

    def forward(self, x_BTHD: torch.Tensor):
        # x_BTHD shape: [B, T, H, D] - need to convert to [B, T, H, D] for torchtune
        # torchtune expects [batch, seq_len, num_heads, head_dim]
        # Our input is already [B, T, H, D] which matches torchtune's expectation
        return self.rope(x_BTHD)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        kv_heads: int,
        max_seq_len: int,
        dropout: float = 0.1,
        use_mem_atten: bool = False,
    ):
        super().__init__()
        self.mem_atten = use_mem_atten
        self.d_model = d_model
        self.n_heads = n_heads
        self.kv_heads = kv_heads
        self.d_k = d_model // n_heads
        assert self.d_k * n_heads == d_model, "d_model must be divisible by n_heads."
        self.n_rep = n_heads // kv_heads

        self.qkv = nn.Linear(d_model, d_model + (kv_heads * self.d_k * 2), bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.rotary = Rotary(self.d_k, max_seq_len)
        self.dropout = dropout

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        # B, T = x.size(0), x.size(1)
        # qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_k).permute(2, 0, 3, 1, 4)
        # Q, K, V = qkv[0], qkv[1], qkv[2]  # [B, H, T, D]

        qkv = self.qkv(x)
        Q, K, V = qkv.split(
            (self.d_model, self.kv_heads * self.d_k, self.kv_heads * self.d_k), dim=-1
        )

        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k)
        K = K.view(batch_size, seq_len, self.kv_heads, self.d_k)
        V = V.view(batch_size, seq_len, self.kv_heads, self.d_k)

        # Q = self.rotary(Q)
        # K = self.rotary(K)
        Q = self.rotary(Q)
        K = self.rotary(K)

        K = torch.repeat_interleave(K, self.n_rep, dim=2)
        V = torch.repeat_interleave(V, self.n_rep, dim=2)

        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)

        if self.mem_atten:
            attn_output = memory_efficient_attention(
                Q, K, V, attn_bias=LowerTriangularMask(), p=self.dropout
            )
        else:
            attn_output = F.scaled_dot_product_attention(
                Q, K, V, is_causal=True, dropout_p=self.dropout
            )
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, seq_len, self.d_model
        )
        # attn_output = attn_output.transpose(1, 2).reshape(B, T, self.d_model)
        return self.w_o(attn_output)


class MoETransformerBlock(nn.Module):
    """Transformer block with MoE"""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        kv_heads: int,
        d_ff: int,
        max_seq_len: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
        use_mem_atten: bool = False,
    ):
        super().__init__()

        # Attention layer
        self.attention = MultiHeadAttention(
            d_model, n_heads, kv_heads, max_seq_len, dropout, use_mem_atten
        )

        # MoE layer
        self.feed_forward = MixtureOfExperts(d_model, d_ff, num_experts, top_k, dropout)

        # Normalization layers
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)

        # MoE feed-forward
        ff_out, aux_loss = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        return x, aux_loss

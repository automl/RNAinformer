#Ref:https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Union

#https://github.com/automl-private/RNAformer_private/blob/dev_rnaformer2/RNAformer/module/attention.py
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, end: int, theta: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.end = end
        self.theta = theta
        self.freqs_cis: torch.Tensor
        self._initialized_buffer = False

    def init_rotary_embeddings(self, device):
        if self._initialized_buffer is True:
            # Buffer if already initialized
            return
        self.register_buffer(
            "freqs_cis",
            torch.empty(self.end, self.dim // 2, 2, dtype=torch.float, device=device),
            persistent=False,
        )
        if self.freqs_cis.dtype != torch.float:
            self.freqs_cis = self.freqs_cis.to(torch.float)
        assert self.freqs_cis.dtype == torch.float
        freqs = 1.0 / (
                self.theta
                ** (torch.arange(0, self.dim, 2, dtype=torch.float, device=device)[: (self.dim // 2)] / self.dim)
        )
        t = torch.arange(self.end, device=device)
        freqs = torch.outer(t, freqs).float()
        complex_freqs = torch.polar(torch.ones_like(freqs), freqs)
        freqs = torch.view_as_real(complex_freqs)
        self.freqs_cis.copy_(freqs)
        self._initialized_buffer = True

    def forward(
            self,
            x: torch.Tensor,  # [batch_size, seq_length, num_heads, d_qk]
            position_ids: Optional[torch.LongTensor],  # [batch_size, seq_length]
    ):
        batch_size, seq_length, num_heads, inner_dim = x.shape
        while (
                position_ids is not None and position_ids[-1, -1] >= self.end
        ) or seq_length >= self.end:
            self.end *= 2
            self._initialized_buffer = False
        if self._initialized_buffer is False:
            self.init_rotary_embeddings(device=x.device)
        dtype = x.dtype
        assert inner_dim % 2 == 0
        x = x.view(
            batch_size, seq_length, num_heads, inner_dim // 2, 2
        )  # [batch_size, q_length, num_heads, inner_dim]
        if x.dtype == torch.bfloat16:
            x = x.float()
        complex_x = torch.view_as_complex(x)  # [batch_size, q_length, num_heads, inner_dim // 2]
        if position_ids is None:
            freqs_cis = self.freqs_cis[None, :seq_length, None, :]
        else:
            if position_ids[-1, -1] < 0 or position_ids[-1, -1] >= self.end:  # Quick test hopefully
                raise ValueError(f"Position ids must be in the range [0, {self.end}), but got {position_ids}")
            freqs_cis = self.freqs_cis[position_ids][:, :, None, :]
        complex_freqs = torch.view_as_complex(freqs_cis)
        x_out = torch.view_as_real(complex_x * complex_freqs).view(batch_size, seq_length, num_heads, inner_dim)
        return x_out.type(dtype)

def expand_mask(mask):
    assert mask.ndim > 2, "Mask must be at least 2-dimensional with seq_length x seq_length"
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    while mask.ndim < 4:
        mask = mask.unsqueeze(0)
    return mask

class MultiheadSelfAttention(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        
        self.p_dropout = dropout

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, _ = x.size()
        if mask is not None:
            mask = mask.unsqueeze(1)
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)
 
        # Determine value outputs
        with torch.backends.cuda.sdp_kernel(enable_flash=True):
            values = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.p_dropout, scale=self.scale)            
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)

        return o

class MultiheadCrossAttention(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads,dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.q_proj = nn.Linear(input_dim, embed_dim)
        self.kv_proj = nn.Linear(input_dim, 2*embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        self.p_dropout = dropout
        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.kv_proj.weight)
        self.kv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, enc_out, mask=None, return_attention=False):
        batch_size, seq_length, _ = x.size()
        
        if mask is not None:
            mask = mask.unsqueeze(1)
        q = self.q_proj(x)
        kv = self.kv_proj(enc_out)

        q = q.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)
        # Separate K, V from linear output
        kv = kv.reshape(batch_size, enc_out.shape[1], self.num_heads, 2*self.head_dim)
        kv = kv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        k, v = kv.chunk(2, dim=-1)

        # Determine value outputs
        with torch.backends.cuda.sdp_kernel(enable_flash=True):
            values = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,dropout_p=self.p_dropout, scale=self.scale)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)

        return o
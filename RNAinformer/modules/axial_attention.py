import numpy as np
import torch
import math
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import warnings
from termcolor import colored
import torch.nn.functional as F
from einops import rearrange, repeat

def is_package_installed(package_name):
    import pkg_resources
    installed_packages = {pkg.key for pkg in pkg_resources.working_set}
    return package_name in installed_packages

if is_package_installed('flash-attn'):
    from flash_attn.bert_padding import unpad_input, pad_input
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
    
#from rotary_embedding_torch import apply_rotary_emb, RotaryEmbedding, broadcat
from RNAinformer.modules.attention import RotaryEmbedding

class Attention2d(nn.Module):
    def __init__(self, model_dim, num_head, softmax_scale,
                 precision, zero_init, use_bias,
                 initializer_range, n_layers):
        super().__init__()
        assert model_dim % num_head == 0
        assert model_dim % num_head == 0
        self.key_dim = model_dim // num_head
        self.value_dim = model_dim // num_head

        if softmax_scale:
            self.softmax_scale = torch.sqrt(torch.FloatTensor([self.key_dim]))
        else:
            self.softmax_scale = False

        self.num_head = num_head
        self.model_dim = model_dim

        if precision == "fp32" or precision == 32 or precision == "bf16":
            self.mask_bias = -1e9
        elif precision == "fp16" or precision == 16:
            self.mask_bias = -1e4
        else:
            raise UserWarning(f"unknown precision: {precision} . Please us fp16, fp32 or bf16")

        self.Wqkv = nn.Linear(model_dim, 3 * model_dim, bias=use_bias)
        self.out_proj = nn.Linear(model_dim, model_dim, bias=use_bias)

        self.initialize(zero_init, use_bias, initializer_range, n_layers)

    def initialize(self, zero_init, use_bias, initializer_range, n_layers):

        nn.init.normal_(self.Wqkv.weight, mean=0.0, std=initializer_range)

        if use_bias:
            nn.init.constant_(self.Wqkv.bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)

        if zero_init:
            nn.init.constant_(self.out_proj.weight, 0.0)
        else:
            nn.init.normal_(self.out_proj.weight, mean=0.0, std=initializer_range / math.sqrt(2 * n_layers))

    def forward(self, pair_act, attention_mask):

        batch_size = pair_act.size(0)
        N_seq = pair_act.size(1)
        N_res = pair_act.size(2)

        query, key, value = self.Wqkv(pair_act).split(self.model_dim, dim=3)
        query = query.view(batch_size, N_seq, N_res, self.num_head, self.key_dim).permute(0, 1, 3, 2, 4)
        key = key.view(batch_size, N_seq, N_res, self.num_head, self.value_dim).permute(0, 1, 3, 4, 2)
        value = value.view(batch_size, N_seq, N_res, self.num_head, self.value_dim).permute(0, 1, 3, 2, 4)
        attn_weights = torch.matmul(query, key)

        if self.softmax_scale:
            attn_weights = attn_weights / self.softmax_scale.to(pair_act.device)

        if attention_mask is not None:
            attention_mask = attention_mask[:, :, None, None, :]
            attn_weights.masked_fill_(attention_mask, self.mask_bias)
        attn_weights = F.softmax(attn_weights, dim=-1)

        weighted_avg = torch.matmul(attn_weights, value).permute(0, 1, 3, 2, 4)

        output = self.out_proj(weighted_avg.reshape(batch_size, N_seq, N_res, self.num_head * self.value_dim))
        return output


class TriangleAttention(nn.Module):
    def __init__(self, model_dim, num_head, orientation, softmax_scale,
                 precision, zero_init, use_bias, flash_attn,
                 initializer_range, n_layers):
        super().__init__()

        self.model_dim = model_dim
        self.num_head = num_head

        assert orientation in ['per_row', 'per_column']
        self.orientation = orientation

        self.input_norm = nn.LayerNorm(model_dim, eps=1e-6)

        self.attn = Attention2d(model_dim, num_head, softmax_scale,
                                    precision, zero_init, use_bias, initializer_range, n_layers)

    def forward(self, pair_act, pair_mask, cycle_infer=False):

        assert len(pair_act.shape) == 4

        if self.orientation == 'per_column':
            pair_act = torch.swapaxes(pair_act, -2, -3)
            if pair_mask is not None:
                pair_mask = torch.swapaxes(pair_mask, -1, -2)

        pair_act = self.input_norm(pair_act)

        if self.training and not cycle_infer:
            pair_act = checkpoint(self.attn, pair_act, pair_mask, use_reentrant=True)
        else:
            pair_act = self.attn(pair_act, pair_mask)

        if self.orientation == 'per_column':
            pair_act = torch.swapaxes(pair_act, -2, -3)

        return pair_act

class AxialAttention(nn.Module):
    def __init__(self, embed_dim, num_head, use_bias, softmax_scale, dropout,
                 max_position_embeddings, orientation) -> None:
        super().__init__()

        self.embed_dim = embed_dim

        self.dropout = dropout

        assert orientation in ['per_row', 'per_column']
        self.orientation = orientation

        self.num_head = num_head
        assert self.embed_dim % num_head == 0, "self.kdim must be divisible by num_head"
        self.head_dim = self.embed_dim // num_head
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"

        if softmax_scale:
            self.scale = self.head_dim ** (-0.5)
        else:
            self.scale = None

        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=use_bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)



        # self.rotary_embedding = RotaryEmbedding(
        #     dim=embed_dim // num_head,
        #     cache_if_possible=False,
        #     seq_before_head_dim=True,
        # )
        self.rotary_embedding = RotaryEmbedding(
            dim=embed_dim // num_head,
            end=max_position_embeddings,
        )

    def forward(self, pair_act, pair_mask):

        assert len(pair_act.shape) == 4

        if self.orientation == 'per_column':
            pair_act = torch.swapaxes(pair_act, -2, -3)
            if pair_mask is not None:
                pair_mask = torch.swapaxes(pair_mask, -1, -2)

        batch_size = pair_act.shape[0]
        seqlenX = pair_act.shape[1]
        seqlenY = pair_act.shape[2]
        extended_batch_size = batch_size * seqlenX


        query, key, value = self.Wqkv(pair_act).chunk(3, dim=-1)

        query = query.contiguous().view(extended_batch_size, seqlenY, self.num_head, self.head_dim)
        key = key.contiguous().view(extended_batch_size, seqlenY, self.num_head, self.head_dim)
        query = self.rotary_embedding(query, position_ids=None)
        key = self.rotary_embedding(key, position_ids=None)
        query = query.view(extended_batch_size, seqlenY, -1)
        key = key.view(extended_batch_size, seqlenY, -1)

        value = value.contiguous().view(extended_batch_size, seqlenY, -1)
        pair_mask = pair_mask.contiguous().view(extended_batch_size, seqlenY)
        full_pair_mask_id = torch.where(pair_mask.sum(-1) == 0)
        attn_mask = pair_mask.clone().unsqueeze(1).repeat(1, seqlenY, 1)
        attn_mask[full_pair_mask_id] = True


        if self.training:
            dropout_p = self.dropout
        else:
            dropout_p = 0


        with torch.backends.cuda.sdp_kernel(enable_flash=True):
            attn_out = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=False, scale=self.scale)

        attn_out[full_pair_mask_id] = 0

        attn_out = attn_out.view(batch_size, seqlenX, seqlenY, self.num_head * self.head_dim)

        pair_act = self.out_proj(attn_out)

        if self.orientation == 'per_column':
            pair_act = torch.swapaxes(pair_act, -2, -3)

        return pair_act





class AxialFlashAttention(nn.Module):
    def __init__(self, model_dim, config, orientation):
        super().__init__()

        self.model_dim = model_dim
        self.num_head = config.num_head
        assert self.model_dim % config.num_head == 0
        self.head_dim = self.model_dim // config.num_head

        self.use_flash_attn = True # TODO #config.flash_attn

        assert orientation in ['per_row', 'per_column']
        self.orientation = orientation

        if config.softmax_scale:
            self.softmax_scale = self.head_dim ** (-0.5)
        else:
            self.softmax_scale = None

        # precision = str(config.precision)
        # if "32" in config.precision or "bf16" in config.precision:
        self.mask_bias = -1e9
        # elif "16" in config.precision:
        #     self.mask_bias = -1e4
        # else:
        #     raise UserWarning(f"Unknown precision: {precision} . Please us fp16, fp32 or bf16")

        # rotary_emb_dim = int(config.rotary_emb_fraction * self.head_dim)
        # self.rotary_emb = RotaryEmbedding(rotary_emb_dim, )

        # self.rotary_emb = RotaryEmbedding(
        #     dim=config.model_dim // config.num_head,
        #     max_freq=config.max_len,
        #     freqs_for='pixel',
        #     # end=config.max_len,
        # )
        if config.get('rotary_emb', False):
            self.rotary_emb = RotaryEmbedding(
                dim=self.model_dim // config.num_head,
                end=config.max_len,
            )
        else:
            self.rotary_emb = None

        # rotary_emb_dim = int(config.rotary_emb_fraction * self.head_dim)
        # rotary_emb_base = getattr(config, 'rotary_emb_base', 10000.0)
        # rotary_emb_scale_base = getattr(config, 'rotary_emb_scale_base', None)
        # rotary_emb_interleaved = getattr(config, 'rotary_emb_interleaved', False)
        #
        # self.rotary_emb = RotaryEmbedding(
        #     rotary_emb_dim,
        #     base=rotary_emb_base,
        #     scale_base=rotary_emb_scale_base,
        #     interleaved=rotary_emb_interleaved,
        #     device=None,
        #     pos_idx_in_fp32=False,
        # )


        self.Wqkv = nn.Linear(self.model_dim, 3 * self.model_dim, bias=config.use_bias)
        self.out_proj = nn.Linear(self.model_dim, self.model_dim, bias=config.use_bias)

    #     self.initialize(config.zero_init, config.use_bias, config.initializer_range, config.n_layers)
    #
    # def initialize(self, zero_init, use_bias, initializer_range, n_layers):
    #
    #     nn.init.normal_(self.Wqkv.weight, mean=0.0, std=initializer_range)
    #
    #     if use_bias:
    #         nn.init.constant_(self.Wqkv.bias, 0.0)
    #         nn.init.constant_(self.out_proj.bias, 0.0)
    #
    #     if zero_init:
    #         nn.init.constant_(self.out_proj.weight, 0.0)
    #     else:
    #         nn.init.normal_(self.out_proj.weight, mean=0.0, std=initializer_range / math.sqrt(2 * n_layers))

    @staticmethod
    # @torch.jit.script
    def _multihead_attn(query, key, value, attention_mask, mask_bias, softmax_scale, batch_size, seqlen, num_head, head_dim):

        query = query.view(batch_size, seqlen, seqlen, num_head, head_dim).permute(0, 1, 3, 2, 4)
        key = key.view(batch_size, seqlen, seqlen, num_head, head_dim).permute(0, 1, 3, 4, 2)
        value = value.view(batch_size, seqlen, seqlen, num_head, head_dim).permute(0, 1, 3, 2, 4)

        attn_weights = torch.matmul(query, key)

        if softmax_scale:
            attn_weights = attn_weights / softmax_scale

        if attention_mask is not None:
            attention_mask = attention_mask[:, :, None, None, :]
            attn_weights.masked_fill_(attention_mask, mask_bias)
        attn_weights = F.softmax(attn_weights, dim=-1)

        weighted_avg = torch.matmul(attn_weights, value).permute(0, 1, 3, 2, 4)
        output = weighted_avg.reshape(batch_size, seqlen, seqlen, num_head * head_dim)

        return output


    def forward(self, pair_act, pair_mask):

        assert len(pair_act.shape) == 4

        if self.orientation == 'per_column':
            pair_act = torch.swapaxes(pair_act, -2, -3)
            if pair_mask is not None:
                pair_mask = torch.swapaxes(pair_mask, -1, -2)

        batch_size = pair_act.shape[0]
        seqlen = pair_act.shape[1]
        extended_batch_size = batch_size * seqlen

        query, key, value = self.Wqkv(pair_act).split(self.model_dim, dim=3)

        # freqs_h = self.rotary_emb(torch.linspace(-1, 1, steps=seqlen).to(pair_act.device).to(query.dtype), seq_len=seqlen)
        # freqs_w = self.rotary_emb(torch.linspace(-1, 1, steps=seqlen).to(pair_act.device).to(query.dtype), seq_len=seqlen)
        # freqs = broadcat((freqs_h[:, None, :], freqs_w[None, :, :]), dim=-1)


        # # new rot
        # freqs = self.rotary_emb.get_axial_freqs(seqlen, seqlen)
        #
        # freqs = freqs.to(query.dtype).to(pair_act.device)
        #
        # query = apply_rotary_emb(freqs, query)
        # key = apply_rotary_emb(freqs, key)

        # freqs_h = self.rotary_emb(torch.linspace(-1, 1, steps=seqlen).to(pair_act.device), cache_key=seqlen)
        # freqs_w = self.rotary_emb(torch.linspace(-1, 1, steps=seqlen).to(pair_act.device), cache_key=seqlen)
        # freqs = broadcat((freqs_h[:, None, :], freqs_w[None, :, :]), dim=-1)
        #
        # query = apply_rotary_emb(freqs, query)
        # key = apply_rotary_emb(freqs, key)


        query = rearrange(query, 'b s f (h d) -> (b s) f h d', b=batch_size, f=seqlen, s=seqlen, h=self.num_head, d=self.head_dim)
        key = rearrange(key, 'b s f (h d) -> (b s) f h d', b=batch_size, f=seqlen, s=seqlen, h=self.num_head, d=self.head_dim)
        # query = query.contiguous().view(extended_batch_size, seqlen, self.num_head, self.head_dim)
        # key = key.contiguous().view(extended_batch_size, seqlen, self.num_head, self.head_dim)
        
        if self.rotary_emb is not None:
            query = self.rotary_emb(query, position_ids=None)
            key = self.rotary_emb(key, position_ids=None)
        
        # query = query.view(batch_size, seqlen, seqlen,  self.num_head * self.head_dim)
        # key = key.view(batch_size, seqlen, seqlen, self.num_head * self.head_dim)
        query = rearrange(query, '(b s) f h d -> b s f (h d)', b=batch_size, f=seqlen, s=seqlen, h=self.num_head, d=self.head_dim)
        key = rearrange(key, '(b s) f h d -> b s f (h d)', b=batch_size, f=seqlen, s=seqlen, h=self.num_head, d=self.head_dim)

        # query = self.rotary_emb(query, position_ids=None)
        # key = self.rotary_emb(key, position_ids=None)
        # # query = query.view(batch_size, seqlen, seqlen,  self.num_head * self.head_dim)
        # # key = key.view(batch_size, seqlen, seqlen, self.num_head * self.head_dim)
        # query = rearrange(query, '(b s) f h d -> b s f (h d)', b=batch_size, f=seqlen, s=seqlen, h=self.num_head, d=self.head_dim)
        # key = rearrange(key, '(b s) f h d -> b s f (h d)', b=batch_size, f=seqlen, s=seqlen, h=self.num_head, d=self.head_dim)

        qkv = torch.cat((query, key, value), dim=-1)
        not_attention_mask = torch.logical_not(pair_mask)

        x_qkv = rearrange(qkv, 'b s f ... -> (b s) f ...', b=batch_size, f=seqlen, s=seqlen)
        key_padding_mask = rearrange(not_attention_mask, 'b s f ... -> (b s) f ...', b=batch_size, f=seqlen, s=seqlen)

        x_unpad, indices, cu_seqlens, max_s = unpad_input(x_qkv, key_padding_mask)
        x_unpad = rearrange(x_unpad, 'nnz (three h d) -> nnz three h d', three=3, h=self.num_head)

        output_unpad = flash_attn_varlen_qkvpacked_func(
            x_unpad, cu_seqlens, max_s, 0.0,
            softmax_scale=self.softmax_scale, causal=False
        )

        pre_pad_latent = rearrange(output_unpad, 'nnz h d -> nnz (h d)')
        padded_latent = pad_input(pre_pad_latent, indices, extended_batch_size, seqlen)
        output = rearrange(padded_latent, 'b f (h d) -> b f h d', h=self.num_head)

        output = rearrange(output, '(b s) f h d -> b s f (h d)', b=batch_size, f=seqlen, s=seqlen)

        #
        # qkv = torch.cat((query, key, value), dim=-1)
        #
        #
        # not_attention_mask = torch.logical_not(pair_mask)
        #
        # x_qkv = rearrange(qkv, 'b s f ... -> (b s) f ...', b=batch_size, f=seqlen, s=seqlen)
        #
        # key_padding_mask = rearrange(not_attention_mask, 'b s f ... -> (b s) f ...', b=batch_size, f=seqlen, s=seqlen)
        #
        # x_unpad, indices, cu_seqlens, max_s = unpad_input(x_qkv, key_padding_mask)
        # x_unpad = rearrange(x_unpad, 'nnz (three h d) -> nnz three h d', three=3, h=self.num_head)
        #
        #
        # output_unpad = flash_attn_varlen_qkvpacked_func(
        #     x_unpad, cu_seqlens, max_s, 0.0,
        #     softmax_scale=self.softmax_scale, causal=False
        # )
        #
        # pre_pad_latent = rearrange(output_unpad, 'nnz h d -> nnz (h d)')
        # padded_latent = pad_input(pre_pad_latent, indices, extended_batch_size, seqlen)
        # output = rearrange(padded_latent, '(b s) f hd -> b s f hd', b=batch_size, f=seqlen, s=seqlen)






        # pytroch attention

        # query = rearrange(query, 'b s f ... -> (b s) f ...', b=batch_size, f=seqlen, s=seqlen)
        # key = rearrange(key, 'b s f ... -> (b s) f ...', b=batch_size, f=seqlen, s=seqlen)
        # value = rearrange(value, 'b s f ... -> (b s) f ...', b=batch_size, f=seqlen, s=seqlen)
        # # pair_mask = rearrange(pair_mask, 'b s f ... -> (b s) f ...', b=batch_size, f=seqlen, s=seqlen)
        #
        # # attn_mask = pair_mask.unsqueeze(-1).repeat(1, 1, seqlen)
        #
        # output = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None)
        #
        # # output = torch.masked_fill(output, torch.logical_not(pair_mask).unsqueeze(-1), 0)
        #
        # output = rearrange(output, '(b s) f ... -> b s f ...', b=batch_size, f=seqlen, s=seqlen)

        pair_act = self.out_proj(output)

        if self.orientation == 'per_column':
            pair_act = torch.swapaxes(pair_act, -2, -3)

        return pair_act
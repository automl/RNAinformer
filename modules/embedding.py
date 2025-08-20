import math
import torch
from torch import nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

#https://github.com/tatp22/multidim-positional-encoding/blob/master/positional_encodings/torch_encodings.py
class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, max_len=5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, max_len, d_model)
        d_model = int(math.ceil(d_model / 4) * 2)
        pe_x = torch.zeros(max_len, d_model)
        pe_y = torch.zeros(max_len, d_model)
        position_x = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        position_y = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe_x[:, 0::2] = torch.sin(position_x * div_term)
        pe_x[:, 1::2] = torch.cos(position_x * div_term)
        pe_y[:, 0::2] = torch.sin(position_y * div_term)
        pe_y[:, 1::2] = torch.cos(position_y * div_term)

        pe[:,:,:d_model]= pe_x.unsqueeze(1)
        pe[:,:,d_model:]= pe_y
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe, persistent=False)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :x.size(2)]
        return x

class PositionalEncoding2D_sym(nn.Module):
    def __init__(self, d_model, max_len=5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, max_len, d_model)
        d_model = int(math.ceil(d_model / 4) * 2)
        pe_x = torch.zeros(max_len, d_model)
        pe_y = torch.zeros(max_len, d_model)
        position_x = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        position_y = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe_x[:, 0::2] = torch.sin(position_x * div_term)
        pe_x[:, 1::2] = torch.cos(position_x * div_term)
        pe_y[:, 0::2] = torch.sin(position_y * div_term)
        pe_y[:, 1::2] = torch.cos(position_y * div_term)

        pe[:,:,:d_model]= pe_x.unsqueeze(1)
        pe[:,:,d_model:]= pe_y
        i,j = torch.triu_indices(max_len, max_len)
        pe.transpose(0,1)[i,j] = pe[i,j]
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe, persistent=False)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :x.size(2)]
        return x

class Embedding(nn.Module):
    def __init__(self, embed_dim,vocab_size,dropout=0.0,padding_idx=None):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.embed(x))


class PosEmbedding(nn.Module):
    def __init__(self, vocab, model_dim, max_len, rel_pos_enc, initializer_range):

        super().__init__()

        self.rel_pos_enc = rel_pos_enc
        self.max_len = max_len

        self.embed_seq = nn.Embedding(vocab, model_dim)

        self.scale = nn.Parameter(torch.sqrt(torch.FloatTensor([model_dim // 2])), requires_grad=False)

        if rel_pos_enc:
            self.embed_pair_pos = nn.Linear(max_len, model_dim, bias=False)
        else:
            pe = torch.zeros(max_len, model_dim)
            position = torch.arange(0, max_len).unsqueeze(1).type(torch.FloatTensor)
            div_term = torch.exp(
                torch.arange(0, model_dim, 2).type(torch.FloatTensor) * -(math.log(10000.0) / model_dim))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)
        self.initialize(initializer_range) 

    def initialize(self, initializer_range):
        nn.init.normal_(self.embed_seq.weight, mean=0.0, std=initializer_range)

    def relative_position_encoding(self, src_seq):

        residue_index = torch.arange(src_seq.size()[1], device=src_seq.device).expand(src_seq.size())
        rel_pos = F.one_hot(torch.clip(residue_index, min=0, max=self.max_len - 1), self.max_len)

        if isinstance(self.embed_pair_pos.weight, torch.cuda.BFloat16Tensor):
            rel_pos = rel_pos.type(torch.bfloat16)
        elif isinstance(self.embed_pair_pos.weight, torch.cuda.HalfTensor):
            rel_pos = rel_pos.half()
        else:
            rel_pos = rel_pos.type(torch.float32)

        pos_encoding = self.embed_pair_pos(rel_pos)
        return pos_encoding

    def forward(self, src_seq):

        seq_embed = self.embed_seq(src_seq) * self.scale

        if self.rel_pos_enc:
            seq_embed = seq_embed + self.relative_position_encoding(src_seq)
        else:
            seq_embed = seq_embed + self.pe[:, :src_seq.size(1)]

        return seq_embed

class EmbedSequence2Matrix(nn.Module):
    def __init__(self, vocab_size, model_dim, config):
        super().__init__()

        self.embed_1 = PosEmbedding(vocab_size, model_dim, config.max_len,
                                        config.rel_pos_enc, config.initializer_range)
        self.embed_2 = PosEmbedding(vocab_size, model_dim, config.max_len,
                                        config.rel_pos_enc, config.initializer_range)
        self.dropout = nn.Dropout(config.embed_dropout)

    def forward(self, seq, feature_mask=None):
        seq_1_embed = self.embed_1(seq)
        seq_2_embed = self.embed_2(seq)

        pair_latent = seq_1_embed.unsqueeze(1) + seq_2_embed.unsqueeze(2)

        return self.dropout(pair_latent)
    
class EmbedMatrix(nn.Module):
    def __init__(self, vocab_size, model_dim, config):
        super().__init__()
        self.embed = Embedding(model_dim,vocab_size, config.embed_dropout)
        if config.get("sym_pos_enc", False):
            self.pos = PositionalEncoding2D_sym(model_dim, config.max_len)
        else:
            self.pos = PositionalEncoding2D(model_dim, config.max_len)

    def forward(self, seq, feature_mask=None):
        seq_embed = self.embed(seq)
        seq_embed = self.pos(seq_embed)

        return seq_embed
import torch
from torch import nn
from RNAinformer.modules.encoder import TransformerEncoder, RNAformerBlock, RNAformerBlockFlash
from RNAinformer.modules.decoder import TransformerDecoder
from RNAinformer.modules.embedding import Embedding, PositionalEncoding, EmbedMatrix, EmbedSequence2Matrix

class RiboDesignFormer(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.pos_enc = PositionalEncoding(cfg.model_dim, cfg.max_len)

        self.gc_cond = cfg.get('gc_conditioning', False)
        self.energy_cond = cfg.get('energy_conditioning', False)
        if self.gc_cond:
            self.gc_encoder = nn.Linear(1, cfg.model_dim)
        if self.energy_cond:
            self.energy_encoder = nn.Linear(1, cfg.model_dim)
            self.energy_scale = -100
            
        self.matrix = cfg.get('matrix', False)
        if self.matrix:
            model_dim = cfg.model_dim // 2
            self.src_struct_emb = EmbedMatrix(cfg.struct_vocab_size, model_dim//2, cfg)
            self.src_seq_emb = EmbedSequence2Matrix(cfg.seq_vocab_size, model_dim//2, cfg)
            if cfg.get('flash', False):
                self.matrix_enc = RNAformerBlockFlash(model_dim,cfg)
            else:
                self.matrix_enc = RNAformerBlock(model_dim,cfg)
            self.encoder = TransformerEncoder(cfg.n_layers-1, cfg.model_dim, cfg.num_head, cfg.model_dim*cfg.ff_factor, cfg.attn_dropout, cfg.resi_dropout)
        else:
            self.src_struct_emb = Embedding(cfg.model_dim//2, cfg.struct_vocab_size, cfg.embed_dropout)
            self.src_seq_emb = Embedding(cfg.model_dim//2, cfg.seq_vocab_size, cfg.embed_dropout)
            self.encoder = TransformerEncoder(cfg.n_layers, cfg.model_dim, cfg.num_head, cfg.model_dim*cfg.ff_factor, cfg.attn_dropout, cfg.resi_dropout)
        self.trg_emb = Embedding(cfg.model_dim, cfg.trg_vocab_size, cfg.embed_dropout)
        self.decoder = TransformerDecoder(cfg.n_layers, cfg.model_dim, cfg.num_head, cfg.model_dim*cfg.ff_factor, cfg.attn_dropout, cfg.resi_dropout)
        self.generator = nn.Linear(cfg.model_dim, cfg.trg_vocab_size)

    def make_src_mask(self, src, src_len):
        src_mask = torch.arange(src.shape[1], device=src.device).expand(src.shape[:2]) < src_len.unsqueeze(1)
        src_mask = src_mask.type(torch.bool).unsqueeze(1)
        return src_mask

    def make_trg_mask(self, trg, trg_len):
        mask = torch.arange(trg.size()[1], device=trg.device).expand(
            trg.shape[:2]) < trg_len.unsqueeze(1)
        mask = mask.unsqueeze(-1)
        sub_mask = torch.triu(
            torch.ones((1, trg.size()[1], trg.size()[1]), dtype=torch.bool, device=trg.device),
            diagonal=1)
        diag_mask = torch.eye(trg.size()[1], dtype=torch.bool, device=trg.device).unsqueeze(0)
        sub_mask = sub_mask == 0
        trg_mask = mask & sub_mask | diag_mask
        trg_mask = trg_mask.type(torch.bool)
        return trg_mask
    
    def make_pair_mask(self, src, src_len):
        encode_mask = torch.arange(src.shape[1], device=src.device).expand(src.shape[:2]) < src_len.unsqueeze(1)

        pair_mask = encode_mask[:, None, :] * encode_mask[:, :, None]

        assert isinstance(pair_mask, torch.BoolTensor) or isinstance(pair_mask, torch.cuda.BoolTensor)
        return torch.bitwise_not(pair_mask)
    
    
    def encode(self, src_seq, src_struct, src_mask, pair_mask=None, gc_emb=None, energy_emb=None):
        seq_emb = self.src_seq_emb(src_seq)
        struct_emb = self.src_struct_emb(src_struct)
        src_emb = torch.cat([struct_emb, seq_emb], dim=-1)
        if self.matrix:
            src_emb.masked_fill_(pair_mask[:, :, :, None], 0.0)
            src_emb = self.matrix_enc(src_emb, pair_mask)
            src_emb = torch.cat([src_emb,src_emb.transpose(1,2)], dim=-1).mean(dim=2)
            
        if gc_emb is not None:
            src_emb = src_emb + gc_emb.unsqueeze(1)
        if energy_emb is not None:
            src_emb = src_emb + energy_emb.unsqueeze(1)
        return self.encoder(self.pos_enc(src_emb), src_mask)
    
    def decode(self, trg, enc_out, trg_mask, src_mask):
        return self.decoder(self.pos_enc(self.trg_emb(trg)), enc_out, trg_mask, src_mask)
    
    def forward(self, src_seq, src_struct, seq_mask, struct_mask, trg, seq_len, gc_content=None, energy=None):
        trg = trg[:, :-1]
        trg_mask = self.make_trg_mask(trg, seq_len)
        trg = trg.masked_fill(trg == -100, 0)
        pair_mask = None
        if self.matrix:
            pair_mask = self.make_pair_mask(src_seq, seq_len)
        src_mask = self.make_src_mask(src_struct, seq_len)
        gc_emb = self.gc_encoder(gc_content.unsqueeze(1)) if self.gc_cond and gc_content is not None else None
        energy_emb = self.energy_encoder(energy.unsqueeze(1) / self.energy_scale) if self.energy_cond and energy is not None else None
        enc_out = self.encode(src_seq, src_struct, src_mask, pair_mask, gc_emb, energy_emb)
        dec_out = self.decode(trg, enc_out, trg_mask, src_mask)
        return self.generator(dec_out)
    
    def generate(self, src_seq, src_struct, seq_mask, struct_mask, seq_len, gc_content=None, energy = None, max_len=100, greedy=False, constrained_generation=True):
        src_mask = self.make_src_mask(src_struct, seq_len)
        pair_mask = None
        if self.matrix:
            pair_mask = self.make_pair_mask(src_seq, seq_len)
        
        gc_emb = self.gc_encoder(gc_content.unsqueeze(1)) if self.gc_cond and gc_content is not None else None
        energy_emb = self.energy_encoder(energy.unsqueeze(1) / self.energy_scale) if self.energy_cond and energy is not None else None
        enc_out = self.encode(src_seq, src_struct, src_mask, pair_mask, gc_emb, energy_emb)
        trg = torch.zeros(src_seq.shape[0], 1, dtype=torch.long, device=src_seq.device)
        max_len = seq_len.max()
        for i in range(max_len):
            trg_mask = self.make_trg_mask(trg, torch.ones(trg.shape[0], dtype=torch.long, device=src_seq.device)*(i+1))
            dec_out = self.decode(trg, enc_out, trg_mask, src_mask)
            dec_out = self.generator(dec_out[:, -1, :]).softmax(dim=-1)
            if greedy:
                dec_out = torch.argmax(dec_out, dim=-1).unsqueeze(1)
            else:
                dec_out = torch.multinomial(dec_out, num_samples=1)
            if constrained_generation:
                dec_out[seq_mask[:, i] == 1] = src_seq[seq_mask[:, i] == 1, i].unsqueeze(1)
            trg = torch.cat([trg, dec_out], dim=1)
        return trg[:, 1:]
    
    
    

        
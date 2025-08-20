from torch import nn
from RNAinformer.modules.attention import MultiheadSelfAttention, MultiheadCrossAttention

class DecoderBlock(nn.Module):

    def __init__(self, input_dim, num_heads, dim_feedforward, attn_dropout=0.0, dropout=0.0):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Attention layer
        self.self_attn = MultiheadSelfAttention(input_dim, input_dim, num_heads, attn_dropout)

        # Cross-attention layer
        self.cross_attn = MultiheadCrossAttention(input_dim, input_dim, num_heads, attn_dropout)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim)
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.norm3 = nn.LayerNorm(input_dim)
        self.norm4 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_out, src_mask=None, trg_mask=None):
        # Self-attention part
        x = self.norm1(x)
        self_attn_out = self.self_attn(x, mask=trg_mask)
        x = x + self.dropout(self_attn_out)
        # Cross-attention part
        x = self.norm2(x)
        encoder_out = self.norm3(encoder_out)
        cross_attn_out = self.cross_attn(x, encoder_out, mask=src_mask)
        x = x + self.dropout(cross_attn_out)

        # MLP part
        x = self.norm4(x)
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)

        return x
    

class TransformerDecoder(nn.Module):

    def __init__(self, num_layers, hidden_dim, num_heads, dim_feedforward, attn_dropout=0.0, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([DecoderBlock(hidden_dim, num_heads, dim_feedforward, attn_dropout, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x, enc_out, trg_mask=None, src_mask=None):
        for l in self.layers:
            x = l(x, enc_out, trg_mask=trg_mask, src_mask=src_mask)
        return self.norm(x)

        
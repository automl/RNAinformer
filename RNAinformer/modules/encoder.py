from torch import nn
from RNAinformer.modules.attention import MultiheadSelfAttention
from RNAinformer.modules.axial_attention import TriangleAttention, AxialFlashAttention
from RNAinformer.modules.feed_forward import FeedForward, ConvFeedForward

class EncoderBlock(nn.Module):

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
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention part
        x = self.norm1(x)
        attn_out = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out)

        # MLP part
        x = self.norm2(x)
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)

        return x

class TransformerEncoder(nn.Module):

    def __init__(self, num_layers, hidden_dim, num_heads, dim_feedforward, attn_dropout=0.0, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(hidden_dim, num_heads, dim_feedforward, attn_dropout, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x, mask=None):
        for l in self.layers:
            x = l(x, mask=mask)
        return self.norm(x)

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for l in self.layers:
            _, attn_map = l.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = l(x)
        return attention_maps
    


class RNAformerBlock(nn.Module):

    def __init__(self,model_dim, config):
        super().__init__()

        #model_dim = config.model_dim // 2
        #model_dim = config.model_dim
        ff_dim = int(config.ff_factor * model_dim)

        self.attn_pair_row = TriangleAttention(model_dim, config.num_head, 'per_row', config.softmax_scale,
                                               config.precision, config.zero_init, config.use_bias, config.flash_attn,
                                               config.initializer_range, config.n_layers)
        self.attn_pair_col = TriangleAttention(model_dim, config.num_head, 'per_column', config.softmax_scale,
                                               config.precision, config.zero_init, config.use_bias, config.flash_attn,
                                               config.initializer_range, config.n_layers)

        self.pair_dropout_row = nn.Dropout(p=config.attn_dropout / 2)
        self.pair_dropout_col = nn.Dropout(p=config.attn_dropout / 2)

        if config.ff_kernel:
            self.pair_transition = ConvFeedForward(model_dim, ff_dim, use_bias=config.use_bias,
                                                   kernel=config.ff_kernel,
                                                   initializer_range=config.initializer_range,
                                                   zero_init=config.zero_init,
                                                   n_layers=config.n_layers)
        else:
            self.pair_transition = FeedForward(model_dim, ff_dim, use_bias=config.use_bias, glu=config.use_glu,
                                               initializer_range=config.initializer_range, zero_init=config.zero_init,
                                               n_layers=config.n_layers)

        self.res_dropout = nn.Dropout(p=config.resi_dropout)

    def forward(self, pair_act, pair_mask, cycle_infer=False):

        pair_act = pair_act + self.pair_dropout_row(self.attn_pair_row(pair_act, pair_mask, cycle_infer))
        pair_act = pair_act + self.pair_dropout_col(self.attn_pair_col(pair_act, pair_mask, cycle_infer))
        pair_act = pair_act + self.res_dropout(self.pair_transition(pair_act))

        return pair_act

class RNAformerBlockFlash(nn.Module):

    def __init__(self, model_dim, config):
        super().__init__()
        ff_dim = int(config.ff_factor * model_dim)
        self.norm_row = nn.LayerNorm(model_dim)
        self.norm_col = nn.LayerNorm(model_dim)
        self.norm_trans = nn.LayerNorm(model_dim)
        self.attn_pair_row = AxialFlashAttention(model_dim, config, orientation='per_row')
        self.attn_pair_col = AxialFlashAttention(model_dim, config, orientation='per_column')

       

        self.pair_dropout_row = nn.Dropout(p=config.resi_dropout / 2)
        self.pair_dropout_col = nn.Dropout(p=config.resi_dropout / 2)
        self.res_dropout = nn.Dropout(p=config.resi_dropout)

        if config.ff_kernel:
            self.pair_transition = ConvFeedForward(model_dim, ff_dim, use_bias=config.use_bias,
                                                   kernel=config.ff_kernel,
                                                   initializer_range=config.initializer_range,
                                                   zero_init=config.zero_init,
                                                   n_layers=config.n_layers)
        else:
            self.pair_transition = FeedForward(model_dim, ff_dim, use_bias=config.use_bias, glu=config.use_glu,
                                               initializer_range=config.initializer_range, zero_init=config.zero_init,
                                               n_layers=config.n_layers)


    def forward(self, pair_act, pair_mask):

        pair_act = pair_act + self.pair_dropout_row(self.attn_pair_row(self.norm_row(pair_act), pair_mask))
        pair_act = pair_act + self.pair_dropout_col(self.attn_pair_col(self.norm_col(pair_act), pair_mask))


        pair_act = pair_act + self.res_dropout(self.pair_transition(self.norm_trans(pair_act)))

        return pair_act
    
class RNAformerStack(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.output_ln = nn.LayerNorm(config.model_dim)

        module_list = []
        for idx in range(config.n_layers):
            layer = RNAformerBlock(config=config)
            module_list.append(layer)
        self.layers = nn.ModuleList(module_list)

    def forward(self, pair_act, pair_mask, cycle_infer=False):

        for idx, layer in enumerate(self.layers):
            pair_act = layer(pair_act, pair_mask, cycle_infer=cycle_infer)

        pair_act = self.output_ln(pair_act)

        return pair_act
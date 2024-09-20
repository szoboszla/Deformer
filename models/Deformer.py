import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer, Decoder0, LSTM
from layers.Autoformer_EncDec import moving_avg
from layers.SelfAttention_Family import FullAttention, AttentionLayer, ProbAttention, DSAttention
from layers.Embed import DataEmbedding, PositionalEmbedding
import numpy as np
from layers.RevIN import RevIN
from layers.TransformerBlocks import Encoder2
from layers.Autoformer_EncDec import series_decomp,series_decomp_multi
import numpy as np
from scipy.linalg import hankel
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        configs.d_model = configs.seq_len
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.Linear = nn.Sequential()
        self.Linear.add_module('Linear',nn.Linear(configs.seq_len, self.pred_len))
        self.revin_layer = RevIN(configs.enc_in)

        self.decompsition = series_decomp(kernel_size=configs.moving_avg)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        x_enc = self.revin_layer(x_enc, 'norm')
        enc_out = x_enc.permute(0, 2, 1)
        dec_out1, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        dec_out1 = dec_out1.permute(0, 2, 1)

        dec_out = self.revin_layer(dec_out1[:, -self.pred_len:, :], 'denorm')
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]



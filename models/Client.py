import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.Autoformer_EncDec import moving_avg
from layers.SelfAttention_Family import FullAttention, AttentionLayer, ProbAttention, DSAttention
from layers.Embed import DataEmbedding
import numpy as np
from layers.RevIN import RevIN


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
        self.proj = nn.Linear(configs.d_model, self.pred_len, bias=True)
        self.Linear = nn.Sequential()
        self.Linear.add_module('Linear',nn.Linear(configs.seq_len, self.pred_len))
        self.w_dec = torch.nn.Parameter(torch.FloatTensor([configs.w_lin]*configs.enc_in),requires_grad=True)
        self.revin_layer = RevIN(configs.enc_in)


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        x_enc = self.revin_layer(x_enc, 'norm')
        enc_out = x_enc.permute(0, 2, 1)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        dec_out = self.proj(enc_out)
        dec_out = dec_out.permute(0, 2, 1)
        linear_out = self.Linear(x_enc.permute(0,2,1)).permute(0,2,1)
        dec_out = self.revin_layer(dec_out[:, -self.pred_len:, :]+self.w_dec*linear_out, 'denorm')

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out

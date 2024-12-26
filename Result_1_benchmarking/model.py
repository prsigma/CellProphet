import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from einops import rearrange
import os

class FullAttention(nn.Module):
    def __init__(self, attention_dropout):
        super(FullAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, no_tf_genes_index,tau=None, delta=None):
        B, T, L, H, E = queries.shape
        _, T, S, _, D = values.shape
        scale = 1. / sqrt(E)

        scores = torch.einsum("btlhe,btshe->bthls", queries, keys)
        
        if no_tf_genes_index is not None:
            scores.masked_fill_(attn_mask, -np.inf)
            scores[:,:,no_tf_genes_index,no_tf_genes_index] = 1.0
        
        else:
            scores.masked_fill_(attn_mask, -np.inf)

        attention = torch.softmax(scale * scores, dim=-1)

        A = self.dropout(attention)
        V = torch.einsum("bthls,btshd->btlhd", A, values)

        return (V.contiguous(), A)
        
class AttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = FullAttention(dropout)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, no_tf_genes_index,tau=None, delta=None):
        B, T, L, _ = queries.shape
        _, _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, T, L, H, -1)
        keys = self.key_projection(keys).view(B, T, S, H,-1)
        values = self.value_projection(values).view(B, T, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            no_tf_genes_index,
            tau=tau,
            delta=delta
        )

        out = out.view(B, T, L, -1)
        out = self.out_projection(out)

        return out, attn

class TimeLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super(TimeLayer, self).__init__()
        self.time_attention = AttentionLayer(d_model, n_heads,dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.MLP = nn.Sequential(nn.Linear(d_model, d_ff),
                                nn.GELU(),
                                nn.Linear(d_ff, d_model))
        
    def forward(self, x):
        batch = x.shape[0]
        cell = x.shape[-2]

        time_mask = torch.ones((cell,cell))
        time_mask = torch.triu(time_mask, diagonal=1).to(device=x.device).to(torch.bool)

        time_enc, _ = self.time_attention(
            x, x, x, time_mask, None
        )

        out = x + self.dropout(time_enc)
        out = self.norm1(out)
        out = out + self.dropout(self.MLP(out))

        final_out = self.norm2(out)

        return final_out

class SpatialLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout) -> None:
        super().__init__()
        self.heads = n_heads
        self.spatial_attention = AttentionLayer(d_model, n_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.MLP = nn.Sequential(nn.Linear(d_model, d_ff),
                                nn.GELU(),
                                nn.Linear(d_ff, d_model))
    
    def forward(self,x,prior_mask,no_tf_genes_index):
        out, attention = self.spatial_attention(x, x, x, prior_mask,no_tf_genes_index)
        attn_scores = attention.mean(1)

        out = x + self.dropout(out)
        out = self.norm1(out)
        out = out + self.dropout(self.MLP(out))
        out = self.norm2(out)

        return out, attn_scores
        
class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.time_encoder = TimeLayer(d_model, heads, d_ff, dropout)
        self.spatial_encoder = SpatialLayer(d_model, heads, d_ff, dropout)

    def forward(self, x, prior_mask,no_tf_genes_index):
        time_in = rearrange(x, 'b cell gene d_model -> b gene cell d_model')
        time_out = self.time_encoder(time_in)
        time_out = rearrange(time_out, 'b gene cell d_model -> b cell gene d_model')
        spatial_out,spatial_attention = self.spatial_encoder(time_out,prior_mask,no_tf_genes_index)
        return spatial_out, spatial_attention
        
class Encoder(nn.Module):
    def __init__(self, encoder_layers):
        super(Encoder, self).__init__()
        self.encoder_layers = nn.ModuleList(encoder_layers)

    def forward(self, x, prior_mask,no_tf_genes_index):
        for encoder_layer in self.encoder_layers:
            x, spatial_attention = encoder_layer(x, prior_mask,no_tf_genes_index)

        return x, spatial_attention

class MTGRN(nn.Module):
    def __init__(self, in_len, out_len, d_model, d_ff, n_heads, dropout, e_layers):
        super(MTGRN, self).__init__()
        self.in_len = in_len
        self.out_len = out_len
        self.d_model = d_model
        self.d_ff = d_ff
        self.heads = n_heads
        self.dropout = dropout
        self.e_layers = e_layers

        # Embedding
        self.embedding = nn.Linear(1, self.d_model)
        self.pre_norm = nn.LayerNorm(self.d_model)

        self.pooling = nn.AdaptiveAvgPool1d(1)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    self.d_model, self.heads, self.d_ff, self.dropout
                ) for _ in range(self.e_layers)
            ]
        )

        self.projector = nn.Linear(self.in_len, self.out_len, bias=True)
    
    def forward(self, x, prior_mask,no_tf_genes_index):
        x = x.unsqueeze(-1)  # Shape becomes [cell, gene, 1]
        gene_num = x.shape[-2]
        x = self.embedding(x)  # Now x has shape [cell, gene, d_model]
        x = self.pre_norm(x) 
        
        x, spatial_attention = self.encoder(x,prior_mask,no_tf_genes_index)   #input [cell, gene]
        x = rearrange(x, 'b cell gene d_model -> (b gene) cell d_model')
        x = self.pooling(x).squeeze(-1) 
        x = rearrange(x,'(b gene) cell -> b gene cell', gene=gene_num)
        out = self.projector(x)

        out = rearrange(out, 'b gene time -> b time gene')
        return out, spatial_attention
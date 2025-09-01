import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from einops import rearrange
import os

class FullAttention(nn.Module):
    """
    Full attention mechanism that computes attention scores between all query-key pairs.
    Implements scaled dot-product attention with optional masking.
    """
    def __init__(self, attention_dropout):
        """
        Initialize the FullAttention module.
        
        Args:
            attention_dropout: Dropout rate for attention weights
        """
        super(FullAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, no_tf_genes_index,tau=None, delta=None):
        """
        Compute full attention between queries, keys, and values.
        
        Args:
            queries: Query tensor [B, T, L, H, E]
            keys: Key tensor [B, T, S, H, E]
            values: Value tensor [B, T, S, H, D]
            attn_mask: Attention mask to control gene interactions
            no_tf_genes_index: Index of genes which have no interaction with TFs
            
        Returns:
            tuple: (attention_output, attention_weights)
        """
        B, T, L, H, E = queries.shape
        _, T, S, _, D = values.shape
        scale = 1. / sqrt(E)

        scores = torch.einsum("btlhe,btshe->bthls", queries, keys)
        
        # if no_tf_genes_index is not None:
        #     scores.masked_fill_(attn_mask, -np.inf)
        #     scores[:,:,:,no_tf_genes_index,no_tf_genes_index] = 1.0
        
        # else:
        #     scores.masked_fill_(attn_mask, -np.inf)

        scores = scores * (attn_mask * 0.1 + (~attn_mask) * 0.9)    #soft mask

        attention = torch.softmax(scale * scores, dim=-1)

        A = self.dropout(attention)
        V = torch.einsum("bthls,btshd->btlhd", A, values)

        return (V.contiguous(), A)
        
class AttentionLayer(nn.Module):
    """
    Multi-head attention layer that projects inputs to queries, keys, and values,
    then applies full attention mechanism.
    """
    def __init__(self, d_model, n_heads, dropout, d_keys=None,
                 d_values=None):
        """
        Initialize the attention layer.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout rate
            d_keys: Dimension of keys (defaults to d_model // n_heads)
            d_values: Dimension of values (defaults to d_model // n_heads)
        """
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
        """
        Forward pass of the attention layer.
        
        Args:
            queries: Query tensor [B, T, L, d_model]
            keys: Key tensor [B, T, S, d_model]
            values: Value tensor [B, T, S, d_model]
            attn_mask: Attention mask
            no_tf_genes_index: Index of genes which have no interaction with TFs
            
        Returns:
            tuple: (output, attention_weights)
        """
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
    """
    Temporal attention layer that models dependencies across time steps.
    Uses causal masking to prevent information leakage from future time steps.
    """
    def __init__(self, d_model, n_heads, d_ff, dropout):
        """
        Initialize the time layer.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward network dimension
            dropout: Dropout rate
        """
        super(TimeLayer, self).__init__()
        self.time_attention = AttentionLayer(d_model, n_heads,dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.MLP = nn.Sequential(nn.Linear(d_model, d_ff),
                                nn.GELU(),
                                nn.Linear(d_ff, d_model))
        
    def forward(self, x):
        """
        Forward pass of the time layer.
        
        Args:
            x: Input tensor [batch, gene, time, d_model]
            
        Returns:
            tensor: Output tensor with temporal dependencies modeled
        """
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
    """
    Spatial attention layer that models gene regulatory interactions.
    Uses gene regulatory network priors to guide attention patterns.
    """
    def __init__(self, d_model, n_heads, d_ff, dropout) -> None:
        """
        Initialize the spatial layer.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward network dimension
            dropout: Dropout rate
        """
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
        """
        Forward pass of the spatial layer.
        
        Args:
            x: Input tensor [batch, time, gene, d_model]
            prior_mask: Gene regulatory network prior mask
            no_tf_genes_index: Index of genes which have no interaction with TFs
            
        Returns:
            tuple: (output_tensor, attention_scores)
        """
        out, attention = self.spatial_attention(x, x, x, prior_mask,no_tf_genes_index)
        attn_scores = attention.mean(1)

        out = x + self.dropout(out)
        out = self.norm1(out)
        out = out + self.dropout(self.MLP(out))
        out = self.norm2(out)

        return out, attn_scores
        
class EncoderLayer(nn.Module):
    """
    Combined encoder layer that applies both temporal and spatial attention.
    First models temporal dependencies, then gene regulatory interactions.
    """
    def __init__(self, d_model, heads, d_ff, dropout):
        """
        Initialize the encoder layer.
        
        Args:
            d_model: Model dimension
            heads: Number of attention heads
            d_ff: Feed-forward network dimension
            dropout: Dropout rate
        """
        super(EncoderLayer, self).__init__()
        self.time_encoder = TimeLayer(d_model, heads, d_ff, dropout)
        self.spatial_encoder = SpatialLayer(d_model, heads, d_ff, dropout)

    def forward(self, x, prior_mask,no_tf_genes_index):
        """
        Forward pass of the encoder layer.
        
        Args:
            x: Input tensor [batch, time, gene, d_model]
            prior_mask: Gene regulatory network prior mask
            no_tf_genes_index: Index of genes which have no interaction with TFs
            
        Returns:
            tuple: (encoded_output, spatial_attention_scores)
        """
        time_in = rearrange(x, 'b cell gene d_model -> b gene cell d_model')
        time_out = self.time_encoder(time_in)
        time_out = rearrange(time_out, 'b gene cell d_model -> b cell gene d_model')
        spatial_out,spatial_attention = self.spatial_encoder(time_out,prior_mask,no_tf_genes_index)
        return spatial_out, spatial_attention
        
class Encoder(nn.Module):
    """
    Stack of encoder layers that processes input through multiple 
    temporal and spatial attention layers.
    """
    def __init__(self, encoder_layers):
        """
        Initialize the encoder.
        
        Args:
            encoder_layers: List of EncoderLayer modules
        """
        super(Encoder, self).__init__()
        self.encoder_layers = nn.ModuleList(encoder_layers)

    def forward(self, x, prior_mask,no_tf_genes_index):
        """
        Forward pass through all encoder layers.
        
        Args:
            x: Input tensor
            prior_mask: Gene regulatory network prior mask
            no_tf_genes_index: Index of genes which have no interaction with TFs
            
        Returns:
            tuple: (final_output, final_spatial_attention)
        """
        for encoder_layer in self.encoder_layers:
            x, spatial_attention = encoder_layer(x, prior_mask,no_tf_genes_index)

        return x, spatial_attention

class CellProphet(nn.Module):
    """
    CellProphet: A transformer-based model for gene regulatory network inference 
    and gene expression prediction. Combines temporal and spatial attention 
    mechanisms to model gene expression dynamics.
    """
    def __init__(self, in_len, out_len, d_model, d_ff, n_heads, dropout, e_layers):
        """
        Initialize the CellProphet model.
        
        Args:
            in_len: Input sequence length (number of time points)
            out_len: Output sequence length (prediction horizon)
            d_model: Model dimension
            d_ff: Feed-forward network dimension
            n_heads: Number of attention heads
            dropout: Dropout rate
            e_layers: Number of encoder layers
        """
        super(CellProphet, self).__init__()
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
        """
        Forward pass of the CellProphet model.
        
        Args:
            x: Input gene expression tensor [batch, time, gene]
            prior_mask: Gene regulatory network prior mask
            no_tf_genes_index: Index of genes which have no interaction with TFs
            
        Returns:
            tuple: (predicted_expression, spatial_attention_weights)
                - predicted_expression: [batch, out_len, gene]
                - spatial_attention_weights: Attention scores for GRN inference
        """
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
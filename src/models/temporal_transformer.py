import math
import torch
import torch.nn as nn
from typing import Optional


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Seq_len, d_model)
        return x + self.pe[:, : x.size(1)]


class TemporalTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # src: (B * N, T_in, d_model)
        src2 = self.norm1(src)
        attn_out, _ = self.self_attn(src2, src2, src2)
        src = src + self.dropout1(attn_out)

        src2 = self.norm2(src)
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(ff_out)
        return src


class TemporalTransformer(nn.Module):
    def __init__(
        self,
        num_nodes: int = 207,
        in_features: int = 2,
        in_steps: int = 12,
        out_steps: int = 12,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        d_ff: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.d_model = d_model

        self.input_proj = nn.Linear(in_features, d_model)
        self.node_embedding = nn.Parameter(torch.empty(num_nodes, d_model))
        nn.init.xavier_uniform_(self.node_embedding)

        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=in_steps + 10)

        self.layers = nn.ModuleList([
            TemporalTransformerEncoderLayer(d_model=d_model, nhead=nhead, d_ff=d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        
        # Temporal projection: (T_in * d_model) -> out_steps
        self.head = nn.Sequential(
            nn.Linear(in_steps * d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, out_steps)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: (B, T_in, N, F)
        B, T, N, F = x.shape

        # Linear projection: (B, T, N, d_model)
        h = self.input_proj(x)

        # Add spatial node embedding
        h = h + self.node_embedding.view(1, 1, N, self.d_model)

        # Reshape to (B * N, T, d_model) for parallel temporal attention across all sensors
        h = h.permute(0, 2, 1, 3).contiguous().view(B * N, T, self.d_model)

        # Add temporal positional encoding
        h = self.pos_encoder(h)

        # Transformer encoder layers
        for layer in self.layers:
            h = layer(h)

        h = self.norm(h)

        # Flatten time and feature dimension: (B * N, T * d_model)
        h = h.view(B * N, T * self.d_model)

        # Project to out_steps: (B * N, out_steps)
        out = self.head(h)

        # Reshape back to (B, out_steps, N)
        out = out.view(B, N, self.out_steps).permute(0, 2, 1).contiguous()
        return out

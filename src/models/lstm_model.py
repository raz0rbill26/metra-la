import torch
import torch.nn as nn
from typing import Optional


class MultiLayerLSTM(nn.Module):
    def __init__(
        self,
        num_nodes: int = 207,
        in_features: int = 2,
        in_steps: int = 12,
        out_steps: int = 12,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Linear(in_features, hidden_dim)
        self.node_embedding = nn.Parameter(torch.empty(num_nodes, hidden_dim))
        nn.init.xavier_uniform_(self.node_embedding)

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.norm = nn.LayerNorm(hidden_dim)

        self.head = nn.Sequential(
            nn.Linear(in_steps * hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, out_steps)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (B, in_steps, num_nodes, in_features)
        B, T, N, F = x.shape

        # Linear projection: (B, T, N, hidden_dim)
        h = self.input_proj(x)

        # Add spatial node embedding
        h = h + self.node_embedding.view(1, 1, N, self.hidden_dim)

        # Reshape to (B * N, T, hidden_dim) for parallel sequential modeling across sensors
        h = h.permute(0, 2, 1, 3).contiguous().view(B * N, T, self.hidden_dim)

        # LSTM forward pass
        lstm_out, _ = self.lstm(h) # (B * N, T, hidden_dim)
        lstm_out = self.norm(lstm_out)

        # Flatten sequence and hidden dimension: (B * N, T * hidden_dim)
        lstm_out = lstm_out.contiguous().view(B * N, T * self.hidden_dim)

        # Project to out_steps: (B * N, out_steps)
        out = self.head(lstm_out)

        # Reshape back to (B, out_steps, num_nodes)
        out = out.view(B, N, self.out_steps).permute(0, 2, 1).contiguous()
        return out

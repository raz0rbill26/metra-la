import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 2, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B * N, C, T)
        out = self.conv(x)
        if self.padding != 0:
            out = out[:, :, :-self.padding]
        return out


class DilatedResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 2, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        self.filter_conv = CausalConv1d(channels, channels, kernel_size, dilation)
        self.gate_conv = CausalConv1d(channels, channels, kernel_size, dilation)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B * N, C, T)
        filter_out = torch.tanh(self.filter_conv(x))
        gate_out = torch.sigmoid(self.gate_conv(x))
        gated = filter_out * gate_out
        gated = self.dropout(gated)
        proj = self.proj(gated)
        return self.norm(x + proj)


class Dilated1DCNN(nn.Module):
    def __init__(
        self,
        num_nodes: int = 207,
        in_features: int = 2,
        in_steps: int = 12,
        out_steps: int = 12,
        channels: int = 64,
        kernel_size: int = 2,
        dilations: List[int] = [1, 2, 4, 8],
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.channels = channels

        self.input_proj = nn.Linear(in_features, channels)
        self.node_embedding = nn.Parameter(torch.empty(num_nodes, channels))
        nn.init.xavier_uniform_(self.node_embedding)

        self.blocks = nn.ModuleList([
            DilatedResidualBlock(channels=channels, kernel_size=kernel_size, dilation=d, dropout=dropout)
            for d in dilations
        ])

        self.head = nn.Sequential(
            nn.Conv1d(channels, channels * 2, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels * 2, channels, kernel_size=1),
            nn.GELU()
        )

        self.fc_out = nn.Linear(in_steps * channels, out_steps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (B, in_steps, num_nodes, in_features)
        B, T, N, F = x.shape

        # Linear feature projection: (B, T, N, channels)
        h = self.input_proj(x)

        # Add spatial node embedding
        h = h + self.node_embedding.view(1, 1, N, self.channels)

        # Reshape to (B * N, channels, T) for 1D temporal convolution
        h = h.permute(0, 2, 3, 1).contiguous().view(B * N, self.channels, T)

        # Pass through dilated causal convolution blocks
        for block in self.blocks:
            h = block(h)

        h = self.head(h) # (B * N, channels, T)

        # Flatten temporal and channel dimensions: (B * N, T * channels)
        h = h.view(B * N, T * self.channels)

        # Project to out_steps: (B * N, out_steps)
        out = self.fc_out(h)

        # Reshape back to (B, out_steps, num_nodes)
        out = out.view(B, N, self.out_steps).permute(0, 2, 1).contiguous()
        return out

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.workspace.state import WorkspaceState, N_SLOTS, K_MEM, D


class WorkspaceUpdateTransformer(nn.Module):
    """4-layer 4-head transformer implementing W_{t+1} = f_theta(W_t, Z_t).

    At each step:
      1. Concatenate [workspace_tokens; input_tokens]
      2. Self-attend over the combined sequence
      3. Slice workspace portion back out → new object slots

    Args:
        d_model: token dimensionality (must match workspace D=256)
        nhead: number of attention heads
        num_layers: transformer encoder depth
        ffn_dim: feed-forward hidden size
        dropout: dropout probability
    """

    def __init__(
        self,
        d_model: int = D,
        nhead: int = 4,
        num_layers: int = 4,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Learnable per-slot embeddings give each slot a unique identity
        self.slot_embeddings = nn.Embedding(N_SLOTS, d_model)
        self.mem_embeddings = nn.Embedding(K_MEM, d_model)
        # Project input tokens to workspace dim when dimensions differ
        self.input_proj = nn.Linear(d_model, d_model)

    def forward(self, workspace: WorkspaceState, input_tokens: Tensor) -> WorkspaceState:
        """Update workspace given new input tokens.

        Args:
            workspace: current WorkspaceState (not mutated)
            input_tokens: [B, L, D] — encoded input representations

        Returns:
            New WorkspaceState with updated objects and memory_slots.
        """
        B, L, _ = input_tokens.shape
        device = input_tokens.device

        # Bring workspace tensors to the same device as input
        ws_objects = workspace.objects.to(device)        # [N, D]
        ws_memory = workspace.memory_slots.to(device)    # [K, D]

        # Add batch dim and inject slot-specific positional embeddings
        slot_ids = torch.arange(N_SLOTS, device=device)
        mem_ids = torch.arange(K_MEM, device=device)
        ws_tokens = (ws_objects + self.slot_embeddings(slot_ids)).unsqueeze(0).expand(B, -1, -1)
        mem_tokens = (ws_memory + self.mem_embeddings(mem_ids)).unsqueeze(0).expand(B, -1, -1)

        z = self.input_proj(input_tokens)  # [B, L, D]

        # Concatenate: [workspace_slots | memory_slots | input]
        combined = torch.cat([ws_tokens, mem_tokens, z], dim=1)  # [B, N+K+L, D]

        out = self.transformer(combined)  # [B, N+K+L, D]

        new_objects = out[:, :N_SLOTS, :]            # [B, N, D]
        new_memory = out[:, N_SLOTS:N_SLOTS + K_MEM, :]  # [B, K, D]

        # Squeeze batch dim (single-sample update; take mean over batch if B>1)
        if B == 1:
            new_objects = new_objects.squeeze(0)   # [N, D]
            new_memory = new_memory.squeeze(0)     # [K, D]
        else:
            new_objects = new_objects.mean(0)
            new_memory = new_memory.mean(0)

        new_state = workspace.clone()
        new_state.objects = new_objects
        new_state.memory_slots = new_memory
        return new_state
